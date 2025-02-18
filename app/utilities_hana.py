from datetime import datetime
from hana_ml.dataframe import ConnectionContext
import pandas as pd

from hana_ml.algorithms.pal.decomposition import CATPCA
from hana_ml.algorithms.pal.tsne import TSNE
from hana_ml.algorithms.pal.clustering import DBSCAN, HDBSCAN, KMeans

from gen_ai_hub.proxy.native.openai import chat

def kmeans_and_tsne(connection,                                     ## Hana ConnectionContext
                    table_name,                                     ## Name of the table containing the projects
                    result_table_name,                              ## Name for results with clusters
                    n_components,                                   ## number of components for the PCA algorithm (64 for instance)
                    perplexity= 5,                                  ## perplexity for T-SNE algorithm  
                    start_date='1900-01-01',                        ## first date for projects to be considered in the analysis
                    end_date=datetime.now().strftime('%Y-%m-%d')    ## last date for projects to be considered in the analysis
                    ): 
    
    # Retrieve data from the table we want to analyze
    hdf=connection.table(table_name)
    
    # Dimenstionality reduction. Reduce embeddings to a lower dimension (n_components)
    cpc = CATPCA(scaling=True,
                 thread_ratio=0.9,
                 scores=True,
                 n_components=n_components,
                 component_tol=1e-5)
    
    cpc.fit(data=hdf.select(['project_number','topic_embedding']), key='project_number') 
    
    # Categorial PCA, outputs the individual component dimensions seperately, not as single vector
    # The individual component dimensions are output as rows, therefore require to be transposed into columns for further analysis
    compl_pcavecs_pivot = cpc.scores_.pivot_table(columns = 'COMPONENT_ID',
    values = 'COMPONENT_SCORE',
    index = 'project_number',
    aggfunc = 'AVG')

    # Display embeddings in a 2-dim space with T-SNE algorithm
    tsne = TSNE(n_iter = 5000,
            random_state = 1,
            n_components = 2,
            angle = 0.0,
            exaggeration = 20,
            learning_rate = 10,
            perplexity = perplexity,
            object_frequency = 50,
            thread_ratio = 0.5)

    df_tsne_res, stats, obj = tsne.fit_predict(
                                   data = compl_pcavecs_pivot, 
                                   key = 'project_number')
    
    ## Add project date in pca results
    compl_pcavecs_pivot = compl_pcavecs_pivot.set_index("project_number").join(hdf.select(['project_number','project_date']).set_index("project_number"))
    
    ## Filter projects by dates
    filter_str=f" PROJECT_DATE >= TO_DATE('{start_date}') AND PROJECT_DATE < TO_DATE('{end_date}')"
    compl_pcavecs_pivot = compl_pcavecs_pivot.rename_columns({'project_date' : 'PROJECT_DATE'}).filter(filter_str).deselect('PROJECT_DATE')

    # Run the clustering algorithm on the filtered data
    km = KMeans(n_clusters_min=5, n_clusters_max=10, max_iter=5000, distance_level='euclidean')    
    df_clusters  = km.fit_predict(data=compl_pcavecs_pivot, key='project_number')

     # Merge Cluster Results with T-SNE data
    df_clusters_1 = df_clusters.select('project_number', 'CLUSTER_ID','DISTANCE').rename_columns({'project_number' : 'PROJECT_NUMBER_1'})
    df_tsne_with_cluster = df_tsne_res.alias('TSNE').rename_columns({'project_number' : 'PROJECT_NUMBER'}).join(other = df_clusters_1.alias('CLST'),
                                                      condition = 'PROJECT_NUMBER = PROJECT_NUMBER_1' ).drop('PROJECT_NUMBER_1')
    # Collect the results
    df_tsne_with_cluster.collect()
    
    df_tsne_with_cluster.save(result_table_name, force=True )

    # Profiling and labeling clusters
    clustered_df = pd.merge(hdf.deselect(['topic_embedding','solution_embedding']).rename_columns({'project_number' : 'PROJECT_NUMBER'}).collect(), df_tsne_with_cluster.collect(), how='left',on='PROJECT_NUMBER') 
    
    profiling_string=""
    for name, group in clustered_df.groupby('CLUSTER_ID'):
        profiling_string+=f"CLUSTER {name}\n"
        group.sort_values(by='DISTANCE', ascending=True)
        most_representative_topics=group.reindex().topic.tolist()
        for d in most_representative_topics[:20]:
            profiling_string+=f"- {d}\n"

    generated_labels=label_clusters(profiling_string)
    
    # Extracting dictionary of clusters names
    clusters_labels =  [ x.split(':') for x in generated_labels.split('CLUSTER')]
    clusters_labels= [i for i in clusters_labels if len(i)==2]
    clusters_dict={}
    for c in clusters_labels:
        key=c[0].strip()
        key="{:0.0f}".format(float(key))
        clusters_dict[key]=c[1]
    clusters_dict
    
    return df_tsne_with_cluster, clusters_dict


def label_clusters(profiling_string):
    prompt=f"You will help to analyze the result of a machine learnirn algorithm for clustering on text data. \
    The algorithm was used to find clusters in topics of customer advisory servises around various services of the SAP Cloud platform (BTP). \
    For each cluster, find a good label based on the topics of a few datapoint samples.\
    Return the output in the following format: \n CLUSTER 1 : label CLUSTER 2 : label ....\ \n do not add anythng other than clusters named and labels\
    \n{profiling_string} "
    messages = [{"role": "user", "content": prompt} ]
    kwargs = dict(model_name='gpt-4o', messages=messages)
    response = chat.completions.create(**kwargs)
    return response.to_dict()["choices"][0]["message"]["content"].strip()

#Perform a vector search on the table using the specified metric and return the top k results
def run_vector_search(cc: ConnectionContext,\
                      query: str, \
                      k, \
                      table_name, \
                      vector_col, \
                      columns_to_return):
    
    cursor = cc.connection.cursor()
    
    return_columns_string = '''  '''
    for c in columns_to_return:
       return_columns_string+=''' "{}", '''.format(c) 
        
    sql = '''SELECT TOP {k} {cols}  
        COSINE_SIMILARITY("{vector_col}", VECTOR_EMBEDDING('{query}', 'QUERY', 'SAP_NEB.20240715')) AS "COSINE_SIMILARITY"
        FROM "{table_name}"
        ORDER BY "COSINE_SIMILARITY" DESC'''.format(k=k, cols=return_columns_string,vector_col=vector_col, query=query, table_name=table_name)
    cursor.execute(sql)
    hdf = cursor.fetchall()
    return hdf[:k]