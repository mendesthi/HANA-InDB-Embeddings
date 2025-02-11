import os
import configparser
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS
from hana_ml import dataframe

from app.utilities_hana import kmeans_and_tsne  # Correct import statement CF
# from utilities_hana import kmeans_and_tsne  # Correct import statement LOCALHOST

# Check if the application is running on Cloud Foundry
if 'VCAP_APPLICATION' in os.environ:
    # Running on Cloud Foundry, use environment variables
    hanaURL = os.getenv('DB_ADDRESS')
    hanaPort = os.getenv('DB_PORT')
    hanaUser = os.getenv('DB_USER')
    hanaPW = os.getenv('DB_PASSWORD')
else:
    # Not running on Cloud Foundry, read from config.ini file
    config = configparser.ConfigParser()
    config.read('config.ini')
    hanaURL = config['database']['address']
    hanaPort = config['database']['port']
    hanaUser = config['database']['user']
    hanaPW = config['database']['password']

# Step 1: Establish a connection to SAP HANA
connection = dataframe.ConnectionContext(hanaURL, hanaPort, hanaUser, hanaPW)

app = Flask(__name__)
CORS(app)

# Function to create the PROJECT_BY_CATEGORY table if it doesn't exist
def create_project_by_category_table_if_not_exists():
    create_table_sql = """
        DO BEGIN
            DECLARE table_exists INT;
            
            -- Check and create PROJECT_BY_CATEGORY table
            SELECT COUNT(*) INTO table_exists
            FROM SYS.TABLES 
            WHERE TABLE_NAME = 'PROJECT_BY_CATEGORY' AND SCHEMA_NAME = CURRENT_SCHEMA;
            
            IF table_exists = 0 THEN
                CREATE TABLE PROJECT_BY_CATEGORY (
                    PROJECT_ID INT,
                    CATEGORY_ID INT
                );
            END IF;
        END;
    """
    
    # Use cursor to execute the query
    cursor = connection.connection.cursor()
    cursor.execute(create_table_sql)
    cursor.close()  # Always close the cursor after execution

@app.route('/update_categories_and_projects', methods=['POST'])
def update_categories_and_projects():
    data = request.get_json()
    categories = data
    
    if not categories:
        return jsonify({"error": "No categories provided"}), 400
    
    cursor = connection.connection.cursor()
    
    # Drop existing values from the CATEGORIES table
    cursor.execute("TRUNCATE TABLE CATEGORIES")
    
    # Add new categories to the CATEGORIES table
    for index, (title, description) in enumerate(categories.items()):
        insert_sql = f"""
            INSERT INTO CATEGORIES ("index", "category_label", "category_descr")
            VALUES ({index}, '{title.replace("'", "''")}', '{description.replace("'", "''")}')
        """
        cursor.execute(insert_sql)
    
    # Ensure the PROJECT_BY_CATEGORY table exists
    create_project_by_category_table_if_not_exists()
    
    # Retrieve categories from the CATEGORIES table
    categories_df = dataframe.DataFrame(connection, 'SELECT * FROM CATEGORIES')
    
    # Retrieve topics from the ADVISORIES2 table
    advisories_df = dataframe.DataFrame(connection, 'SELECT "project_number", "topic" FROM ADVISORIES2')
    
    # Calculate COSINE similarity and update PROJECT_BY_CATEGORY table
    cursor.execute("TRUNCATE TABLE PROJECT_BY_CATEGORY")
    
    for advisory in advisories_df.collect().to_dict(orient='records'):
        # print("Advisory columns:", advisory.keys())
        project_number = advisory['project_number']
        topic = advisory['topic']
        
        # Check if project_number is an integer
        if not isinstance(project_number, int):
            print(f"Skipping project_number={project_number} as it is not an integer")
            continue
    
        # Calculate COSINE similarity with each category
        similarities = []
        for category in categories_df.collect().to_dict(orient='records'):

            category_id = category['index']
            category_description = category['category_descr']
            
            # Use HANA SQL for COSINE similarity
            similarity_sql = f"""
                SELECT COSINE_SIMILARITY(
                    VECTOR_EMBEDDING('{topic.replace("'", "''")}', 'DOCUMENT', 'SAP_NEB.20240715'),
                    VECTOR_EMBEDDING('{category_description.replace("'", "''")}', 'DOCUMENT', 'SAP_NEB.20240715')
                ) AS similarity
                FROM DUMMY
            """

            similarity_df = dataframe.DataFrame(connection, similarity_sql)
            similarity_results = similarity_df.collect()
            
            if not similarity_results.empty:
                similarity = similarity_results.iloc[0]['SIMILARITY']
                similarities.append((category_id, similarity))
            else:
                print(f"No similarity result for category_id={category_id} and topic={topic}")

        # Find the most similar category
        if similarities:
            most_similar_category = max(similarities, key=lambda x: x[1])
            category_id = most_similar_category[0]

            # Update PROJECT_BY_CATEGORY table
            insert_sql = f"""
                INSERT INTO "PROJECT_BY_CATEGORY" ("PROJECT_ID", "CATEGORY_ID")
                VALUES ('{project_number}', {category_id})
            """
            cursor.execute(insert_sql)
        else:
            print(f"No valid similarities found for project_number={project_number}")
    
    cursor.close()
    return jsonify({"message": "Categories and project categories updated successfully"}), 200

# Function to create the CLUSTERING table if it doesn't exist
def create_clustering_table_if_not_exists():
    create_table_sql = """
        DO BEGIN
            DECLARE table_exists INT;
            SELECT COUNT(*) INTO table_exists
            FROM SYS.TABLES 
            WHERE TABLE_NAME = 'CLUSTERING' AND SCHEMA_NAME = CURRENT_SCHEMA;
            
            IF table_exists = 0 THEN
                CREATE TABLE CLUSTERING (
                    PROJECT_NUMBER NVARCHAR(255),
                    x DOUBLE,
                    y DOUBLE,
                    CLUSTER_ID INT
                );
            END IF;
            
            -- Check and create CLUSTERING_DATA table
            SELECT COUNT(*) INTO table_exists
            FROM SYS.TABLES 
            WHERE TABLE_NAME = 'CLUSTERING_DATA' AND SCHEMA_NAME = CURRENT_SCHEMA;
            
            IF table_exists = 0 THEN
                CREATE TABLE CLUSTERING_DATA (
                    CLUSTER_ID INT,
                    CLUSTER_DESCRIPTION NVARCHAR(255),
                    EMBEDDING REAL_VECTOR GENERATED ALWAYS AS VECTOR_EMBEDDING(CLUSTER_DESCRIPTION, 'DOCUMENT', 'SAP_NEB.20240715')
                );
            END IF;
        END;
    """
    
    # Use cursor to execute the query
    cursor = connection.connection.cursor()
    cursor.execute(create_table_sql)
    cursor.close()  # Always close the cursor after execution

@app.route('/refresh_clusters', methods=['POST'])
def refresh_clusters():
    # Ensure the CLUSTERING table exists
    create_clustering_table_if_not_exists()
    
    # Perform clustering and t-SNE on the ADVISORIES table
    df_clusters, labels = kmeans_and_tsne(connection,  ## Hana ConnectionContext
                    table_name='ADVISORIES2', 
                    result_table_name='CLUSTERING', 
                    n_components=64, 
                    perplexity= 5, ## perplexity for T-SNE algorithm  
                    start_date='1900-01-01', ## first date for projects to be considered in the analysis
                    end_date='2024-01-01'
                    )
    
    # Insert the values of the "labels" variable into the CLUSTERING_DATA table
    cursor = connection.connection.cursor()
    for cluster_id, cluster_description in labels.items():
        insert_sql = f"""
            INSERT INTO CLUSTERING_DATA (CLUSTER_ID, CLUSTER_DESCRIPTION)
            VALUES ({cluster_id}, '{cluster_description.replace("'", "''")}')
        """
        cursor.execute(insert_sql)
    cursor.close()

    return jsonify({"message": "Clusters refreshed successfully"}), 200

@app.route('/get_clusters', methods=['GET'])
def get_clusters():
    # Ensure the CLUSTERING table exists
    create_clustering_table_if_not_exists()
    
    # Retrieve data from the CLUSTERING table
    sql_query = "SELECT * FROM CLUSTERING"
    hana_df = dataframe.DataFrame(connection, sql_query)
    clusters = hana_df.collect()  # Return results as a pandas DataFrame
    
    # Convert DataFrame to list of dictionaries
    formatted_clusters = [
        {
            "x": row["x"],
            "y": row["y"],
            "CLUSTER_ID": row["CLUSTER_ID"],
            "PROJECT_NUMBER": row["PROJECT_NUMBER"]
        }
        for _, row in clusters.iterrows()
    ]
    
    return jsonify(formatted_clusters), 200

@app.route('/get_clusters_description', methods=['GET'])
def get_clusters_description():
    # Ensure the CLUSTERING table exists
    create_clustering_table_if_not_exists()
    
    # Retrieve data from the CLUSTERING table
    sql_query = "SELECT * FROM CLUSTERING_DATA"
    hana_df = dataframe.DataFrame(connection, sql_query)
    clusters = hana_df.collect()  # Return results as a pandas DataFrame
    
    # Convert DataFrame to list of dictionaries
    formatted_cluster_description = [
        {
            "CLUSTER_ID": row["CLUSTER_ID"],
            "CLUSTER_DESCRIPTION": row["CLUSTER_DESCRIPTION"]
        }
        for _, row in clusters.iterrows()
    ]
    
    return jsonify(formatted_cluster_description), 200

# Step 2: Function to create the table if it doesn't exist
def create_table_if_not_exists(schema_name, table_name):
    create_table_sql = f"""
        DO BEGIN
            DECLARE table_exists INT;
            SELECT COUNT(*) INTO table_exists
            FROM SYS.TABLES 
            WHERE TABLE_NAME = '{table_name.upper()}' AND SCHEMA_NAME = '{schema_name.upper()}';
            
            IF table_exists = 0 THEN
                CREATE TABLE {schema_name}.{table_name} (
                    TEXT_ID INT GENERATED BY DEFAULT AS IDENTITY,
                    TEXT NVARCHAR(5000),
                    EMBEDDING REAL_VECTOR GENERATED ALWAYS AS VECTOR_EMBEDDING(TEXT, 'DOCUMENT', 'SAP_NEB.20240715')
                );
            END IF;
        END;
    """
    
    # Use cursor to execute the query
    cursor = connection.connection.cursor()
    cursor.execute(create_table_sql)
    cursor.close()  # Always close the cursor after execution
    
# Step 3: Function to insert text and its embedding vector into the "TCM_SAMPLE" table
@app.route('/insert_text_and_vector', methods=['POST'])
def insert_text_and_vector():

    data = request.get_json()
    schema_name = data.get('schema_name', 'DBUSER')  # Default schema
    table_name = data.get('table_name', 'TCM_SAMPLE')  # Default table
    text = data.get('text')
    # text_type = data.get('text_type', 'DOCUMENT')
    # model_version = data.get('model_version', 'SAP_NEB.20240715')

    # Create the table if it doesn't exist
    create_table_if_not_exists(schema_name, table_name)
    
    # Generate the embedding vector using VECTOR_EMBEDDING
    sql_insert = f"""
        INSERT INTO {schema_name}.{table_name} (TEXT) SELECT '{text}' FROM DUMMY
    """
    
    # Use cursor to execute the query
    cursor = connection.connection.cursor()
    cursor.execute(sql_insert)
    cursor.close()  # Always close the cursor after execution
    
    return jsonify({"message": f"Text inserted successfully into {schema_name}.{table_name}"}), 200

# Step 4: Function to compare a new text's vector to existing stored vectors using COSINE_SIMILARITY
@app.route('/compare_text_to_existing', methods=['POST'])
def compare_text_to_existing():
    data = request.get_json()
    schema_name = data.get('schema_name', 'DBUSER')  # Default schema
    query_text = data.get('query_text')
    text_type = data.get('text_type', 'QUERY')
    model_version = data.get('model_version', 'SAP_NEB.20240715')
    
    if not query_text:
        return jsonify({"error": "Query text is required"}), 400
    
    # Generate the new text's embedding and compare using COSINE_SIMILARITY
    sql_query = f"""
        SELECT "solution" AS text,
               "project_number", 
               COSINE_SIMILARITY(
                   "solution_embedding", 
                   VECTOR_EMBEDDING('{query_text}', '{text_type}', '{model_version}')
               ) AS similarity
        FROM {schema_name}.ADVISORIES
        UNION ALL
        SELECT "comment" AS text, 
               "project_number", 
               COSINE_SIMILARITY(
                   "comment_embedding", 
                   VECTOR_EMBEDDING('{query_text}', '{text_type}', '{model_version}')
               ) AS similarity
        FROM {schema_name}.COMMENTS
        ORDER BY similarity DESC
        LIMIT 5
    """
    hana_df = dataframe.DataFrame(connection, sql_query)
    similarities = hana_df.collect()  # Return results as a pandas DataFrame

    # Convert results to a list of dictionaries for JSON response
    results = similarities.to_dict(orient='records')
    return jsonify({"similarities": results}), 200

@app.route('/get_project_details', methods=['GET'])
def get_project_details():
    schema_name = request.args.get('schema_name', 'DBUSER')  # Default schema
    project_number = request.args.get('project_number')
    
    if not project_number:
        return jsonify({"error": "Project number is required"}), 400
    
    # SQL query to join ADVISORIES and COMMENTS tables on project_number, excluding embeddings
    sql_query = f"""
        SELECT a."architect", a."index" AS advisories_index, a."pcb_number", a."project_date", 
               a."project_number", a."solution", a."topic",
               c."comment", c."comment_date", c."index" AS comments_index
        FROM {schema_name}.advisories a
        LEFT JOIN {schema_name}.comments c
        ON a."project_number" = c."project_number"
        WHERE a."project_number" = {project_number}
    """
    hana_df = dataframe.DataFrame(connection, sql_query)
    project_details = hana_df.collect()  # Return results as a pandas DataFrame

    # Convert results to a list of dictionaries for JSON response
    results = project_details.to_dict(orient='records')
    return jsonify({"project_details": results}), 200

@app.route('/get_all_projects', methods=['GET'])
def get_all_projects():
    schema_name = request.args.get('schema_name', 'DBUSER')  # Default schema
    
    # SQL query to retrieve all data from ADVISORIES and COMMENTS tables, excluding embeddings
    sql_query = f"""
        SELECT a."architect", a."index" AS advisories_index, a."pcb_number", a."project_date", 
               a."project_number", a."solution", a."topic",
               c."comment", c."comment_date", c."index" AS comments_index
        FROM {schema_name}.advisories a
        LEFT JOIN {schema_name}.comments c
        ON a."project_number" = c."project_number"
    """
    hana_df = dataframe.DataFrame(connection, sql_query)
    all_projects = hana_df.collect()  # Return results as a pandas DataFrame

    # Convert results to a list of dictionaries for JSON response
    results = all_projects.to_dict(orient='records')
    return jsonify({"all_projects": results}), 200

@app.route('/', methods=['GET'])
def root():
    return 'Embeddings API: Health Check Successfull.', 200

def create_app():
    return app

# Start the Flask app
if __name__ == '__main__':
    app.run('0.0.0.0', 8080)