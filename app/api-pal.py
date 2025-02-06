import os
import configparser

from hana_ml import dataframe

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
connection = dataframe.ConnectionContext(address=hanaURL, port=hanaPort, user=hanaUser, password=hanaPW)

sql_select = f"""
    SELECT TOP 3 TEXT_ID, TEXT
    FROM DBUSER.TCM_MYKNOWLEDGEBASE
"""

myknowledgebase_hdf = connection.sql(sql_select)

### Generating Text Embeddings in SAP HANA Cloud with the new PAL function, function available with hana-ml 2.23.
from hana_ml.text.pal_embeddings import PALEmbeddings
pe = PALEmbeddings()
textembeddings = pe.fit_transform(myknowledgebase_hdf, key="TEXT_ID", target=["TEXT"])

# Review the generated Text Embeddings
print(textembeddings.head(3).collect())