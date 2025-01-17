import os
import configparser

from flask import Flask, request, jsonify
from flask_cors import CORS
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
connection = dataframe.ConnectionContext(hanaURL, hanaPort, hanaUser, hanaPW)

app = Flask(__name__)
CORS(app)

# Step 2: Function to create the table if it doesn't exist
def create_table_if_not_exists():
    create_table_sql = """
        DO BEGIN
            DECLARE table_exists INT;
            SELECT COUNT(*) INTO table_exists
            FROM SYS.TABLES 
            WHERE TABLE_NAME = 'TCM_SAMPLE' AND SCHEMA_NAME = 'DBUSER';
            
            IF table_exists = 0 THEN
                CREATE COLUMN TABLE DBUSER.TCM_SAMPLE (
                    TEXT NVARCHAR(500),
                    VECTOR REAL_VECTOR
                );
            END IF;
        END;
    """
    
    # Use cursor to execute the query
    cursor = connection.connection.cursor()
    cursor.execute(create_table_sql)
    cursor.close()  # Always close the cursor after execution
    
# Step 3: Function to insert text and its embedding vector into the "tcm_sample" table
@app.route('/insert_text_and_vector', methods=['POST'])
def insert_text_and_vector():
    # Create the table if it doesn't exist
    create_table_if_not_exists()

    data = request.get_json()
    text = data.get('text')
    text_type = data.get('text_type', 'DOCUMENT')
    model_version = data.get('model_version', 'SAP_NEB.20240715')

    # Generate the embedding vector using VECTOR_EMBEDDING
    sql_insert = f"""
        INSERT INTO tcm_sample (TEXT, VECTOR)
        SELECT '{text}', VECTOR_EMBEDDING('{text}', '{text_type}', '{model_version}')
        FROM DUMMY
    """
    
    # Use cursor to execute the query
    cursor = connection.connection.cursor()
    cursor.execute(sql_insert)
    cursor.close()  # Always close the cursor after execution
    
    return jsonify({"message": "Text and vector inserted successfully"}), 200

# Step 4: Function to compare a new text's vector to existing stored vectors using COSINE_SIMILARITY
@app.route('/compare_text_to_existing', methods=['POST'])
def compare_text_to_existing():
    data = request.get_json()
    query_text = data.get('query_text')
    text_type = data.get('text_type', 'QUERY')
    model_version = data.get('model_version', 'SAP_NEB.20240715')

    # Generate the new text's embedding and compare using COSINE_SIMILARITY
    sql_query = f"""
        SELECT TOP 3
            TEXT, 
            COSINE_SIMILARITY(
                VECTOR, 
                VECTOR_EMBEDDING('{query_text}', '{text_type}', '{model_version}')
            ) AS SIMILARITY
        FROM tcm_sample
        ORDER BY SIMILARITY DESC
    """
    hana_df = dataframe.DataFrame(connection, sql_query)
    similarities = hana_df.collect()  # Return results as a pandas DataFrame

    # Convert results to a list of dictionaries for JSON response
    results = similarities.to_dict(orient='records')
    return jsonify({"similarities": results}), 200

@app.route('/', methods=['GET'])
def root():
    return 'Embeddings API: Health Check Successfull.', 200

def create_app():
    return app

# Start the Flask app
if __name__ == '__main__':
    app.run('0.0.0.0', 8080)