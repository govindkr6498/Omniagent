import os
import logging
import boto3
import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
import json

logger = logging.getLogger("data_pipeline")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

S3_BUCKET ="accountrecord" #os.getenv("BUCKET_NAME")
S3_KEY = "emaarGroupRecords/b8db7d18-6340-4f1b-9bbb-a2736503cc98/1593091727-2025-06-13T07:17:56" #os.getenv("INDEX_KEY")
LOCAL_FILE = 'downloaded_file_from_s3.jsonl'
# Ensure data directory exists
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, 'properties.db')
TABLE_NAME = 'properties'

AWS_ACCESS_KEY_ID = os.getenv("AWS_CLIENT_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_CLIENT_SECRET")
AWS_REGION = "us-east-1" #os.getenv("AWS_REGION")


def download_from_s3():
    logger.info("Downloading data from S3 bucket: %s, key: %s", S3_BUCKET, S3_KEY)
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    s3 = session.client('s3')
    s3.download_file(S3_BUCKET, S3_KEY, LOCAL_FILE)
    logger.info("Downloaded file to %s", LOCAL_FILE)
    return LOCAL_FILE

def normalize_jsonl_to_df(filepath):
    logger.info("Normalizing JSONL data from %s", filepath)
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            obj = json.loads(line)
            # Normalize keys: lower, replace __c with _c
            norm = {k.lower().replace('__c', '_c'): v for k, v in obj.items()}
            records.append(norm)
    df = pd.DataFrame(records)
    logger.info("Loaded %d records into DataFrame", len(df))
    return df

def ingest_to_sqlite(df):
    logger.info("Ingesting data into SQLite DB at %s", DB_PATH)
    engine = create_engine(f'sqlite:///{DB_PATH}')
    # Drop table if exists
    with engine.connect() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS {TABLE_NAME}'))
    df.to_sql(TABLE_NAME, engine, index=False)
    logger.info("Ingested %d rows into table '%s'", len(df), TABLE_NAME)

def run_data_pipeline():
    local_file = download_from_s3()
    df = normalize_jsonl_to_df(local_file)
    ingest_to_sqlite(df)
    logger.info("Data pipeline complete.")

if __name__ == "__main__":
    run_data_pipeline()

