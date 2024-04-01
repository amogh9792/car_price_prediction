import os
import pandas as pd
from pandas import DataFrame
from source.exception import CustomException
from pymongo.mongo_client import MongoClient
from sklearn.model_selection import train_test_split
from source.utility.utility import export_data_csv
from source.logger import logging

class DataIngestion:
    def __init__(self, utility_config):
        self.utility_config = utility_config

    def export_data_into_feature_store(self, key) -> DataFrame:
        try:
            logging.info("Start: Data load from mongodb")

            if key == 'train':
                collection_name = self.utility_config.train_collection_name
                feature_store_file_path = self.utility_config.train_feature_store_file_path
                feature_store_file_name = self.utility_config.train_feature_store_file_name

            else:
                pass

            client = MongoClient(self.utility_config.mongodb_url_key)
            database = client[self.utility_config.database_name]

            collection = database[collection_name]
            cursor = collection.find()
            data = pd.DataFrame(list(cursor))

            export_data_csv(data, feature_store_file_name, feature_store_file_path)

            logging.info("Complete: Data Load From MongoDB")

            return data

        except CustomException as e:
            raise e


    def initiate_data_ingestion(self, key):

        try:

            logging.info("Start: Data Ingestion")

            data = self.export_data_into_feature_store(key)

            logging.info("Complete: Data Ingestion")
        except CustomException as e:
            raise e
