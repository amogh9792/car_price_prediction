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
                feature_store_file_path = self.utility_config.train_feature_store_dir_path
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

    def split_data_test_train(self, data: DataFrame):
        try:

            logging.info("Start: train, test split data")

            train_set, test_set = train_test_split(data, test_size = self.utility_config.train_test_split_ratio, random_state = 42)

            logging.info("Complete: Train test Split")

            return train_set, test_set

        except CustomException as e:
            raise e

    def clean_data(self, data, key):
        try:
            logging.info("Start: clean data")

            if key == 'train':
                data = data.drop_duplicates()

                data = data.loc[:, data.nunique() > 1]

                drop_column = []

                for col in data.select_dtypes(include = ['object']).columns:
                    unique_count = data[col].nunique()

                    if unique_count / len(data) > 0.5:
                        data.drop(col, axis = 1, inplace = True)
                        drop_column.append(col)

                logging.info(f"dropped columns : {drop_column}")

                logging.info("Complete: clean data")

            return data


        except CustomException as e:
            raise e

    def process_data(self, data, key):
        logging.info("Start: Processing the data")

        if key == "train":
            mandatory_cols = self.utility_config.mandatory_col_list.copy()

        if key in ['test', 'predict']:
            mandatory_cols = self.utility_config.mandatory_col_list.copy()
            mandatory_cols.remove('selling_price')

            data = data.drop(self.utility_config.di_col_drop_in_clean, axis=1)

        for col in mandatory_cols:
            if col not in data.columns:
                raise CustomException(f"missing mandatory column: {col}")

            if data[col].dtype != self.utility_config.mandatory_col_data_type[col]:
                try:
                    # Handle converting string values to float
                    data[col] = pd.to_numeric(data[col], errors='coerce')

                except ValueError as e:
                    raise CustomException(f"ERROR: Converting data type for column: {col}")

        data = data[mandatory_cols]

        logging.info("Complete: Process data")

        return data

    def initiate_data_ingestion(self, key):

        try:

            logging.info("Start: Data Ingestion")

            data = self.export_data_into_feature_store(key)
            data = self.process_data(data, key)
            data = self.clean_data(data, key)

            if key == 'train':

                train_data, test_data = self.split_data_test_train(data)
                export_data_csv(train_data, self.utility_config.train_file_name, self.utility_config.train_di_train_file_path)
                export_data_csv(test_data, self.utility_config.test_file_name,self.utility_config.train_di_test_file_path)

            logging.info("Complete: Data Ingestion")
        except CustomException as e:
            raise e
