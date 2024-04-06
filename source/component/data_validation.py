import os
import pandas as pd
import numpy as np
from source.utility.utility import import_csv_file, export_data_csv
from source.exception import CustomException
from source.logger import logging


class DataValidation:
    def __init__(self, utility_config):
        self.utility_config = utility_config
        self.outlier_params = {}

    def handle_missing_value(self, data, key):

        try:

            logging.info("Start Handling Missing Values")

            if key == 'train':

                numerical_columns = data.select_dtypes(include=['number']).columns
                numerical_imputation_values = data[numerical_columns].median()
                data[numerical_columns] = data[numerical_columns].fillna(numerical_imputation_values)

                categorical_columns = data.select_dtypes(include=['object']).columns

                categorical_imputation_values = data[categorical_columns].mode().iloc[0]
                data[categorical_columns] = data[categorical_columns].fillna(categorical_imputation_values)

                imputation_values = pd.concat([numerical_imputation_values, categorical_imputation_values])
                imputation_values.to_csv(self.utility_config.imputation_values_file, header=['imputation_value'])

            if key in ['test', 'predict']:

                imputation_values = pd.read_csv(self.utility_config.imputation_values_file, index_col=0)['imputation_value']

                numerical_columns = data.select_dtypes(include=['number']).columns
                data[numerical_columns] = data[numerical_columns].fillna(imputation_values[numerical_columns])

                categorical_columns = data.select_dtypes(include=['object']).columns
                data[categorical_columns] = data[categorical_columns].fillna(imputation_values[categorical_columns].iloc[0])

            logging.info("Complete: Handling the missing values")

            return data

        except CustomException as e:
            raise e

    def outlier_detection_handle(self, data, key):
        try:

            logging.info("Start: Outlier Detection And Handling..")

            if key == 'train':
                # Your existing outlier detection logic for the 'train' dataset
                for column_name in data.select_dtypes(include=['number']).columns:
                    if column_name != 'seats':  # Skip outlier detection for 'seats' column
                        Q1 = data[column_name].quantile(0.25)
                        Q3 = data[column_name].quantile(0.75)
                        IQR = Q3 - Q1

                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        outlier_mask_lower = data[column_name] < lower_bound
                        outlier_mask_upper = data[column_name] > upper_bound

                        data.loc[outlier_mask_lower, column_name] = lower_bound
                        data.loc[outlier_mask_upper, column_name] = upper_bound

                        self.outlier_params[column_name] = {'Q1': Q1, 'Q3': Q3, 'IQR': IQR}

                outlier_params_df = pd.DataFrame(self.outlier_params)
                outlier_params_df.to_csv(self.utility_config.outlier_params_file, index=False)

            if key in ['test', 'predict']:
                outlier_params_df = pd.read_csv(self.utility_config.outlier_params_file)
                self.outlier_params = outlier_params_df.to_dict(orient='list')

                for column_name in data.select_dtypes(include=['number']).columns:
                    if column_name != 'seats':  # Skip outlier handling for 'seats' column
                        if column_name in self.outlier_params:
                            Q1 = self.outlier_params[column_name][0]
                            Q3 = self.outlier_params[column_name][1]
                            IQR = self.outlier_params[column_name][2]

                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR

                            outlier_mask_lower = data[column_name] < lower_bound
                            outlier_mask_upper = data[column_name] > upper_bound

                            data.loc[outlier_mask_lower, column_name] = lower_bound
                            data.loc[outlier_mask_upper, column_name] = upper_bound

                # Convert 'engine' column to integer, handling non-integer values gracefully

                data['engine'] = data['engine'].astype(float).astype(int)

            logging.info("Complete: Outlier detection and handling")

            return data

        except CustomException as e:
            raise e

    def initiate_data_validation(self, key):

        logging.info(">>>>>>> INITIATED DATA VALIDATION <<<<<<<<<")

        if key == 'train':

            train_data = import_csv_file(self.utility_config.train_file_name, self.utility_config.train_di_train_file_path)
            test_data = import_csv_file(self.utility_config.test_file_name, self.utility_config.train_di_test_file_path)

            train_data = self.handle_missing_value(train_data, key)
            test_data = self.handle_missing_value(test_data, key='test')

            train_data = self.outlier_detection_handle(train_data, key)
            test_data = self.outlier_detection_handle(test_data, key='test')

            export_data_csv(train_data, self.utility_config.train_file_name,  self.utility_config.train_dv_train_file_path)
            export_data_csv(test_data, self.utility_config.test_file_name,self.utility_config.train_dv_test_file_path)

        if key == 'predict':

            data = import_csv_file(self.utility_config.predict_file, self.utility_config.predict_file_path)
            data = self.handle_missing_value(data, key)
            data = self.outlier_detection_handle(data, key='predict')

            export_data_csv(data, self.utility_config.predict_file, self.utility_config.predict_dv_file_path)

        logging.info(">>>>>> COMPLETE DATA VALIDATION <<<<<<<")