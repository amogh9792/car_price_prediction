import os
import pandas as pd
import numpy as np
from source.utility.utility import import_csv_file, export_data_csv
from source.exception import CustomException

class DataValidation:
    def __init__(self, utility_config):
        self.utility_config = utility_config
        self.outlier_params = {}

    def handle_missing_values(self, data, key):
        try:

            if key == 'train':
                numerical_columns = data.select_dtypes(include=['number']).columns
                numerical_imputation_values = data[numerical_columns].median()
                data[numerical_columns] = data[numerical_columns].fillna(numerical_imputation_values)

                categorical_columns = data.select_dtypes(include = ['object']).columns

                categorical_imputation_values = data[categorical_columns].mode().iloc[0]
                data[categorical_columns] = data[categorical_columns].fillna(categorical_imputation_values)

                imputation_values = pd.concat([numerical_imputation_values, categorical_imputation_values])
                imputation_values.to_csv(self.utility_config.imputation_values_file_name, header=['imputation_value'])

                return data

            if key in ['test', 'predict']:

                imputation_values = pd.read_csv(self.utility_config.imputation_values_file_name, index_col=0)['imputation_value']

                numerical_columns = data.select_dtypes(include = ['number']).columns
                data[numerical_columns] = data[numerical_columns].fillna(imputation_values[numerical_columns])

                categorical_columns = data.select_dtypes(include = ['object']).columns
                data[categorical_columns] = data[categorical_columns].fillna(imputation_values[categorical_columns].iloc[0])

                return data

        except CustomException as e:
            raise e

    def initiate_data_validation(self, key):
        if key == 'train':

            train_data = import_csv_file(self.utility_config.train_file_name, self.utility_config.train_di_train_file_path)
            test_data = import_csv_file(self.utility_config.test_file_name, self.utility_config.train_di_test_file_path)

            train_data = self.handle_missing_values(train_data, key)
            test_data = self.handle_missing_values(test_data, key = 'test')

            export_data_csv(train_data, self.utility_config.train_file_name, self.utility_config.train_dv_train_file_path)
            export_data_csv(test_data, self.utility_config.test_file_name, self.utility_config.train_dv_test_file_path)

        if key == 'predict':

            pass