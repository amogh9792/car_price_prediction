import numpy as np
import os
import pandas as pd
import pickle
import category_encoders as ce
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from source.exception import CustomException
from source.logger import logging
from source.utility.utility import export_data_csv, import_csv_file

warnings.filterwarnings('ignore')

class DataTransformation:
    def __init__(self, utility_config):
        self.utility_config = utility_config

    def feature_encoding(self, data,  target, save_encoder_path = None, load_encoder_path = None, key = None):
        try:
            for col in self.utility_config.dt_binary_class_col:
                data[col] = data[col].map({'Manual': 1, 'Automatic': '0'})

            if save_encoder_path:
                encoder = ce.TargetEncoder(cols = self.utility_config.dt_multi_class_col)
                data_encoded = encoder.fit_transform(data[self.utility_config.dt_multi_class_col], data[self.utility_config.target_column])

        except CustomException as e:
            raise e
