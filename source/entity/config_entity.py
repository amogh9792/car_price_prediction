import os
from source.constant import constant
import warnings
warnings.filterwarnings("ignore")


class PipelineConfig:
    def __init__(self, global_timestamp):
        self.artifact_dir = os.path.join(constant.ARTIFACT_DIR, global_timestamp)
        self.global_timestamp = global_timestamp
        self.target_column = constant.TARGET_COLUMN
        self.train_pipeline = constant.TRAIN_PIPELINE_NAME
        self.train_file_name = constant.TRAIN_FILE_NAME
        self.test_file_name = constant.TEST_FILE_NAME

        # Data Ingestion Constant

        self.train_di_dir = os.path.join(self.artifact_dir, constant.DI_DIR_NAME)

        self.train_feature_store_file_path = os.path.join(self.artifact_dir, self.train_pipeline, constant.DI_DIR_NAME, constant.DI_FEATURE_STORE_DIR)
        self.train_feature_store_file_name = constant.FILE_NAME
        self.train_di_train_file_path = os.path.join(self.artifact_dir, self.train_pipeline, constant.DI_DIR_NAME, constant.DI_INGESTED_DIR)
        self.train_di_test_file_path = os.path.join(self.artifact_dir, self.train_pipeline, constant.DI_DIR_NAME, constant.DI_INGESTED_DIR)

        self.train_test_split_ratio = constant.DI_TRAIN_TEST_SPLIT_RATIO
        self.mongodb_url_key = os.environ[constant.MONGODB_KEY]
        self.database_name = constant.DATABASE_NAME
        self.train_collection_name = constant.TRAIN_DI_COLLECTION_NAME

        self.di_col_drop_in_clean = constant.DI_COL_DROP_IN_CLEAN
        self.mandatory_col_list = constant.MANDATORY_COL_LIST
        self.mandatory_col_data_type = constant.MANDATORY_COL_DATA_TYPE

        # Data Validation
        self.imputation_values_file_name = constant.DV_IMPUTATION_VALUES_FILE_NAME
        self.outlier_params_file = constant.DV_OUTLIER_PARAMS_FILE
        self.train_dv_imputation_values_file_name = constant.DV_IMPUTATION_VALUES_FILE_NAME

        self.train_dv_train_file_path = os.path.join(self.artifact_dir, self.train_pipeline, constant.DV_DIR_NAME)
        self.train_dv_test_file_path = os.path.join(self.artifact_dir, self.train_pipeline, constant.DV_DIR_NAME)

        # Data Transformation

        self.dt_binary_class_col = constant.DT_BINARY_CLASS_COL
        self.dt_multi_class_col = constant.DT_MULTI_CLASS_COL
        self.dt_multi_class_encoder = constant.DT_ENCODER_PATH
        self.train_dt_train_file_path = os.path.join(self.artifact_dir, self.train_pipeline, constant.DT_DIR_NAME)
        self.train_dt_test_file_path = os.path.join(self.artifact_dir, self.train_pipeline, constant.DT_DIR_NAME)





