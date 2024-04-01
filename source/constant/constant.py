# Common Constants

TARGET_COLUMN = 'selling_price'
TRAIN_PIPELINE_NAME = 'train'
ARTIFACT_DIR = 'artifact'
FILE_NAME = 'train_data.csv'

TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'

MONGODB_KEY = "MONGODB_KEY"
DATABASE_NAME = "car_sales"

# Data Ingestion Constants

TRAIN_DI_COLLECTION_NAME = "car_dataset"
DI_DIR_NAME = "data_ingestion"
DI_FEATURE_STORE_DIR = "feature_store"
DI_INGESTED_DIR = 'ingested'
DI_TRAIN_TEST_SPLIT_RATIO = 0.2
DI_COL_DROP_IN_CLEAN = ['name', '_id']

# Data Validation Constant

DV_IMPUTATION_VALUES_FILE_NAME = "source/ml/imputation_values.csv"

DV_OUTLIER_PARAMS_FILE = 'source/ml/outlier_details.csv'
DV_DIR_NAME = "data_validation"



