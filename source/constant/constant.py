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

DI_MANDATORY_COLUMN_LIST = ['year', 'selling_price', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage(km/ltr/kg)', 'engine', 'max_power', 'seats']
DI_MANDATORY_COLUMN_DATA_TYPE = {
    'year': 'int64',
    'selling_price': 'int64',
    'km_driven': 'int64',
    'fuel': 'object',
    'seller_type': 'object',
    'transmission': 'object',
    'owner': 'object',
    'mileage(km/ltr/kg)': 'float64',
    'engine': 'int64',
    'max_power': 'float64',
    'seats': 'int64'
}
# Data Validation Constant

DV_IMPUTATION_VALUES_FILE_NAME = "source/ml/imputation_values.csv"

DV_OUTLIER_PARAMS_FILE = 'source/ml/outlier_details.csv'
DV_DIR_NAME = "data_validation"

# Data Transformation Constant

DT_MULTI_CLASS_COL = ['year', 'km_driven', 'fuel', 'seller_type', 'owner', 'mileage(km/ltr/kg)', 'engine', 'max_power', 'seats']
DT_BINARY_CLASS_COL = ['transmission']
DT_ENCODER_PATH = "source/ml/multi_class_encoder.pkl"
DT_DIR_NAME = "data_transformation"
MP_DIR_NAME = "model_prediction"

# Model train & evaluate
MODEL_PATH = "source/ml/artifact"
FINAL_MODEL_PATH = "source/ml/final_model"

# Prediction constant
PREDICT_PIPELINE_NAME = 'predict'
PREDICT_DATA_FILE_NAME = 'predict_data.csv'
PREDICT_FILE = 'predict.csv'

FINAL_MODEL_FILE_NAME = 'CatBoostRegressor.pkl'







