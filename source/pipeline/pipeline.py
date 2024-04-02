from source.component.data_ingestion import DataIngestion
from source.component.data_validation import DataValidation
from source.entity.config_entity import PipelineConfig

class DataPipeline:

    def __init__(self, global_timestamp):
        self.utility_config = PipelineConfig(global_timestamp)

    def start_data_ingestion(self,key):
        data_ingestion_obj = DataIngestion(self.utility_config)
        data_ingestion_obj.initiate_data_ingestion(key)

    def start_data_validation(self, key):
        data_validation_obj = DataValidation(self.utility_config)
        data_validation_obj.initiate_data_validation(key)

    def run_train_pipeline(self):
        self.start_data_ingestion('train')
        self.start_data_validation('train')