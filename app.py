from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import sys
from src.mlproject.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformation,DataTransformationConfig
from src.mlproject.components.model_tranier import ModelTrainerConfig,ModelTrainer
if __name__ == "__main__":
    logging.info("Logging working fine...")
    
    try:
        data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation()
        train_array,test_array,_ = data_transformation.initiate_data_transormation(train_data_path,test_data_path)

        #Model Training
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_array,test_array))
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)