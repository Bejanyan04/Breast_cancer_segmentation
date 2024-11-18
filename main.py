
from src import logger 
from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_02_model_training import ModelTrainingPipeline

STAGE_NAME = 'data_ingestion'

if __name__ == '__main__':
    try:
        
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        
        STAGE_NAME ='model training'
        train_obj = ModelTrainingPipeline()
        train_obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logger.exception(e)
        raise e
