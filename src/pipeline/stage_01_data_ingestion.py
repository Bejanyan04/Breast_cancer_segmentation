
from src.config.configuration import ConfigurationManager
from src import logger
from src.components.data_ingestion import unzip_data_file, separate_annotations, preprocess_mask_labels, split_dataset



STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        ConfigManager = ConfigurationManager()

        data_ingestion_config = ConfigManager.get_data_ingestion_config()
        zip_file_path = data_ingestion_config.zip_file_path
        extract_dir = data_ingestion_config.unzip_folder
        whole_data_folder = data_ingestion_config.raw_unziped_data_path
        images_folder = data_ingestion_config.images_path
        annotations_folder  = data_ingestion_config.annotations_path
        raw_unziped_data_path = data_ingestion_config.raw_unziped_data_path
        root_data_path = data_ingestion_config.root_data_path

        random_seed = data_ingestion_config.split_random_seed
        train_ratio = data_ingestion_config.train_ratio
        test_ratio = data_ingestion_config.test_ratio
        val_ratio = data_ingestion_config.val_ratio


        unzip_data_file(zip_file_path, extract_dir)
        logger.info(f"file unziped in \n{extract_dir}")
        
        separate_annotations(whole_data_folder, images_folder, annotations_folder)
        logger.info("data separated in image and annotation parts")

        #preprocess_mask_labels(annotations_folder)
        logger.info("get multiclass labels from binaruies by doing  {'nomal': 0, 'benign': 1, 'malignant': 2}")

        split_dataset(unziped_data_path= raw_unziped_data_path, data_folder=root_data_path, random_seed=random_seed, train_ratio=train_ratio, test_ratio=test_ratio, val_ratio=val_ratio)
        logger.info("data is splitted into train test validation groups")


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e