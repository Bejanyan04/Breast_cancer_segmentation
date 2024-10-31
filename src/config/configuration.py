from src.path_constants import *
import os
from pathlib import Path
from src import logger
from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                PrepareCallbacksConfig,
                                                TrainingConfig,
                                                EvaluationConfig)



class ConfigurationManager:
    def __init__(
        self,
        config_filepath =   Path(CONFIG_FILE_PATH)):
        logger.info(f"config_path {config_filepath}")
        self.config = read_yaml(config_filepath)

        #create_directories([self.config.artifacts_root])

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

       # create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
        root_data_path = Path(config.root_data_path),
        unzip_dir = Path(config.unzip_folder),
        raw_data_path= Path(config.raw_data_path),
        zip_file_path = Path(config.zip_file_path),
        unzip_folder = Path(config.unzip_folder),
        raw_unziped_data_path = Path(config.raw_unziped_data_path),
        annotations_path = Path(config.annotations_path),
        images_path = Path(config.images_path),
        train_data_path = Path(config.train_data_path),
        test_data_path = Path(config.test_data_path),
        val_data_path = Path(config.val_data_path),
        split_random_seed = config.split_random_seed,
        train_ratio = config.train_ratio,
        test_ratio = config.test_ratio, 
        val_ratio = config.val_ratio
    )

        return data_ingestion_config    


        