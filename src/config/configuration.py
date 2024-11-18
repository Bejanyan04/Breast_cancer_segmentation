from src.path_constants import *
import os
from pathlib import Path
from src import logger
from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import (DataIngestionConfig )
from src.entity.config_entity import (TrainArguments, InferenceArguments,TrainAugmentation, InferenceAugmentation, resize, 
                           horizontal_flip, vertical_flip, random_rotate90,
                            shift_scale_rotate, 
                            activation, normalize, transform_settings, Model)



class ConfigurationManager:
    def __init__(
        self,
        config_filepath =   Path(CONFIG_FILE_PATH)):
        logger.info(f"config_path {config_filepath}")
        self.config = read_yaml(config_filepath)


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

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

        

    def get_train_inference_config(self):
        config_filepath =  Path(TRAINING_PARAMS_CONFIG)
        config = read_yaml(config_filepath)

        model_config = config.model
        activation_config = model_config.activation
        training_augmentation= config.training.augmentation
        normalize = config.training.normalize
        transform =config.training.transform_settings
        batch_size = config.training.batch_size
    
        num_epochs = config.training.num_epochs
        loss_function= config.loss_function.type
        optimzer = config.optimizer
        inference_batch_size = config.inference.batch_size
        model_endpoint_path = config.inference.model_path
        metrics_reduction = config.inference.metrics.reduction

        resize_obj = resize(probability=training_augmentation.resize.probability, height = training_augmentation.resize.height, width = training_augmentation.resize.width)
        horizontal_flip_obj = horizontal_flip(probability=training_augmentation.horizontal_flip.probability)
        vertical_flip_obj = vertical_flip(probability=training_augmentation.vertical_flip.probability)
        random_rotate90_obj = random_rotate90(probability=training_augmentation.random_rotate_90.probability)
        shift_scale_rotate_config = training_augmentation.shift_scale_rotate
        shift_scale_rotate_obj = shift_scale_rotate(shift_limit = shift_scale_rotate_config.shift_limit,scale_limit = shift_scale_rotate_config.scale_limit,
                                                 rotate_limit = shift_scale_rotate_config.rotate_limit, probability = shift_scale_rotate_config.probability)
        activation_obj = activation(type=activation_config.type, dim = activation_config.parameters.dim)


        model_obj = Model(model_name = model_config.model_name, encoder_name = model_config.encoder_name, encoder_weights = model_config.encoder_weights, in_channels = model_config.in_channels, classes = model_config.classes, activation= activation_obj )
        activation_obj = model_config.activation
        train_augmentation_obj = TrainAugmentation(resize=resize_obj, horizontal_flip=horizontal_flip_obj,
                                                   vertical_flip=vertical_flip_obj,random_rotate90=random_rotate90_obj, 
                                                   shift_scale_rotate=shift_scale_rotate_obj)
        inference_augmentation_obj = InferenceAugmentation(resize=resize_obj)
        train_arguments = TrainArguments(optimizer = config.optimizer.type, augmentation=train_augmentation_obj,
                                         model=model_obj, batch_size=config.training.batch_size, num_epochs = config.training.num_epochs,
                                          loss_function=config.loss_function.type,lr_rate= config.optimizer.parameters.lr,
                                            weight_decay=config.optimizer.parameters.weight_decay, normalize = config.training.normalize, 
                                            transform_settings = config.training.transform_settings, device = config.training.device)
        
        inference_arguments = InferenceArguments(metrics_reduction = config.inference.metrics.reduction,batch_size = config.inference.batch_size,
                                            model_path = config.inference.model_path, augmentation=inference_augmentation_obj,
                                            transform_settings =config.training.transform_settings, normalize = config.training.normalize, 
                                                model_architecture=model_obj, device = config.inference.device)
        return train_arguments, inference_arguments
        

    def get_train_config(self):
        return self.get_train_inference_config()[0]
    
    def get_inference_config(self):
        return self.get_train_inference_config()[1]

    
