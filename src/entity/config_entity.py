from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    unzip_dir : str
    root_data_path: Path
    split_random_seed: int
    raw_data_path: Path
    zip_file_path: Path
    unzip_folder: Path
    raw_unziped_data_path: Path
    annotations_path: Path
    images_path: Path
    train_data_path: Path
    test_data_path:  Path
    val_data_path:   Path
    split_random_seed : int
    train_ratio: float
    test_ratio: float
    val_ratio: float
        
 
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int



@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path



@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list



@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    params_image_size: list
    params_batch_size: int