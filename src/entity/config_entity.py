from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List


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
        
 
@dataclass 
class resize:
    probability: str
    height: str
    width: str

@dataclass
class  horizontal_flip:
    probability: float

@dataclass 
class vertical_flip:
    probability : float

@dataclass 
class random_rotate90:
    probability : float

@dataclass
class shift_scale_rotate:
    shift_limit: float
    scale_limit: float
    rotate_limit: float
    probability: float

@dataclass
class activation:
    type: str
    dim: int

@dataclass 
class normalize:
    function: str
    apply_always: bool

@dataclass
class transform_settings:
    to_tensor: bool


@dataclass 
class Model:
    activation: activation
    model_name: str
    encoder_name:str
    encoder_weights: str
    in_channels: int
    classes: int

     
@dataclass
class TrainAugmentation:
    resize: resize
    horizontal_flip: horizontal_flip
    vertical_flip: vertical_flip 
    random_rotate90: random_rotate90
    shift_scale_rotate: shift_scale_rotate

@dataclass (frozen=True)
class TrainArguments:
    lr_rate: float
    weight_decay: float
    batch_size: int
    num_epochs: int
    model: Model
    augmentation: TrainAugmentation
    loss_function: str
    optimizer: str
    normalize: normalize
    transform_settings: transform_settings
    device: str


@dataclass 
class InferenceAugmentation:
    resize: resize

@dataclass (frozen=True)
class InferenceArguments:
    metrics_reduction: str
    batch_size : int
    augmentation: InferenceAugmentation
    normalize: normalize
    transform_settings: transform_settings
    model_path:  Path
    model_architecture: Model
    device: str
