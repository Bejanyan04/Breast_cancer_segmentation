from src.config.configuration import ConfigurationManager
import torch.optim as optim
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

class ComponentsLoader:
    def __init__(self):
        configuration_manager = ConfigurationManager()
        self.train_arguments = configuration_manager.get_train_config()
        self.inference_arguments = configuration_manager.get_inference_config()

    
    def get_model(self):
        model_config = self.train_arguments.model
        model_name = model_config.model_name
        sample_model = getattr(smp, model_config.model_name)
        sample_model = sample_model(encoder_name = model_config.encoder_name,
                                    encoder_weights = model_config.encoder_weights,
                                    in_channels = model_config.in_channels,
                                    classes = model_config.classes
        )
        return sample_model

    def get_optimizer(self):
        model= self.get_model()
        optimizer_name = self.train_arguments.optimizer
        lr_rate = self.train_arguments.lr_rate
        weight_decay = self.train_arguments.weight_decay
        optimizer = getattr(optim, optimizer_name)
        optimizer = optimizer(model.parameters(), lr=lr_rate, weight_decay=weight_decay)
        return optimizer

    
    def get_loss_function(self):
        loss_function_name = self.train_arguments.loss_function
        loss = getattr(nn, loss_function_name)
        loss = loss()
        return loss
 
    
    def get_train_transform(self):
        def trivial_normalize(x, **kwargs):
            x = x / 255.0
            return x
   
        augmentations = self.train_arguments.augmentation

        resize = augmentations.resize
        shift_scale_rotate =  augmentations.shift_scale_rotate

        train_transform = A.Compose([
            A.Resize(height=resize.height, width=resize.width, p=resize.probability), 
            A.HorizontalFlip(p=augmentations.horizontal_flip.probability),
            A.VerticalFlip(p=augmentations.vertical_flip.probability),
            A.RandomRotate90(p=augmentations.random_rotate90.probability),
            A.ShiftScaleRotate(shift_limit=shift_scale_rotate.shift_limit, scale_limit=shift_scale_rotate.scale_limit,
                                rotate_limit=shift_scale_rotate.rotate_limit, p=shift_scale_rotate.probability),
            A.Lambda(trivial_normalize, always_apply=True),
            ToTensorV2(),
        ])
        return train_transform


    def get_test_transform(self):
        def trivial_normalize(x, **kwargs):
            x = x / 255.0
            return x
        resize = self.inference_arguments.augmentation.resize

        
        test_transform = A.Compose([
            A.Resize(height=resize.height, width=resize.width, p=resize.probability), 
            A.Lambda(trivial_normalize, always_apply=True),
            ToTensorV2(),
        ])

        return test_transform
        

    





  
                                    

        
        




