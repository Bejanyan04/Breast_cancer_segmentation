from src.config.configuration import ConfigurationManager
from src import logger
from src.utils.component_loaders import ComponentsLoader
from src.components.model_training import CancerDataset, train_loop
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader
import mlflow


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        components_loader = ComponentsLoader()

        config_manager = ConfigurationManager()

        train_config = config_manager.get_train_config()
        val_config = config_manager.get_inference_config()

        train_transform = components_loader.get_train_transform()
        test_transform = components_loader.get_test_transform()
        model = components_loader.get_model()
        optimizer = components_loader.get_optimizer()
        loss = components_loader.get_loss_function()
        device = train_config.device
        num_epochs = train_config.num_epochs
        metrics_reduction = val_config.metrics_reduction
        classes = train_config.model.classes
        ingestion_config_manager = config_manager.get_data_ingestion_config()

        train_dataset = CancerDataset(ingestion_config_manager.train_data_path,  train_transform)
        test_dataset = CancerDataset(ingestion_config_manager.test_data_path, test_transform)
        val_dataset = CancerDataset(ingestion_config_manager.val_data_path, test_transform)
        
        train_dataloader = DataLoader(train_dataset, batch_size= train_config.batch_size, shuffle=True)


        val_dataloader = DataLoader(val_dataset, batch_size = val_config.batch_size, shuffle=True)
        logger.info(f"start training")


        with mlflow.start_run(run_name = 'unet_plus_plus') as run:
            params = {
            "epochs": train_config.num_epochs,
            "learning_rate": train_config.lr_rate,
            "batch_size":  train_config.batch_size,
            "loss_function":train_config.loss_function,
            "optimizer": train_config.optimizer,
        }

            # Log training parameters.
            mlflow.log_params(params)
            train_loop(model, train_dataloader, val_dataloader, optimizer, loss, device, num_epochs, metrics_reduction, classes)
