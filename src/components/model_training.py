import torch
from torch.utils.data import Dataset, DataLoader
from typing import Literal
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import mlflow
import torch.nn.functional as F
from mlflow.models.signature import infer_signature

softmax = torch.nn.Softmax(dim=1)
sigmoid = torch.nn.Sigmoid()

import os
from PIL import Image

from torch.utils.data import Dataset
#import torch.segmentation_models as smp
from tqdm import tqdm

class CancerDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform

        self.annotation_folder_path = f"{data_folder}/annotations"
        self.image_folder_path = f"{data_folder}/images"

        self.sample_names = [
            f for f in os.listdir(self.image_folder_path)
            if f.endswith(('.png', '.jpg', '.jpeg')) 
        ]

    def __len__(self): 
        return len(self.sample_names)

    def __getitem__(self, idx):
        img_name = self.sample_names[idx]
        img_path = os.path.join(self.image_folder_path, img_name)
        mask_path = os.path.join(self.annotation_folder_path, img_name)

        image = np.array(Image.open(img_path))
        if 'normal' in mask_path:
          mask = np.zeros((image.shape[0], image.shape[1]))
        else:
          mask = np.array(Image.open(mask_path))

        mask = mask.astype(np.uint8)
        if 'malignant' in mask_path:
          mask[mask == 1] = 2

        if self.transform:
            transform_result = self.transform(image=image, mask = mask)
            image = transform_result['image']
            mask = transform_result['mask']
        return image, mask
     

def train_loop(model, train_dataloader, val_dataloader, optimizer, loss_function, device, num_epochs, metrics_reduction, classes, model_logging_dir  = None):

    for epoch in range(0, num_epochs):
        train_loss = 0
        train_accuracy = 0
        train_f1_score = 0
        train_iou_score = 0
        train_recall = 0
        train_precision = 0

        model.to(device)
        model.train()

        for image, mask in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            image = image.to(device, dtype=torch.float)
            gr_tr_mask = mask.to(device, dtype=torch.long)

            predicted_mask = model(image)
            loss = loss_function(predicted_mask,gr_tr_mask)

            predicted_mask = torch.argmax(softmax(predicted_mask), axis=1)

            loss.backward()
            optimizer.step()

            tp, fp, fn, tn = smp.metrics.get_stats(predicted_mask, gr_tr_mask, mode='multiclass', num_classes = classes)

            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction=metrics_reduction)
            f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction=metrics_reduction)
            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction=metrics_reduction)
            recall = smp.metrics.recall(tp, fp, fn, tn, reduction=metrics_reduction)
            precision =  smp.metrics.precision(tp, fp, fn, tn, reduction=metrics_reduction)

            train_loss += loss.item()
            train_accuracy +=  accuracy
            train_precision += precision
            train_iou_score += iou_score
            train_recall += recall
            train_f1_score += f1_score

        batch_num = len(train_dataloader)
        print( f'log train metrics, batch_num {batch_num}')
        mlflow.log_metric('train_loss', f"{train_loss/batch_num}", step=epoch)
        mlflow.log_metric('train_accuracy', f'{train_accuracy/batch_num}', step=epoch)
        mlflow.log_metric('train_precision', f'{train_precision/batch_num}', step=epoch)
        mlflow.log_metric('train_iou_score', f'{train_iou_score/batch_num}', step=epoch)
        mlflow.log_metric('train_f1_score', f'{train_f1_score/batch_num}', step=epoch)


        #write evaluation part
        print('validation')
        with torch.no_grad():
            
            val_loss = 0
            val_accuracy = 0
            val_f1_score = 0
            val_iou_score = 0
            val_recall = 0
            val_precision = 0
            best_loss = float('inf')
            best_f1 = 0

            model.eval()
            batch_num = len(val_dataloader)

            for image, mask in val_dataloader:
                image = image.to(device, dtype=torch.float)
                gr_tr_mask = mask.to(device, dtype=torch.long)

                predicted_mask = model(image)
                loss = loss_function(predicted_mask, gr_tr_mask)


                predicted_mask = torch.argmax(softmax(predicted_mask), axis=1)

                tp, fp, fn, tn = smp.metrics.get_stats(predicted_mask, gr_tr_mask, mode='multiclass', num_classes=classes)

                iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction= metrics_reduction)
                f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction=metrics_reduction)
                accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction=metrics_reduction)
                recall = smp.metrics.recall(tp, fp, fn, tn, reduction=metrics_reduction)
                precision =  smp.metrics.precision(tp, fp, fn, tn, reduction=metrics_reduction)


                val_loss += loss.item()
                val_accuracy +=  accuracy
                val_precision += precision
                val_iou_score += iou_score
                val_recall += recall
                val_f1_score += f1_score

            print('log eval metrics')
            signature = infer_signature(image.cpu().numpy(), predicted_mask.cpu().numpy())

            mlflow.log_metric('val_loss', f"{val_loss/batch_num:2f}", step=epoch)
            mlflow.log_metric('val_accuracy', f'{val_accuracy/batch_num:2f}', step=epoch)
            mlflow.log_metric('val_precision', f'{val_precision/batch_num:2f}', step=epoch)
            mlflow.log_metric('val_iou_score', f'{val_iou_score/batch_num:2f}', step=epoch)
            mlflow.log_metric('val_f1_score', f'{val_f1_score/batch_num:2f}', step=epoch)

            if val_loss / batch_num < best_loss:
                mlflow.pytorch.log_model(model, "best_loss_model", signature=signature, input_example = image.cpu().numpy())
                best_loss =  val_loss / batch_num

            if val_f1_score / batch_num > best_f1:
                mlflow.pytorch.log_model(model, "best_f1_model", signature=signature, input_example = image.cpu().numpy())
                best_loss =  val_f1_score / batch_num
            
