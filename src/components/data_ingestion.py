from pathlib import Path
import zipfile
import os
import yaml

print(os.getcwd())

import shutil
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def unzip_data_file(zip_file_path, extract_dir):
    # Create the directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True) 

    # Extract the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # List the extracted files
    extracted_files = os.listdir(extract_dir)
    print("Extracted files:", extracted_files)


def separate_annotations(data_folder_dir, destination_images_folder, destination_masks_folder):
    for folder_path in Path(data_folder_dir).iterdir():
        print(f"in separatae annotations func dest mask folder {destination_masks_folder}")
        print(f"in separate annotations func dest images folder {destination_images_folder}")

        os.makedirs(destination_masks_folder ,exist_ok=True)
        os.makedirs(destination_images_folder, exist_ok=True)
        
        for sample in os.listdir(folder_path):
            image_name = sample.replace(' ', '_')
            if 'mask' in sample:
                print(sample)
                shutil.copy(f'{folder_path}\{sample}', f'{destination_masks_folder}\\{image_name}')
            else:
                shutil.copy(f'{folder_path}\{sample}', f'{destination_images_folder}\\{image_name}')


def map_labels(sample_path):
  """
  get label mapping {'nomal': 1, 'benign': 2, 'malignant': 3}
  }
  """
  if 'normal' in sample_path:
     return None

  mask = Image.open(sample_path)
  mask = np.array(mask).astype(bool).astype(np.uint8)

  if 'malignant' in sample_path:
    mask[mask==1] = 2

  plt.imsave(sample_path, mask)


def preprocess_mask_labels(annotation_folder_path):
  annotations = [os.path.join(annotation_folder_path, sample_rel_path) for sample_rel_path in os.listdir(annotation_folder_path)]
  list(map(map_labels, annotations))


def copy_in_folder(data_folder,  tumor_type, data_group, indices):
    print(f"copiing group {data_group}")
    for idx in indices:
      #copy image
    
      source_data_path = f"{data_folder}\\processed_data\\images\\{tumor_type}_({idx}).png"

      destination_data_path = f"{data_folder}\\final_data\\{data_group}\\images\\{tumor_type}_{idx}.png"

      #copy annotation
      mask_source_data_path = f"{data_folder}\\processed_data\\annotations\\{tumor_type}_({idx})_mask.png"
      mask_destination_data_path = f"{data_folder}\\final_data\\{data_group}\\annotations\\{tumor_type}_{idx}.png"

      if not os.path.exists(mask_source_data_path) or not os.path.exists(source_data_path):
        continue

      os.makedirs(os.path.dirname(mask_source_data_path), exist_ok=True)
      os.makedirs(os.path.dirname(mask_destination_data_path), exist_ok=True)

      os.makedirs(os.path.dirname(destination_data_path), exist_ok=True)
      os.makedirs(os.path.dirname(source_data_path), exist_ok=True)


      shutil.copy(source_data_path, destination_data_path)
      shutil.copy(mask_source_data_path, mask_destination_data_path)


def save_in_group_folder(tumor_type, indices, data_folder, random_seed, train_ratio, test_ratio, val_ratio):

 # assert tumor_type not in ['benign', 'malignant', 'normal']
  random.seed(int(random_seed))
  random.shuffle(indices)

  count = len(indices)
  train_indices = indices[:int(count*train_ratio)]
  test_indices = indices[int(count*train_ratio) : int(count * (train_ratio+val_ratio))]
  val_indices = indices[int(count* train_ratio + val_ratio):]

  print(f"len train indices {len(train_indices)}")
  print(f"len test indices {len(test_indices)}")
  print(f"len val indices {len(val_indices)}")

  print("copiing in train")
  copy_in_folder(data_folder, tumor_type, 'train', train_indices)
  print("copiing in test")
  copy_in_folder(data_folder, tumor_type, 'test', test_indices)
  print("copiing in val")
  copy_in_folder(data_folder, tumor_type, 'val', val_indices)

def split_dataset(unziped_data_path, data_folder, random_seed, train_ratio, test_ratio, val_ratio):
    len_benign = len(os.listdir(os.path.join(unziped_data_path, 'benign'))) //2
    len_malignant = len(os.listdir(os.path.join(unziped_data_path, 'malignant'))) //2
    len_normal = len(os.listdir(os.path.join(unziped_data_path, 'normal'))) //2
 

    benign_indices, malignant_indices, normal_indices =  list(map(lambda x: list(range(1,x)), [len_benign, len_malignant, len_normal]))
    save_in_group_folder('benign', benign_indices,data_folder, random_seed, train_ratio, test_ratio, val_ratio)
    save_in_group_folder('malignant', malignant_indices, data_folder, random_seed, train_ratio, test_ratio, val_ratio)
    save_in_group_folder('normal', normal_indices,data_folder, random_seed, train_ratio, test_ratio, val_ratio)


