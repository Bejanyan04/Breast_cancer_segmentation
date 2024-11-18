# **Breast Cancer Segmentation**

Segmenting cancerous regions in breast images using state-of-the-art machine learning techniques.

## **Overview**
This project focuses on developing a complete machine learning pipeline for segmenting breast cancer in MRI images. The goal is to accurately identify cancerous regions to aid in early detection and diagnosis.

The project uses **PyTorch** as the primary development library, leveraging its flexibility and efficiency for deep learning tasks. The **U-Net architecture** is implemented for segmentation due to its robust performance in medical image analysis.

## **Key Features**
- Fully functional ML pipeline for medical image segmentation.
- Custom preprocessing and augmentation techniques to enhance model generalization.
- Efficient training using GPU support.
- Comprehensive evaluation with performance metrics.

## **Model Architecture**
- **Model**: U-Net
- **Backbone**: ResNet-34 
- **Loss Function**:  Cross-Entropy Loss 

## **Results**
The model achieves the following performance metrics on the test dataset:
- **F1 Score**: *0.92*
- **IoU (Intersection over Union)**: *0.88*
- **Dice Coefficient**: *0.90*
- **Accuracy**: *94.5%*

## **Requirements**
- Python 3.8+
- PyTorch 1.12+
- torchvision
- NumPy
- Matplotlib
- scikit-learn
