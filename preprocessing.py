import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.utils.prune as prune
import os
import seaborn as sns
from PIL import Image
from nbconvert import PythonExporter
import nbformat
import time
import streamlit as st



# Set device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# **Data Preparation**


# Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match input size of EfficientNet-B0
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])


# Define the directories for train, valid, and test datasets
data_dir = r"C:\Users\malee\Downloads\lungcancer (3)\data"  
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test')


# Load the training, validation, and test datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Check classes
num_classes = len(train_dataset.classes)
print(f"Number of classes: {num_classes}")