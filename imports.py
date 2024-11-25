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