#!/usr/bin/env python
# coding: utf-8

# In[2]:

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


# In[3]:


# Set device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[4]:


# **Data Preparation**

# In[74]:

# Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match input size of EfficientNet-B0
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])


# In[5]:

# Define the directories for train, valid, and test datasets
data_dir = r"C:\Users\malee\Downloads\lungcancer (3)\data"  
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test')

# In[6]:


# Load the training, validation, and test datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)


# In[7]:


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# In[8]:


# Check classes
num_classes = len(train_dataset.classes)
print(f"Number of classes: {num_classes}")


# In[9]:


# Load EfficientNet-B0 with pre-trained weights from ImageNet
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)


# In[10]:


# Modify the classifier layer for your dataset (e.g., number of classes)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(train_dataset.classes))  # Number of classes in your dataset
)


# In[11]:


# Move the model to the appropriate device (GPU or CPU)
model = model.to(device)


# In[12]:


# Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[13]:


# Training function
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute the loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Optimize
        optimizer.step()
        
        running_loss += loss.item()
        
        # Get the predictions
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)
    
    # Calculate average loss and accuracy for this epoch
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct_predictions / total_predictions
    return avg_loss, accuracy


# Validation function
def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():  # No gradients needed for validation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute the loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Get the predictions
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
    
    # Calculate average loss and accuracy for this epoch
    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct_predictions / total_predictions
    return avg_loss, accuracy


# Training loop
num_epochs = 10
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    # Train for one epoch
    train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    
    # Validate for one epoch
    val_loss, val_accuracy = validate_one_epoch(model, valid_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# Save the best model based on validation accuracy
if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved Best Model!")


# In[14]:


# Load the best model
model.load_state_dict(torch.load("best_model.pth"))


# In[15]:


# Testing function
def test_model(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # No gradients needed for testing
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get the predictions
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_labels, all_preds


# In[16]:


# Evaluate on the test dataset
test_labels, test_preds = test_model(model, test_loader, device)


# In[17]:


# Generate Classification Report
print("\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=test_dataset.classes))


# In[18]:


# Generate Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)


# In[19]:


# Plot Confusion Matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[20]:


# Define pruning on convolutional layers
def apply_pruning(model, pruning_amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):  # Prune Conv2d layers
            prune.l1_unstructured(module, name="weight", amount=pruning_amount)
            prune.remove(module, 'weight')  # Make pruning permanent
        elif isinstance(module, nn.Linear):  # Prune Linear layers
            prune.l1_unstructured(module, name="weight", amount=pruning_amount)
            prune.remove(module, 'weight')  # Make pruning permanent

# Apply pruning with 20% of weights removed
apply_pruning(model, pruning_amount=0.2)


# In[21]:


# Save the pruned model
torch.save(model.state_dict(), "pruned_model.pth")
print("Pruned model saved as 'pruned_model.pth'")


# In[22]:


# Fine-tuning after pruning
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Use a smaller learning rate

num_epochs = 5  # Fine-tune for a few epochs
for epoch in range(num_epochs):
    train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    val_loss, val_accuracy = validate_one_epoch(model, valid_loader, criterion, device)
    print(f"Epoch {epoch+1}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")


# In[23]:


# Save the fine-tuned model
torch.save(model.state_dict(), "fine_tuned_pruned_model.pth")
print("Fine-tuned pruned model saved as 'fine_tuned_pruned_model.pth'")


# In[24]:


# Function to evaluate the pruned model
def evaluate_pruned_model(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []
    inference_times = []
    
    with torch.no_grad():  # No gradients needed for evaluation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

# Calculate average inference time per image
    avg_inference_time = sum(inference_times) / len(inference_times)
    return all_labels, all_preds, avg_inference_time


# In[25]:


# Evaluate the model on the test set
test_labels, test_preds, avg_inference_time = evaluate_pruned_model(model, test_loader, device)


# In[26]:


# Print the classification report
print("\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=test_dataset.classes))


# In[27]:


# Generate and display the confusion matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[28]:


# Print the average inference time
print(f"Average Inference Time per Image: {avg_inference_time:.4f} seconds")


# In[29]:


original_model_path = 'C:\\Users\\malee\\fine_tuned_pruned_model.pth'
new_model_path = 'C:\\Users\\malee\\BESTMODEL.pth'

# Check if the file exists and remove it if it does
if os.path.exists(new_model_path):
    os.remove(new_model_path)

# Now rename the file
os.rename(original_model_path, new_model_path)


# In[30]:
# Set Streamlit page config
st.set_page_config(page_title="Lung Cancer Detection", page_icon="🫁", layout="wide")

# Sidebar Information
st.sidebar.title("🫁 Lung Cancer Detection")
st.sidebar.markdown(
    """
    ### About this App:
    - This application uses a pre-trained EfficientNet-B0 model fine-tuned for lung cancer detection.
    - Upload a CT scan image to get predictions.
    - **Ensure the image is in .jpg or .png format.**

    ### Disclaimer:
    - **This model is for demonstration purposes only.**
    - It is not intended for clinical or diagnostic use.
    - Always consult a qualified healthcare professional for medical advice or diagnoses.
    """
)

# Load the pre-trained model
@st.cache_resource  # Cache model to avoid reloading on each interaction
def load_model():
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.classifier[1].in_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, 4)  # 4 classes as per your dataset
    )
    model.load_state_dict(torch.load("C:\\Users\\malee\\BESTMODEL.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Main App Section
st.title("🫁 AI-Assisted Detection of Lung Cancer")
st.markdown(
    """
    **Welcome to the AI-assisted lung cancer detection application.**  
    ---
     ### Instructions:
    1. Upload a CT scan image using the file uploader below.
    2. Click **Predict** to see the results.
    
    """
)

# File Uploader
uploaded_file = st.file_uploader("Upload a CT scan image", type=["jpg", "png"])

if uploaded_file is not None:
    # Display Uploaded Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict Button
    if st.button("Predict"):
        # Apply transformations
        input_image = transform(image).unsqueeze(0)

        # Perform prediction
        with torch.no_grad():
            output = model(input_image)
            _, prediction = torch.max(output, 1)
        
        # Class Names
        class_names = ['squamous.cell.carcinoma', 'large.cell.carcinoma', 'adenocarcinoma', 'normal']
        
        # Display Results
        st.markdown(
            f"""
            ### Prediction Results:
            - **Predicted Class**: {class_names[prediction.item()]}
            - **Confidence Scores**: {torch.softmax(output, dim=1).numpy()[0]}
            ---
            **Note:** These results are for demonstration purposes only.  
            Always seek professional medical advice for health-related concerns.
            """
        )