# Load EfficientNet-B0 with pre-trained weights from ImageNet
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)




# Modify the classifier layer for your dataset (e.g., number of classes)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(train_dataset.classes))  # Number of classes in your dataset
)




# Move the model to the appropriate device (GPU or CPU)
model = model.to(device)



# Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
