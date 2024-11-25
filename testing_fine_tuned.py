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



# Evaluate the model on the test set
test_labels, test_preds, avg_inference_time = evaluate_pruned_model(model, test_loader, device)



# Print the classification report
print("\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=test_dataset.classes))

# Generate and display the confusion matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Print the average inference time
print(f"Average Inference Time per Image: {avg_inference_time:.4f} seconds")


original_model_path = 'C:\\Users\\malee\\fine_tuned_pruned_model.pth'
new_model_path = 'C:\\Users\\malee\\BESTMODEL.pth'

# Check if the file exists and remove it if it does
if os.path.exists(new_model_path):
    os.remove(new_model_path)

# Now rename the file
os.rename(original_model_path, new_model_path)