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



# Save the pruned model
torch.save(model.state_dict(), "pruned_model.pth")
print("Pruned model saved as 'pruned_model.pth'")



# Fine-tuning after pruning
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Use a smaller learning rate

num_epochs = 5  # Fine-tune for a few epochs
for epoch in range(num_epochs):
    train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    val_loss, val_accuracy = validate_one_epoch(model, valid_loader, criterion, device)
    print(f"Epoch {epoch+1}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")



# Save the fine-tuned model
torch.save(model.state_dict(), "fine_tuned_pruned_model.pth")
print("Fine-tuned pruned model saved as 'fine_tuned_pruned_model.pth'")