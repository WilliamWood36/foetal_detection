# Main execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_dataset, test_dataset = load_data(challenge=True)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model, loss, optimizer
model = FetalQRSWindowDetector(in_channels=4).to(device)

criterion = nn.BCELoss()




optimizer = optim.Adam(model.parameters(), lr=0.0005)


# Training loop

batch_size = 32
num_epochs = 200

start_time = time.time()  # Start the timer
progressive_accuracy, progressive_loss = [],[]
for epoch in range(num_epochs):
    train_loss = train(train_loader,model, criterion, optimizer,0.2)
    test_loss, accuracy = test(test_loader,model, criterion,)

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f} - "
          f"Test Loss: {test_loss:.4f} - "
          f"Accuracy: {accuracy:.2f}%")
    progressive_accuracy.append(accuracy)
    progressive_loss.append(test_loss)

base_results_plot(num_epochs,progressive_loss,progressive_accuracy)
end_time = time.time()  # End the timer
elapsed_time = end_time - start_time  # Compute elapsed time
