"""## 6. Evaluation"""

# Plotting training history

def plot_training_history(history):
    # Get the history data
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    dice = history.history['dice_coef']
    val_dice = history.history['val_dice_coef']
    jaccard = history.history['jaccard_coef']
    val_jaccard = history.history['val_jaccard_coef']
    epochs_range = range(len(acc))

    # Plot Accuracy
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy')
    plt.legend(loc='lower right')
    plt.title('Train vs Validation Accuracy')

    # Plot Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.legend(loc='upper right')
    plt.title('Train vs Validation Loss')

    # Plot Dice Coefficient
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, dice, label='Train Dice Coefficient')
    plt.plot(epochs_range, val_dice, label='Val Dice Coefficient')
    plt.legend(loc='lower right')
    plt.title('Train vs Validation Dice Coefficient')

    # Plot Jaccard Coefficient
    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, jaccard, label='Train Jaccard Coefficient')
    plt.plot(epochs_range, val_jaccard, label='Val Jaccard Coefficient')
    plt.legend(loc='lower right')
    plt.title('Train vs Validation Jaccard Coefficient')

    # Show the plots
    plt.tight_layout()
    plt.show()

# Call the function to plot
plot_training_history(history)

def print_numbers_only(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    dice = history.history['dice_coef']
    val_dice = history.history['val_dice_coef']
    jaccard = history.history['jaccard_coef']
    val_jaccard = history.history['val_jaccard_coef']
    for i in range(len(acc)):
        print(f"Epoch {i+1}:")
        print(f"  Training Accuracy: {acc[i]:.4f}")
        print(f"  Validation Accuracy: {val_acc[i]:.4f}")
        print(f"  Training Loss: {loss[i]:.4f}")
        print(f"  Validation Loss: {val_loss[i]:.4f}")
        print(f"  Training Dice: {dice[i]:.4f}")
        print(f"  Validation Dice: {val_dice[i]:.4f}")
        print(f"  Training Jaccard: {jaccard[i]:.4f}")
        print(f"  Validation Jaccard: {val_jaccard[i]:.4f}")

print_numbers_only(history)


# Evaluate model performance on the validation set
val_loss, val_accuracy, val_dice, val_jaccard = model.evaluate(X_val, y_val)
print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, Validation Dice Coefficient: {val_dice}, Validation Jaccard Coefficient: {val_jaccard}')

# Save the model architecture to a .keras file
model.save("segnet_model.keras", save_format='keras')

print("Model architecture and weights saved successfully.")
