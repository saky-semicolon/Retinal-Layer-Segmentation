"""## 7. Prediction"""

# Predict on validation set
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=-1)  # Convert predictions to class labels

# Same color map for actual and predicted labels
color_map = {
    0: [0, 0, 0],        # Background
    1: [128, 0, 0],      # Class 1
    2: [0, 128, 0],      # Class 2
    3: [128, 128, 0],    # Class 3
    4: [0, 128, 128],    # Class 4
    5: [64, 0, 0],       # Class 5
    6: [192, 0, 0],      # Class 6
    7: [64, 128, 0]      # Class 7
}

# Create colored label image from predicted labels
def create_colored_label_image(output):
    colored_image = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)  # Initialize colored image
    for j in range(output.shape[0]):
        for k in range(output.shape[1]):
            colored_image[j, k] = color_map[output[j][k]]
    return colored_image

# Generate colored label images for actual and predicted masks
colored_actual_labels = [create_colored_label_image(np.argmax(y_val[i], axis=-1)) for i in range(5)]
colored_predicted_labels = [create_colored_label_image(y_pred_classes[i]) for i in range(5)]

# Visualize results for the first 5 samples
def plot_results(index):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(X_val[index].squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('True Mask (Colored)')
    plt.imshow(colored_actual_labels[index])
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Predicted Mask (Colored)')
    plt.imshow(colored_predicted_labels[index])
    plt.axis('off')

    plt.show()

# Visualize results for the first 5 samples
for i in range(5):
    plot_results(i)

"""## Additional: Check the Predicted Results for First 25 Samples"""

# Generate colored label images for actual and predicted masks
colored_actual_labels = [create_colored_label_image(np.argmax(y_val[i], axis=-1)) for i in range(25)]
colored_predicted_labels = [create_colored_label_image(y_pred_classes[i]) for i in range(25)]

# Visualize results for the first 25 samples
def plot_results(index):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(X_val[index].squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('True Mask (Colored)')
    plt.imshow(colored_actual_labels[index])
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Predicted Mask (Colored)')
    plt.imshow(colored_predicted_labels[index])
    plt.axis('off')

    plt.show()

# Visualize results for the first 25 samples
for i in range(25):
    plot_results(i)

"""# Extra Visualizations
## --> Might be Useful
"""

def compare_predictions(X_test, y_test, predictions, num_samples=5):
    """Displays original images, ground truth masks, and predicted masks side by side."""
    for i in range(num_samples):
        plt.figure(figsize=(12, 4))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.imshow(X_test[i].squeeze(), cmap='gray')
        plt.title("Original Image")

        # Ground Truth Mask
        plt.subplot(1, 3, 2)
        plt.imshow(np.argmax(y_test[i], axis=-1), cmap='jet')
        plt.title("Ground Truth Mask")

        # Predicted Mask
        plt.subplot(1, 3, 3)
        plt.imshow(np.argmax(predictions[i], axis=-1), cmap='jet')
        plt.title("Predicted Mask")

        plt.show()

# Run visualization
predictions = model.predict(X_val)
compare_predictions(X_val, y_val, predictions, num_samples=5)

import seaborn as sns

def calculate_iou(y_true, y_pred):
    """Computes IoU for each class."""
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)
    iou_scores = []

    for i in range(8):  # Assuming 8 classes
        intersection = np.logical_and(y_true == i, y_pred == i).sum()
        union = np.logical_or(y_true == i, y_pred == i).sum()
        iou = intersection / (union + 1e-7)  # Avoid division by zero
        iou_scores.append(iou)

    return iou_scores

# Compute IoU for validation set
iou_scores = np.mean([calculate_iou(y_val[i], predictions[i]) for i in range(len(y_val))], axis=0)

# Plot IoU as a heatmap
plt.figure(figsize=(8, 5))
sns.heatmap([iou_scores], annot=True, cmap="coolwarm", xticklabels=range(8), yticklabels=["IoU"])
plt.title("Class-wise IoU Scores")
plt.xlabel("Segmentation Class")
plt.show()

def overlay_mask(image, mask, alpha=0.5):
    """Overlays a predicted mask onto the original grayscale image."""
    plt.figure(figsize=(8, 8))

    # Show the grayscale image
    plt.imshow(image.squeeze(), cmap='gray')

    # Overlay the predicted mask with transparency
    plt.imshow(mask, cmap='jet', alpha=alpha)

    plt.title("Overlay: Original Image + Predicted Mask")
    plt.axis("off")
    plt.show()

# Example visualization
sample_idx = 2  # Change this to view different samples
predicted_mask = np.argmax(predictions[sample_idx], axis=-1)  # Convert one-hot to class labels

overlay_mask(X_val[sample_idx], predicted_mask)

from sklearn.metrics import precision_recall_curve

def plot_precision_recall(y_true, y_pred, num_classes=8):
    """Plots Precision-Recall curves for all segmentation classes."""
    plt.figure(figsize=(8, 6))

    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true.flatten() == i, y_pred.flatten() == i)
        plt.plot(recall, precision, label=f'Class {i}')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for Each Class")
    plt.legend()
    plt.grid()
    plt.show()

# Convert one-hot encoded masks to label format
y_true_labels = np.argmax(y_val, axis=-1)
y_pred_labels = np.argmax(predictions, axis=-1)

# Run the visualization
plot_precision_recall(y_true_labels, y_pred_labels)

def class_wise_accuracy(y_true, y_pred, num_classes=8):
    """Calculates per-class accuracy by comparing ground truth and predictions."""
    y_true = np.argmax(y_true, axis=-1)  # Convert one-hot to class labels
    y_pred = np.argmax(y_pred, axis=-1)  # Convert softmax to class labels

    accuracy_per_class = np.zeros(num_classes)
    for i in range(num_classes):
        correct = np.sum((y_true == i) & (y_pred == i))
        total = np.sum(y_true == i)
        accuracy_per_class[i] = correct / (total + 1e-7)  # Avoid division by zero

    # Plot accuracy for each class
    plt.figure(figsize=(8, 5))
    plt.bar(range(num_classes), accuracy_per_class, color='skyblue')
    plt.xlabel("Segmentation Class")
    plt.ylabel("Accuracy")
    plt.title("Class-wise Segmentation Accuracy")
    plt.xticks(range(num_classes))
    plt.ylim(0, 1)
    plt.grid(axis="y")
    plt.show()

# Compute and visualize accuracy per class
class_wise_accuracy(y_val, predictions)
