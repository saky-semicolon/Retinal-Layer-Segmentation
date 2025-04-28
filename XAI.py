"""## 8. Explainable AI (XAI)-Grad-CAM (Gradient-weighted Class Activation Mapping) Integration
Grad-CAM is a method that generates class-specific heatmaps highlighting the areas of an image that are most important for a given prediction.
"""

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from tensorflow.keras.saving import register_keras_serializable

def customized_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred) + 0.5 * (1 - dice_coef(y_true, y_pred))

from tensorflow import keras

# Define or import your customized_loss function here
# ... (Your definition of dice_coef function should go here) ...
# Example:
def dice_coef(y_true, y_pred, smooth=1):
    intersection = keras.backend.sum(y_true * y_pred, axis=[1,2,3])
    union = keras.backend.sum(y_true, axis=[1,2,3]) + keras.backend.sum(y_pred, axis=[1,2,3])
    dice = keras.backend.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

# Define jaccard_coef function: This is the missing piece causing the error.
# You need to provide the actual implementation based on how it was defined when the model was saved.
def jaccard_coef(y_true, y_pred, smooth=1):
    intersection = keras.backend.sum(y_true * y_pred, axis=[1,2,3])
    union = keras.backend.sum(y_true, axis=[1,2,3]) + keras.backend.sum(y_pred, axis=[1,2,3]) - intersection
    jaccard = keras.backend.mean((intersection + smooth)/(union + smooth), axis=0)
    return jaccard


def customized_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred) + 0.5 * (1 - dice_coef(y_true, y_pred))

# Replace 'path/to/your/model.h5' with the actual path to your saved model file.
model = keras.models.load_model(
    '/kaggle/input/segnet-model/keras/default/1/segnet_model.keras',
    custom_objects={'customized_loss': customized_loss, 'dice_coef': dice_coef, 'jaccard_coef': jaccard_coef} # Include jaccard_coef in custom_objects
)

# Now you can use the loaded model for prediction or further training.
# For example, to make a prediction:
# prediction = model.predict(your_input_data)

model.summary()

target_layer = "conv2d_19"

def compute_gradcam(model, image, layer_name, class_idx):
    """
    Generates Grad-CAM heatmap for a given image and target class.

    Parameters:
    - model: Trained segmentation model
    - image: Input image (single example)
    - layer_name: Name of the target convolutional layer
    - class_idx: Class index (0 to 7)

    Returns:
    - heatmap: Grad-CAM heatmap for the target class
    """

    grad_model = tf.keras.models.Model(
        inputs=[model.input],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(np.expand_dims(image, axis=0))

        # Ensure predictions is a tensor
        if isinstance(predictions, list):
            predictions = predictions[0]

        # Compute loss for the specific class index
        loss = tf.reduce_mean(predictions[..., class_idx])

    # Compute gradients
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps with gradients
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1).numpy()
    heatmap = np.squeeze(heatmap)

    # Normalize heatmap (ReLU activation)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)  # Scale between 0-1

    return heatmap


def overlay_gradcam(image, heatmap, alpha=0.5):
    """
    Overlays Grad-CAM heatmap onto the original grayscale image.

    Parameters:
    - image: Original grayscale image (H, W, 1) or (H, W)
    - heatmap: Grad-CAM heatmap (smaller resolution)
    - alpha: Transparency factor (default=0.5)

    Returns:
    - Overlayed image with Grad-CAM heatmap
    """

    # Resize heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Convert grayscale image to RGB if necessary
    if len(image.shape) == 2 or image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Convert heatmap to color map
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Merge with the original image
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return overlay

import numpy as np
import matplotlib.pyplot as plt

# Sample image index (replace with an actual index from your dataset)
sample_idx = 0
input_image = X_val[sample_idx]  # Select an image from the validation set

# Convert to 8-bit grayscale for visualization
original_image = np.uint8(input_image * 255)

# Display the raw OCT image
plt.figure(figsize=(5, 5))
plt.imshow(original_image.squeeze(), cmap="gray")
plt.title("Raw OCT Image")
plt.axis("off")
plt.show()

# Sample image for visualization (Replace with actual input image)
sample_idx = 0
input_image = X_val[sample_idx]  # Use a validation set sample
original_image = np.uint8(input_image * 255)  # Convert to grayscale 8-bit

# Number of segmented classes (e.g., 8 layers in retinal OCT images)
num_classes = 8

# Plot Grad-CAM for all classes
plt.figure(figsize=(15, 10))

for class_idx in range(num_classes):
    # Compute Grad-CAM heatmap for the current class
    gradcam_heatmap = compute_gradcam(model, input_image, target_layer, class_idx)

    # Overlay Grad-CAM on the original image
    overlayed_image = overlay_gradcam(original_image, gradcam_heatmap)

    # Display results
    plt.subplot(2, 4, class_idx + 1)
    plt.imshow(overlayed_image)
    plt.title(f"Grad-CAM for Class {class_idx}")
    plt.axis("off")

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Sample image index (replace with an actual index from your dataset)
sample_idx = 1
input_image = X_val[sample_idx]  # Select an image from the validation set

# Convert to 8-bit grayscale for visualization
original_image = np.uint8(input_image * 255)

# Display the raw OCT image
plt.figure(figsize=(5, 5))
plt.imshow(original_image.squeeze(), cmap="gray")
plt.title("Raw OCT Image")
plt.axis("off")
plt.show()

# Sample image for visualization (Replace with actual input image)
sample_idx = 1
input_image = X_val[sample_idx]  # Use a validation set sample
original_image = np.uint8(input_image * 255)  # Convert to grayscale 8-bit

# Number of segmented classes (e.g., 8 layers in retinal OCT images)
num_classes = 8

# Plot Grad-CAM for all classes
plt.figure(figsize=(15, 10))

for class_idx in range(num_classes):
    # Compute Grad-CAM heatmap for the current class
    gradcam_heatmap = compute_gradcam(model, input_image, target_layer, class_idx)

    # Overlay Grad-CAM on the original image
    overlayed_image = overlay_gradcam(original_image, gradcam_heatmap)

    # Display results
    plt.subplot(2, 4, class_idx + 1)
    plt.imshow(overlayed_image)
    plt.title(f"Grad-CAM for Class {class_idx}")
    plt.axis("off")

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Sample image index (replace with an actual index from your dataset)
sample_idx = 3
input_image = X_val[sample_idx]  # Select an image from the validation set

# Convert to 8-bit grayscale for visualization
original_image = np.uint8(input_image * 255)

# Display the raw OCT image
plt.figure(figsize=(5, 5))
plt.imshow(original_image.squeeze(), cmap="gray")
plt.title("Raw OCT Image")
plt.axis("off")
plt.show()

# Sample image for visualization (Replace with actual input image)
sample_idx = 3
input_image = X_val[sample_idx]  # Use a validation set sample
original_image = np.uint8(input_image * 255)  # Convert to grayscale 8-bit

# Number of segmented classes (e.g., 8 layers in retinal OCT images)
num_classes = 8

# Plot Grad-CAM for all classes
plt.figure(figsize=(15, 10))

for class_idx in range(num_classes):
    # Compute Grad-CAM heatmap for the current class
    gradcam_heatmap = compute_gradcam(model, input_image, target_layer, class_idx)

    # Overlay Grad-CAM on the original image
    overlayed_image = overlay_gradcam(original_image, gradcam_heatmap)

    # Display results
    plt.subplot(2, 4, class_idx + 1)
    plt.imshow(overlayed_image)
    plt.title(f"Grad-CAM for Class {class_idx}")
    plt.axis("off")

plt.tight_layout()
plt.show()

def overlay_colormap(image, heatmap, colormap=cv2.COLORMAP_VIRIDIS, alpha=0.6):
    """
    Overlays a Grad-CAM heatmap with different colormap styles.

    Parameters:
    - image: Original grayscale image
    - heatmap: Grad-CAM heatmap
    - colormap: OpenCV colormap type
    - alpha: Transparency factor

    Returns:
    - Overlayed image with different color mapping
    """
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay

target_class = 1  # Example: Change this to the desired class index (0-7 for 8 classes)
gradcam_heatmap = compute_gradcam(model, input_image, "conv2d_19", target_class)

# Test different colormaps
plt.figure(figsize=(15, 5))

for idx, cmap in enumerate([cv2.COLORMAP_JET, cv2.COLORMAP_HOT, cv2.COLORMAP_VIRIDIS]):
    overlayed_img = overlay_colormap(original_image, gradcam_heatmap, colormap=cmap)

    plt.subplot(1, 3, idx + 1)
    plt.imshow(overlayed_img)
    plt.title(f"Grad-CAM with {['JET', 'HOT', 'VIRIDIS'][idx]}")
    plt.axis("off")

plt.show()

plt.figure(figsize=(15, 10))

for class_idx in range(num_classes):
    gradcam_heatmap = compute_gradcam(model, input_image, "conv2d_18", class_idx)
    overlayed_image = overlay_gradcam(original_image, gradcam_heatmap)

    plt.subplot(2, 4, class_idx + 1)
    plt.imshow(overlayed_image)
    plt.title(f"Class {class_idx}")
    plt.axis("off")

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))

for class_idx in range(num_classes):
    gradcam_heatmap = compute_gradcam(model, input_image, "conv2d_19", class_idx)
    overlayed_image = overlay_gradcam(original_image, gradcam_heatmap)

    plt.subplot(2, 4, class_idx + 1)
    plt.imshow(overlayed_image)
    plt.title(f"Class {class_idx}")
    plt.axis("off")

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))

for class_idx in range(num_classes):
    gradcam_heatmap = compute_gradcam(model, input_image, "conv2d_20", class_idx)
    overlayed_image = overlay_gradcam(original_image, gradcam_heatmap)

    plt.subplot(2, 4, class_idx + 1)
    plt.imshow(overlayed_image)
    plt.title(f"Class {class_idx}")
    plt.axis("off")

plt.tight_layout()
plt.show()
