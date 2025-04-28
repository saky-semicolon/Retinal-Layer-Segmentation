"""## 3. Data Preprocessing"""

# Normalize images to [0, 1] and convert to float32
images = np.clip(images, 0, 255).astype(np.float32) / 255.0

# Resize images to the target shape (256, 256)
def resize_images(images, target_size=(256, 256)):
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        resized_images.append(resized_img)
    return np.array(resized_images)

# Resize the loaded images
images_resized = resize_images(images)

# Resize masks (if needed) to ensure they are the same size as images
def resize_masks(masks, target_size=(256, 256)):
    resized_masks = []
    for mask in masks:
        resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        resized_masks.append(resized_mask)
    return np.array(resized_masks)

masks_resized = resize_masks(masks)

# Convert masks to one-hot encoding
num_classes = 8  # Adjust based on your dataset
masks_categorical = tf.keras.utils.to_categorical(masks_resized, num_classes=num_classes)

# Ensure images are in shape (num_samples, height, width, 1)
images_resized = np.expand_dims(images_resized, axis=-1)  # Now shape is (num_samples, height, width, 1)

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(images_resized, masks_categorical, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of y_val:", y_val.shape)
