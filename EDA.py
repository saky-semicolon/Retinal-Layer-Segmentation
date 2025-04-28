"""## 2. EDA"""

# EDA
print("Images shape:", images.shape)
print("Masks shape:", masks.shape)
print("Images dtype:", images.dtype)
print("Masks dtype:", masks.dtype)
print("Unique mask values:", np.unique(masks))

# Display a few sample images and masks
num_samples_to_display = 3

for i in range(num_samples_to_display):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(images[i], cmap='gray')
    plt.title("Image")

    plt.subplot(1, 2, 2)
    plt.imshow(masks[i], cmap='jet')
    plt.title("Mask")

    plt.show()

# Check for the distribution of values in images
plt.figure(figsize=(8, 5))
plt.hist(images.flatten(), bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Pixel Values in Images')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Check for the distribution of masks
plt.figure(figsize=(8, 5))
plt.hist(masks.flatten(), bins=len(np.unique(masks)), color='lightcoral', edgecolor='black', rwidth=0.8)
plt.title('Distribution of Mask Values')
plt.xlabel('Mask Label')
plt.ylabel('Frequency')
plt.grid(True)
plt.xticks(np.unique(masks))
plt.show()

# Displaying images with their masks side by side
def display_images_with_masks(images, masks, num_samples=5):
    for i in range(num_samples):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(images[i], cmap='gray')
        plt.title("Image")

        plt.subplot(1, 2, 2)
        plt.imshow(masks[i], cmap='jet')
        plt.title("Mask")

        plt.show()

display_images_with_masks(images, masks, num_samples=5)

# Compute and display the mean and standard deviation of pixel values for images.
image_mean = np.mean(images)
image_std = np.std(images)
print(f"Mean of pixel values in images: {image_mean:.2f}")
print(f"Standard deviation of pixel values in images: {image_std:.2f}")

# Check class balance within the masks
mask_values, mask_counts = np.unique(masks, return_counts=True)
print("Mask Value Counts:")
for value, count in zip(mask_values, mask_counts):
  print(f"  Value {value}: {count}")
plt.figure(figsize=(10, 6))
plt.bar(mask_values, mask_counts, color='mediumseagreen', edgecolor='black')
plt.title('Class Balance in Masks')
plt.xlabel('Mask Value (Class)')
plt.ylabel('Number of Pixels')
plt.grid(axis='y')
plt.xticks(mask_values)
plt.show()
