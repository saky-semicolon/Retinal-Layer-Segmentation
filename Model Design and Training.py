"""## 5. Model Design and Training"""

# Define the SegNet model
def segnet_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2), strides=(2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2), strides=(2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2), strides=(2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2), strides=(2, 2))(c4)

    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    p5 = layers.MaxPooling2D((2, 2), strides=(2, 2))(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(p5)
    u6 = layers.concatenate([u6, c5])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c4])
    c7 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c3])
    c8 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c2])
    c9 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c9)

    u10 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c9)
    u10 = layers.concatenate([u10, c1])
    c10 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u10)
    c10 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c10)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c10)  # Softmax for multi-class

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# Initialize and compile the SegNet model
input_shape = (256, 256, 1)  # Grayscale input shape
model = segnet_model(input_shape, num_classes=num_classes)

# model = segnet_model(input_shape=(256, 256, 1), num_classes=8)
model.summary()

import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import os

def plot_model_architecture(model, filename="model_architecture.png"):
    """Plots the architecture of a Keras model and displays it."""

    # Ensure the model visualization is saved
    plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True, dpi=100)

    # Display the image using matplotlib
    img = plt.imread(filename)
    plt.figure(figsize=(50, 30))
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.show()

plot_model_architecture(model)

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss=customized_loss,
              metrics=['accuracy', dice_coef, jaccard_coef])

# Callbacks for training
lr_reducer = ReduceLROnPlateau(factor=0.5, patience=6, min_lr=1e-6)
csv_logger = CSVLogger('training_log.csv')
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs', write_graph=True, write_images=True)

# Train the model
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=32,
                    epochs=100,
                    callbacks=[lr_reducer, csv_logger, model_checkpoint, tensorboard])
