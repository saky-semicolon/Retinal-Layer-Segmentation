"""## 4. Training Strategy

### Defining Custom Loss & Metrics

📌 **Description:**  
Defines evaluation metrics to assess model performance.

✅ **Metrics Implemented:**

- **Dice Coefficient** → Measures segmentation overlap.
- **Jaccard Coefficient** → Measures intersection over union (IoU).
- **Custom Loss Function** → Uses categorical cross-entropy + Dice loss for better segmentation accuracy.
"""

# Dice and Jaccard coefficients
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def jaccard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return intersection / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + K.epsilon())

# Custom loss function
def customized_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred) + 0.5 * (1 - dice_coef(y_true, y_pred))
