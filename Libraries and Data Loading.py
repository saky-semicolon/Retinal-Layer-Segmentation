## 1. Importing Necessary Libraries
"""

# Commented out IPython magic to ensure Python compatibility.
import keras
from keras.layers import Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import UpSampling2D
from keras.layers import Concatenate
from keras.layers import Lambda
from keras.utils import to_categorical
import tensorflow as tf
from keras.layers import Reshape
from keras import backend as K
from keras import regularizers, optimizers
# %matplotlib inline

from keras.callbacks import ReduceLROnPlateau, CSVLogger,EarlyStopping,ModelCheckpoint

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load images and masks
images = np.load(r"/kaggle/input/retinal-layer-segmentation/Images_npy_files/resized_images.npy")  # Shape: (num_samples, height, width)
masks = np.load(r"/kaggle/input/retinal-layer-segmentation/Images_npy_files/resized_labeledimages.npy")  # Shape: (num_samples, height, width)
