from __future__ import division
from keras.models import Model
# from keras.layers.core import Dropout, Activation
from keras.layers import Dropout, Activation
from keras.layers import Input
from keras.layers import Concatenate  # Or Add, Multiply, etc.

from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
import keras.backend as K
import h5py
from eltwise_product import EltWiseProduct
import math
from config import shape_r_gt, shape_c_gt
import tensorflow as tf
import numpy as np
from config import experiment_id
import os



def get_latest_checkpoint(experiment_id):
    """
    Function to find the latest checkpoint file in the experiment directory.
    :param experiment_id: The experiment ID.
    :return: Path to the latest checkpoint file, or None if no checkpoint is found.
    """
    checkpoint_dir = os.path.join('checkpoints', experiment_id)
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.h5')]
    if not checkpoint_files:
        return None

    # Sort checkpoint files by epoch number (assuming the filename format is 'weights.mlnet.{epoch:02d}-{val_loss:.4f}.h5')
    checkpoint_files.sort(key=lambda x: int(x.split('.')[2].split('-')[0]))
    latest_checkpoint = checkpoint_files[-1]
    return os.path.join(checkpoint_dir, latest_checkpoint)


def get_weights_vgg16(f, id):
    g = f['layer_{}'.format(id)]
    weights = [g['param_{}'.format(p)][:] for p in range(g.attrs['nb_params'])]
    
    # Transpose kernel weights to match TensorFlow's expected shape
    if len(weights[0].shape) == 4:  # Check if it's a kernel (not bias)
        weights[0] = np.transpose(weights[0], (2, 3, 1, 0))  # Transpose to (height, width, input_channels, output_channels)
    
    return weights

def ml_net_model(img_rows=480, img_cols=640, downsampling_factor_net=8, downsampling_factor_product=10):
    f = h5py.File("vgg16_weights.h5")
    
    # Check if a checkpoint exists for the current experiment
    # checkpoint_path = get_latest_checkpoint(experiment_id)
    # if checkpoint_path:
    #     print(f"Loading model weights from checkpoint: {checkpoint_path}")
    #     model = tf.keras.models.load_model(checkpoint_path, custom_objects={'loss': loss})
    #     return model
    # else:
    #     print("No checkpoint found. Loading VGG16 weights instead.")
    #     f = h5py.File("vgg16_weights.h5")

    # input_ml_net = Input(shape=(3, img_rows, img_cols))
    input_ml_net = Input(shape=(img_rows, img_cols, 3))  # Channels-last format

    #########################################################
    # FEATURE EXTRACTION NETWORK							#
    #########################################################

    # Layer 1
    weights = get_weights_vgg16(f, 1)
    kernel_initializer = tf.keras.initializers.Constant(weights[0])  # Kernel weights
    bias_initializer = tf.keras.initializers.Constant(weights[1])    # Bias weights
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer)(input_ml_net)

    weights = get_weights_vgg16(f, 3)
    kernel_initializer = tf.keras.initializers.Constant(weights[0])
    bias_initializer = tf.keras.initializers.Constant(weights[1])
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer)(conv1_1)
    conv1_pool = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(conv1_2)

    # Layer 2
    weights = get_weights_vgg16(f, 6)
    kernel_initializer = tf.keras.initializers.Constant(weights[0])
    bias_initializer = tf.keras.initializers.Constant(weights[1])
    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer)(conv1_pool)

    weights = get_weights_vgg16(f, 8)
    kernel_initializer = tf.keras.initializers.Constant(weights[0])
    bias_initializer = tf.keras.initializers.Constant(weights[1])
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer)(conv2_1)
    conv2_pool = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(conv2_2)

    # Layer 3
    weights = get_weights_vgg16(f, 11)
    kernel_initializer = tf.keras.initializers.Constant(weights[0])
    bias_initializer = tf.keras.initializers.Constant(weights[1])
    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer)(conv2_pool)

    weights = get_weights_vgg16(f, 13)
    kernel_initializer = tf.keras.initializers.Constant(weights[0])
    bias_initializer = tf.keras.initializers.Constant(weights[1])
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer)(conv3_1)

    weights = get_weights_vgg16(f, 15)
    kernel_initializer = tf.keras.initializers.Constant(weights[0])
    bias_initializer = tf.keras.initializers.Constant(weights[1])
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer)(conv3_2)
    conv3_pool = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(conv3_3)

    # Layer 4
    weights = get_weights_vgg16(f, 18)
    kernel_initializer = tf.keras.initializers.Constant(weights[0])
    bias_initializer = tf.keras.initializers.Constant(weights[1])
    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer)(conv3_pool)

    weights = get_weights_vgg16(f, 20)
    kernel_initializer = tf.keras.initializers.Constant(weights[0])
    bias_initializer = tf.keras.initializers.Constant(weights[1])
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer)(conv4_1)

    weights = get_weights_vgg16(f, 22)
    kernel_initializer = tf.keras.initializers.Constant(weights[0])
    bias_initializer = tf.keras.initializers.Constant(weights[1])
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer)(conv4_2)
    conv4_pool = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(conv4_3)

    # Layer 5
    weights = get_weights_vgg16(f, 25)
    kernel_initializer = tf.keras.initializers.Constant(weights[0])
    bias_initializer = tf.keras.initializers.Constant(weights[1])
    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer)(conv4_pool)

    weights = get_weights_vgg16(f, 27)
    kernel_initializer = tf.keras.initializers.Constant(weights[0])
    bias_initializer = tf.keras.initializers.Constant(weights[1])
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer)(conv5_1)

    weights = get_weights_vgg16(f, 29)
    kernel_initializer = tf.keras.initializers.Constant(weights[0])
    bias_initializer = tf.keras.initializers.Constant(weights[1])
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer)(conv5_2)



    #########################################################
    # ENCODING NETWORK										#
    #########################################################

    # Adjust conv4_pool and conv5_3 to match conv3_pool's channels (256)
    conv4_pool_adjusted = Conv2D(256, (1, 1), activation='relu', padding='same')(conv4_pool)
    conv5_3_adjusted = Conv2D(256, (1, 1), activation='relu', padding='same')(conv5_3)

    # Concatenate along the channel axis
    concatenated = Concatenate(axis=-1)([conv3_pool, conv4_pool_adjusted, conv5_3_adjusted])
    dropout = Dropout(0.5)(concatenated)

    int_conv = Conv2D(64, (3, 3), kernel_initializer='glorot_normal', activation='relu', padding='same')(dropout)
    pre_final_conv = Conv2D(1, (1, 1), kernel_initializer='glorot_normal', activation='relu')(int_conv)

    #########################################################
    # PRIOR LEARNING										#
    #########################################################
    rows_elt = math.ceil(img_rows / downsampling_factor_net) // downsampling_factor_product
    cols_elt = math.ceil(img_cols / downsampling_factor_net) // downsampling_factor_product
    eltprod = EltWiseProduct(init='zero', W_regularizer=l2(1/(rows_elt*cols_elt)))(pre_final_conv)
    output_ml_net = Activation('relu')(eltprod)

    model = Model(inputs=[input_ml_net], outputs=[output_ml_net])

    return model


def loss(y_true, y_pred):
    # Compute the maximum value of y_pred across height and width dimensions
    max_y = tf.reduce_max(y_pred, axis=[1, 2], keepdims=True)  # Shape: (batch_size, 1, 1, channels)

    print("y_true.shape:", y_true.shape)
    print("y_pred.shape:", y_pred.shape)
    print("max_y.shape:", max_y.shape)
    
    # Normalize y_pred by its maximum value (add epsilon to avoid division by zero)
    y_pred_normalized = y_pred / (max_y + tf.keras.backend.epsilon())
    
    # Compute the loss
    loss_value = tf.reduce_mean(tf.square(y_pred_normalized - y_true) / (1 - y_true + 0.1))
    
    return loss_value