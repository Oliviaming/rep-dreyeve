import numpy as np
import cv2

import os
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from os.path import join

# from keras_dl_modules.custom_keras_extensions.callbacks import Checkpointer

from batch_generators import load_batch
from config import batchsize, each_experiment_id
from config import shape_r, shape_c
from model import get_last_experiment_id
from utils import postprocess_predictions

# convert each_experiment_id type to string
experiment_id = str(each_experiment_id)


# ==========================================================
# REPLACEMENT FOR `computer_vision_utils.stitching.stitch_together`
# ==========================================================
def stitch_together(images, layout):
    """
    Stitch images into a grid.
    Example: stitch_together([img1, img2, img3], layout=(1, 3)) creates a single row of 3 images.
    """
    rows, cols = layout
    # Ensure images are uint8
    images = [img.astype(np.uint8) for img in images]

    # Reshape images into grid
    stitched_rows = []
    for row_idx in range(rows):
        row_images = images[row_idx*cols : (row_idx+1)*cols]
        stitched_row = np.hstack(row_images)
        stitched_rows.append(stitched_row)

    stitched_image = np.vstack(stitched_rows)
    return stitched_image

# ==========================================================
# REPLACEMENT FOR `computer_vision_utils.io_helper.normalize`
# ==========================================================
def normalize(img):
    """
    Normalize image to [0, 255] and convert to uint8.
    """
    if img.dtype == np.float32:
        # Replace NaN/Inf with 0
        img = np.nan_to_num(img)
        img = img - np.min(img)
        img_max = np.max(img)
        if img_max == 0:
            img = np.zeros_like(img, dtype=np.uint8)
        else:
            img = (img / img_max * 255).astype(np.uint8)
    return img


class PredictionCallback(Callback):
    """
    Callback to perform some debug predictions, on epoch end.
    Loads a batch, predicts it and saves images in `predictions/${experiment_id}`.

    :param experiment_id: the experiment id.
    """

    def __init__(self, experiment_id):

        super(PredictionCallback, self).__init__()

        # create output directories if not existent
        out_dir_path = join('predictions', experiment_id)
        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)

        # set out dir as attribute of PredictionCallback
        self.out_dir_path = out_dir_path

    def on_train_begin(self, logs=None):
            self.on_epoch_end(epoch='begin', logs=logs)

    def on_epoch_end(self, epoch, logs={}):

        # create the folder for predictions
        cur_out_dir = join(self.out_dir_path, 'epoch_{:03d}'.format(epoch) if epoch != 'begin' else 'begin')
        if not os.path.exists(cur_out_dir):
            os.makedirs(cur_out_dir)

        # load and predict
        X, Y = load_batch(batchsize=batchsize, mode='val', gt_type='fix')
        X = X.transpose((0, 1, 2, 3))
        P = self.model.predict(X)

        print(f"Model predictions (P) - min: {np.min(P)}, max: {np.max(P)}")
        print(f"Model input (X) - min: {np.min(X)}, max: {np.max(X)}")
        print(f"Model ground truth (Y) - min: {np.min(Y)}, max: {np.max(Y)}")

        for b in range(0, batchsize):
            x = X[b]
            x = normalize(x)
            print(f"x min/max: {np.min(x)}, {np.max(x)}")

            print(f"P shape: {P.shape}")
            p = postprocess_predictions(P[b, 0], shape_r, shape_c)
            p = np.tile(np.expand_dims(p, axis=2), reps=(1, 1, 3))
            print(f"before Normal p min/max: {np.min(p)}, {np.max(p)}")
            p = normalize(p)
            print(f"p min/max: {np.min(p)}, {np.max(p)}")


            print(f"Y shape: {Y.shape}")
            y = postprocess_predictions(Y[b, 0], shape_r, shape_c)
            y = np.tile(np.expand_dims(y, axis=2), reps=(1, 1, 3))
            print(f"before Normal y min/max: {np.min(y)}, {np.max(y)}")
            y = normalize(y)
            print(f"y min/max: {np.min(y)}, {np.max(y)}")

            # # Ground truth (Y) does not need postprocessing
            # y = Y[b, 0]  # Extract the ground truth map
            # y = np.tile(np.expand_dims(y, axis=2), reps=(1, 1, 3))  # Convert to 3 channels
            # y = (y * 255).astype(np.uint8)  # Scale to [0, 255] for visualization


            # stitch and save
            stitch = stitch_together([x, p, y], layout=(1, 3))
            cv2.imwrite(join(cur_out_dir, '{:02d}.png'.format(b)), stitch)


def get_callbacks():
    """
    Function that returns the list of desired Keras callbacks.
    :return: a list of callbacks.
    """
    print("Saved experiment id: {}".format(experiment_id))
    return [
        EarlyStopping(patience=5),
        ModelCheckpoint(  
            filepath=join('checkpoints', experiment_id, 'weights.mlnet.{epoch:02d}-{val_loss:.4f}.h5'),
            save_best_only=True
        ),
        PredictionCallback(experiment_id)
    ]
