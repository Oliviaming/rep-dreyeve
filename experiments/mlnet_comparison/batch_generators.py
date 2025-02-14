import numpy as np
import random
import matplotlib.pyplot as plt
from os.path import join
from utils import preprocess_images, preprocess_maps

from config import dreyeve_train_seq, dreyeve_test_seq
from config import train_frame_range, val_frame_range, test_frame_range
from config import DREYEVE_DIR
from config import shape_r, shape_c, shape_r_gt, shape_c_gt


def load_batch(batchsize, mode, gt_type):
    """
    This function loads a batch for mlnet training.

    :param batchsize: batchsize.
    :param mode: choose among [`train`, `val`, `test`].
    :param gt_type: choose among [`sal`, `fix`].
    :return: X and Y as ndarray having shape (b, c, h, w).
    """
    assert mode in ['train', 'val', 'test'], 'Unknown mode {} for dreyeve batch loader'.format(mode)
    assert gt_type in ['fix'], 'Unknown gt_type {} for dreyeve batch loader'.format(gt_type)

    if mode == 'train':
        sequences = dreyeve_train_seq
        allowed_frames = train_frame_range
        allow_mirror = True
    elif mode == 'val':
        sequences = dreyeve_train_seq
        allowed_frames = val_frame_range
        allow_mirror = False
    elif mode == 'test':
        sequences = dreyeve_test_seq
        allowed_frames = test_frame_range
        allow_mirror = False

    x_list = []
    y_list = []

    for b in range(0, batchsize):
        seq = random.choice(sequences)
        fr = random.choice(allowed_frames)

        # x_list.append(join(DREYEVE_DIR, '{:02d}'.format(seq), 'mean_frame.png'.format(fr))) 
        x_list.append(join(DREYEVE_DIR, '{:02d}'.format(seq), 'mean_frame.png').replace("\\", "/"))
        y_list.append(join(DREYEVE_DIR, '{:02d}'.format(seq), 'mean_gt.png').replace("\\", "/"))
    

    X = preprocess_images(x_list, shape_r=shape_r, shape_c=shape_c)

    # Correctly transpose from (batch_size, channels, height, width) to (batch_size, height, width, channels)
    X = X.transpose((0, 2, 3, 1))  

    Y = preprocess_maps(y_list, shape_r=shape_r_gt, shape_c=shape_c_gt)
    Y = Y.transpose((0, 2, 3, 1))  

    print(f"load batch X - min: {np.min(X)}, max: {np.max(X)}")
    print(f"load batch Y - min: {np.min(Y)}, max: {np.max(Y)}")

    return np.float32(X), np.float32(Y)


def generate_batch(batchsize, mode, gt_type):
    """
    Yields dreyeve batches for mlnet training.

    :param batchsize: batchsize.
    :param mode: choose among [`train`, `val`, `test`].
    :param gt_type: choose among [`sal`, `fix`].
    :return: X and Y as ndarray having shape (b, c, h, w).
    """
    while True:
        yield load_batch(batchsize, mode, gt_type)



# # Test script
# if __name__ == '__main__':
#     X, Y = load_batch(4, mode='train', gt_type='fix')

#     print(X.shape)  # Should be (batch_size, height, width, channels)
#     print(Y.shape)  # Should be (batch_size, height, width, 1)

#     # Visualize the batch
#     fig, axes = plt.subplots(2, 4, figsize=(20, 5))
#     for i in range(4):
#         # Display the input image (X)
#         img = X[i]  # Shape: (height, width, channels)
#         img = np.clip(img, 0, 1)  # Ensure values are in [0, 1] for visualization
#         axes[0, i].imshow(img)
#         axes[0, i].axis('off')
#         axes[0, i].set_title(f'Image {i+1}')
        
#         # Display the ground truth map (Y)
#         gt_map = Y[i, :, :, 0]  # Shape: (height, width)
#         gt_map = np.clip(gt_map, 0, 1)  # Ensure values are in [0, 1] for visualization
#         axes[1, i].imshow(gt_map, cmap='gray')
#         axes[1, i].axis('off')
#         axes[1, i].set_title(f'GT {i+1}')
    
#     plt.show()

