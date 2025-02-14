from __future__ import division
import cv2
import numpy as np


def padding(img, shape_r=480, shape_c=640, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def preprocess_images(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 3))

    for i, path in enumerate(paths):
        original_image = cv2.imread(path)
        # Ensure the image was read successfully
        if original_image is None:
            raise ValueError(f"Failed to load image at {path}. Check the path and integrity of the image.")
        # Ensure the image has at least one channel (e.g., RGB)
        if original_image.shape[-1] != 3:  # Check if the image has 3 channels (RGB)
            raise ValueError(f"Image at {path} does not have 3 channels. It has {original_image.shape[-1]} channels.")
        
        padded_image = padding(original_image, shape_r, shape_c, 3)
        ims[i] = padded_image / 255.0

        # # Print the RGB values of each channel for original_image
        # print(f"Image {i+1} RGB values:")
        # print(f"Red channel min: {np.min(original_image[:, :, 0])}, max: {np.max(original_image[:, :, 0])}")
        # print(f"Green channel min: {np.min(original_image[:, :, 1])}, max: {np.max(original_image[:, :, 1])}")
        # print(f"Blue channel min: {np.min(original_image[:, :, 2])}, max: {np.max(original_image[:, :, 2])}")

    # Calculate the mean values for each channel across all images
    mean_r = np.mean(ims[:, :, :, 0])
    mean_g = np.mean(ims[:, :, :, 1])
    mean_b = np.mean(ims[:, :, :, 2])

    # Subtract the mean values from each channel
    ims[:, :, :, 0] -= mean_r
    ims[:, :, :, 1] -= mean_g
    ims[:, :, :, 2] -= mean_b

    ims = np.clip(ims, 0, 1)
    ims = ims.transpose((0, 3, 1, 2))

    # print(f"Output of preprocess_images after mean subtraction - min: {np.min(ims)}, max: {np.max(ims)}")

    return ims


# def preprocess_maps(paths, shape_r, shape_c):
#     ims = np.zeros((len(paths), 1, shape_r, shape_c))

#     for i, path in enumerate(paths):
#         original_map = cv2.imread(path, 0)
#         padded_map = padding(original_map, shape_r, shape_c, 1)
#         ims[i, 0] = padded_map.astype(np.float32)
#         ims[i, 0] /= 255.0

#         # Debug: Check the input and output
#         print(f"Input to preprocess_maps - min: {np.min(original_map)}, max: {np.max(original_map)}")
#         print(f"Output of preprocess_maps - min: {np.min(ims[i, 0])}, max: {np.max(ims[i, 0])}")

#     return ims

def preprocess_maps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), 1, shape_r, shape_c))

    for i, path in enumerate(paths):
        original_map = cv2.imread(path, 0)  # Load the map as a grayscale image
        if original_map is None:
            raise ValueError(f"Failed to load ground truth map at {path}. Check the path and integrity of the image.")

        # Ensure the map is not all zeros
        if np.max(original_map) == 0:
            print(f"Warning: Ground truth map {i} is all zeros. Check the file: {path}")

        padded_map = padding(original_map, shape_r, shape_c, 1)
        ims[i, 0] = padded_map.astype(np.float32) / 255.0

    return ims


def postprocess_predictions(pred, shape_r, shape_c):
    """
    Resize and normalize a prediction map to the target size.
    :param pred: Input prediction map.
    :param shape_r: Target height of the output map.
    :param shape_c: Target width of the output map.
    :return: Resized and normalized prediction map.
    """
    predictions_shape = pred.shape
    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    # Debug: Check the input and output
    print(f"Input to postprocess_predictions - min: {np.min(pred)}, max: {np.max(pred)}")
    max_val = np.max(img)
    if max_val == 0:
        print("Warning: max value is 0 in postprocess_predictions")
        return np.zeros_like(img)
    print(f"Output of postprocess_predictions - min: {np.min(img)}, max: {np.max(img)}")
    return (img / max_val) * 255
    # return img
