# preprocessing.py

import numpy as np
from tifffile import memmap, imread, imwrite
from scipy.ndimage import shift
import cv2
from utilities import remove_stripes, apply_skew_correction, apply_movement_correction

# This module handles preprocessing tasks such as gain and dark correction,
# stripe artifact removal, and motion correction.

def preprocess_projections(config, logger):
    """
    Applies preprocessing steps to the projection data, including gain/dark corrections,
    stripe removal, and motion correction.

    Args:
        config (dict): Configuration dictionary containing paths and parameters.
        logger (Logger): Logger object for recording processing steps.
    """
    # Load gain and dark-field images using memory mapping
    gain = memmap(config["gain_path"], mode='r')
    dark = memmap(config["dark_path"], mode='r')
    gain_post = memmap(config["gain_path_post"], mode='r')
    dark_post = memmap(config["dark_path_post"], mode='r')

    # Iterate through projection files
    projection_files = config["proj_dir"]  # Adjust based on file loading utility

    for idx, projection_path in enumerate(projection_files):
        # Load projection with memory mapping
        projection = memmap(projection_path, mode='r')

        # Apply gain and dark-field corrections
        corrected = (projection - dark) / (gain - dark)

        # Apply movement correction if enabled
        corrected = apply_movement_correction(corrected, config, idx)

        # Remove stripe artifacts
        corrected = remove_stripes(corrected, config["stripe_removal"])

        # Apply skew correction
        corrected = apply_skew_correction(corrected, config)

        logger.info(f"Processed projection {idx + 1}")

        # Save the corrected projection using memory mapping
        save_path = projection_path.replace(".tif", "_corrected.tif")
        save_with_memmap(save_path, corrected)


def save_with_memmap(output_path, data):
    """
    Save processed data using memory mapping.

    Args:
        output_path (str): Path to save the processed data.
        data (np.ndarray): Data to save.
    """
    memmap = np.memmap(output_path, dtype=data.dtype, mode='w+', shape=data.shape)
    memmap[:] = data[:]
    memmap.flush()


def apply_movement_correction(projection, config, idx):
    """
    Applies motion correction to a projection image.

    Args:
        projection (np.ndarray): The projection image to correct.
        config (dict): Configuration dictionary containing motion correction parameters.
        idx (int): Index of the current projection.

    Returns:
        np.ndarray: Motion-corrected projection.
    """
    if config["movement_correction"]["enabled"]:
        y_shift = config["movement_correction"]["y_shift"]
        x_shift = config["movement_correction"]["x_shift"]
        projection = shift(projection, [y_shift * idx, x_shift * idx], cval=0)
    return projection


def apply_skew_correction(projection, config):
    """
    Applies skew correction to a projection image.

    Args:
        projection (np.ndarray): The projection image to correct.
        config (dict): Configuration dictionary containing skew correction parameters.

    Returns:
        np.ndarray: Skew-corrected projection.
    """
    if config["skew_correction"]["enabled"]:
        angle = config["skew_correction"]["angle"]
        h, w = projection.shape
        cX, cY = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1)
        projection = cv2.warpAffine(projection, M, (w, h))
    return projection
