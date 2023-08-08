from concurrent import futures
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread
from typing import Dict, List, Tuple

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from tqdm.auto import tqdm

from .build_vrt import build_vrt
from .model_settings import Settings


def create_patch_metadata(scene_settings: Settings) -> List[Dict[str, int]]:
    """
    Creates metadata for each patch of the scene based on the provided settings.

    This function processes an image raster and creates a list of dictionaries
    representing the metadata for each image patch. It iterates through the image
    with a sliding window approach, with each window representing a patch.
    Each dictionary contains the top, bottom, left, and right pixel indices that
    define the patch within the original image.

    Note that every second row is offset to the left by half the patch size. If
    the offset results in the omission of a patch on the right edge, an extra
    patch is added.

    Args:
        scene_settings (Settings): The settings of the scene which includes the
            path to the input image, the desired patch size, and the number of
            overlapping pixels between adjacent patches.

    Returns:
        List[Dict[str, int]]: A list of dictionaries, each containing metadata
            for a patch. The metadata includes 'left', 'top', 'right', and 'bottom'
            keys, referring to the corresponding pixel indices in the image.

    Raises:
        FileNotFoundError: If the provided image file path in the scene settings
            does not exist.
    """
    full_size = (10980, 10980)

    width, height = [int(scene_settings.scale_factor * x) for x in full_size]

    patch_size = scene_settings.patch_size
    half_patch_width = (patch_size - scene_settings.patch_overlap_px) // 2

    patch_metadata = []
    for row, top in enumerate(
        range(0, height, patch_size - scene_settings.patch_overlap_px)
    ):
        for left in range(0, width, patch_size - scene_settings.patch_overlap_px):
            if row % 2 != 0:
                left = max(0, left - half_patch_width)

            patch_metadata.append(
                {
                    "left": min(left, width - patch_size),
                    "top": min(top, height - patch_size),
                    "right": min(left + patch_size, width),
                    "bottom": min(top + patch_size, height),
                }
            )

        # Append extra patch for odd rows if needed
        if row % 2 != 0 and width > patch_metadata[-1]["right"]:
            patch_metadata.append(
                {
                    "left": width - patch_size,
                    "top": min(top, height - patch_size),
                    "right": width,
                    "bottom": min(top + patch_size, height),
                }
            )

    return patch_metadata


def get_files_from_safe(scene_settings: Settings) -> List[Path]:
    """
    Retrieves the file paths of the required bands from the directory specified
    in the scene settings. This function works with Sentinel-2 L1C data.

    Args:
        scene_settings (Settings): The settings of the scene from which files
        are to be retrieved. This includes the directory path and the processing
        level of the images.

    Returns:
        List[Path]: A list of paths to the raster files for the required bands.

    Note:
        This function raises an exception if the required band file is not found
        in the directory.
    """

    bands_to_patch_paths = []

    if scene_settings.processing_level == "L1C":
        bands_to_patch_paths = [
            next(scene_settings.sent_safe_dir.rglob(f"*IMG_DATA/*{band}.jp2"))
            for band in scene_settings.required_bands
        ]

    return bands_to_patch_paths


def open_and_resize(
    open_data_list: tuple, scene_progress_pbar: tqdm, pbar_inc: float
) -> np.ndarray:
    """
    Opens a raster file, reads it into a numpy array and resizes it to the
    required size if the original shape is not (10980, 10980). Updates a
    progress bar as it reads and resizes.

    Args:
        open_data_list (tuple): A tuple containing the path to the input raster
        file and the scale factor to resize the array.

        scene_progress_pbar (tqdm): The tqdm progress bar object that tracks the
        progress of the resizing operation.

        pbar_inc (float): The amount to increment the progress bar after
        resizing each array.

    Returns:
        np.ndarray: The resized array read from the raster file.
    """

    input_path, scale_factor = open_data_list
    full_size = (10980, 10980)

    required_size = tuple([int(scale_factor * x) for x in full_size])

    band_array = rio.open(input_path).read(
        1, out_shape=(required_size), resampling=Resampling.nearest
    )
    scene_progress_pbar.update(pbar_inc)
    return band_array


def shift_patch_inwards(
    patch: np.ndarray, resized_arrays: np.ndarray, patch_meta: Dict[str, int]
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Shifts the given patch inwards if it contains rows or columns of zeros."""

    max_bottom, max_right = resized_arrays.shape[1:3]

    # Both top and bottom sides are not zero-filled
    if np.any(patch[:, 0, :]) or np.any(patch[:, -1, :]):
        while (
            not np.any(patch[:, 0, :]) and patch_meta["bottom"] < max_bottom
        ):  # check top row
            patch = patch[:, 1:, :]
            patch_meta["top"] += 1
            patch_meta["bottom"] += 1

        while not np.any(patch[:, -1, :]) and patch_meta["top"] > 0:
            patch = patch[:, :-1, :]
            patch_meta["bottom"] -= 1
            patch_meta["top"] -= 1

    # Both sides are not zero-filled
    if np.any(patch[:, :, 0]) or np.any(patch[:, :, -1]):
        while (
            not np.any(patch[:, :, 0]) and patch_meta["right"] < max_right
        ):  # check left column
            patch = patch[:, :, 1:]
            patch_meta["left"] += 1
            patch_meta["right"] += 1

        while (
            not np.any(patch[:, :, -1]) and patch_meta["left"] > 0
        ):  # check right column
            patch = patch[:, :, :-1]
            patch_meta["right"] -= 1
            patch_meta["left"] -= 1

    patch = resized_arrays[
        :,
        patch_meta["top"] : patch_meta["bottom"],
        patch_meta["left"] : patch_meta["right"],
    ]

    return patch, patch_meta


def make_patches(
    resized_arrays: np.ndarray, patch_meta_list: List[Dict[str, int]]
) -> Tuple[List[np.ndarray], List[Dict[str, int]]]:
    """Generates patches from the given resized array based on the metadata
    list.

    Args:
        resized_arrays (np.ndarray): The array of resized patches.
        patch_meta_list (List[Dict[str, int]]): List of dictionaries with
        metadata about each patch.

    Returns:
        Tuple[List[np.ndarray], List[Dict[str, int]]]: List of generated patches
        and the corresponding
            list of metadata dictionaries.
    """
    patch_arrays = []
    patch_meta_list_subset = []
    for patch_meta in patch_meta_list:
        patch = resized_arrays[
            :,
            patch_meta["top"] : patch_meta["bottom"],
            patch_meta["left"] : patch_meta["right"],
        ]

        # only process if patch contains some valid data
        if patch.sum() != 0:
            # try to shift patch in if it contains zeros
            if patch.min() == 0:
                patch, patch_meta = shift_patch_inwards(
                    patch, resized_arrays, patch_meta
                )
            # if we have shifted a patch on top of another, skip it
            if patch_meta not in patch_meta_list_subset:
                # convert to fp32 now so we dont need to do it at prediction time
                patch_arrays.append(patch.astype(np.float32))
                patch_meta_list_subset.append(patch_meta)

    return (patch_arrays, patch_meta_list_subset)


def scene_to_no_data_mask(resized_arrays: np.ndarray) -> np.ndarray:
    """
    Generates a no data mask for the given resized array. The mask is a boolean array
    """
    non_zero_counts = np.count_nonzero(resized_arrays, axis=0)
    # if there are more than 2 non-zero values in a pixel, it is not a nodata pixel
    nodata_mask = non_zero_counts >= 2
    return nodata_mask


def generate_patches(
    scene_settings: Settings,
) -> Tuple[List[np.ndarray], List[Dict], futures.Future]:
    """
    Generates patches from a scene, given the scene settings. The patches are
    created from the required bands in the SAFE directory of the scene. The
    patches are saved in the path specified by the VRT file. The number of cores
    used for tiling can be specified. An overlap between patches can also be
    specified.

    Args:
        scene_settings (Settings): The settings of the scene from which patches
        are to be generated. It includes details such as the directory path for
        input images, VRT file path, number of cores for tiling, pixel overlap
        between patches, and required bands for patch generation.

    Returns:
        List: A list containing arrays of patches and corresponding patch
        metadata.
    """
    scene_settings.scene_progress_pbar.disable = False
    scene_settings.scene_progress_pbar.desc = "Making patches"

    bands_to_patch_paths = get_files_from_safe(scene_settings)
    # build vrt in a separate thread to avoid blocking
    vrt_thread = Thread(target=build_vrt, args=(scene_settings, bands_to_patch_paths))
    vrt_thread.start()

    # Create the patch metadata for each patch to be generated
    patch_meta_list = create_patch_metadata(scene_settings)

    # Load the entire dataset and its metadata
    open_data_list = []
    for band in bands_to_patch_paths:
        open_data_list.append((band, scene_settings.scale_factor))

    pbar_inc = 33 / len(scene_settings.required_bands)
    with ThreadPool(scene_settings.processing_threads) as tp:
        # Create a partial function with fixed pbar argument

        open_and_resize_with_pbar = partial(
            open_and_resize,
            scene_progress_pbar=scene_settings.scene_progress_pbar,
            pbar_inc=pbar_inc,
        )

        resized_arrays = list(tp.imap(open_and_resize_with_pbar, open_data_list))

    resized_arrays = np.array(resized_arrays)

    # build nodata mask in a separate thread to avoid blocking
    executor = futures.ThreadPoolExecutor(max_workers=1)
    nodata_mask_future = executor.submit(scene_to_no_data_mask, resized_arrays)

    # cut patches out of input data array
    patch_arrays, patch_metadata = make_patches(resized_arrays, patch_meta_list)
    # convert to numpy array

    # make sure the vrt is built before returning
    vrt_thread.join()

    return patch_arrays, patch_metadata, nodata_mask_future
