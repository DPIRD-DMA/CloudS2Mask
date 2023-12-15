from concurrent import futures
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from tqdm.auto import tqdm

from .build_vrt import build_vrt
from .model_settings import Settings


def create_patch_metadata(scene_settings: Settings) -> List[Dict[str, int]]:
    """
    Creates patch metadata for an image based on scene settings.

    Each patch is defined by pixel indices ('left', 'top', 'right', 'bottom').
    Alternating rows are offset for complete coverage, with additional patches
    added as needed.

    Args:
        scene_settings (Settings): Settings for patch size, overlap, and image scale.

    Returns:
        List[Dict[str, int]]: List of metadata for each patch.
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
    Extracts file paths for required bands from a Sentinel-2 L1C data directory.

    Args:
        scene_settings (Settings): Contains the directory path and processing level for the scene.

    Returns:
        List[Path]: Paths to the raster files of required bands.

    Note:
        An exception is raised if any required band file is not found in the
        directory.
    """

    bands_to_patch_paths = []

    if scene_settings.processing_level == "L1C":
        bands_to_patch_paths = [
            next(scene_settings.sent_safe_dir.rglob(f"*IMG_DATA/*{band}.jp2"))
            for band in scene_settings.required_bands
        ]

    return bands_to_patch_paths


def open_and_resize(
    open_data_list: tuple,
    scene_progress_pbar: tqdm,
    pbar_inc: float,
    resized_arrays: np.ndarray,
) -> None:
    """
    Opens a raster file, reads it as a numpy array, and resizes it to a specific
    shape. It also updates a progress bar during the operation.

    Args:
        open_data_list (tuple): Contains the path to the raster file and the numpy output index.
        scene_progress_pbar (tqdm): Progress bar to monitor the resizing process.
        pbar_inc (float): Increment value for the progress bar after each resize operation.
        resized_arrays (np.ndarray): Preallocated array for storing resized data.
    """

    input_path, index = open_data_list
    required_size = resized_arrays.shape[1:3]

    with rio.open(input_path) as src:
        resized_arrays[index] = src.read(
            1, out_shape=(required_size), resampling=Resampling.nearest
        )

    scene_progress_pbar.update(pbar_inc)


def shift_patch_inwards(
    patch: np.ndarray, resized_arrays: np.ndarray, patch_meta: Dict[str, int]
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Adjusts the given patch to exclude rows or columns of zeros by shifting its edges inward.

    Args:
        patch (np.ndarray): The initial patch array to be adjusted.
        resized_arrays (np.ndarray): The complete array from which the patch is derived.
        patch_meta (Dict[str, int]): Metadata dict for the patch, including 'top', 'bottom', 'left', and 'right' indices.

    Returns:
        Tuple[np.ndarray, Dict[str, int]]: The adjusted patch array and its
        updated metadata.
    """
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


def process_patch(
    resized_arrays: np.ndarray, patch_meta: Dict[str, int]
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, int]]]:
    """
    Extracts and processes a patch from a resized image array using provided
    metadata.

    Args:
        resized_arrays (np.ndarray): Array containing resized image data.
        patch_meta (Dict[str, int]): Metadata dict with 'top', 'bottom', 'left', and 'right' indices for the patch.

    Returns:
        Tuple[Optional[np.ndarray], Optional[Dict[str, int]]]: The processed
        patch and its metadata, or (None, None) if the patch is invalid or
        unnecessary.
    """
    patch = resized_arrays[
        :,
        patch_meta["top"] : patch_meta["bottom"],
        patch_meta["left"] : patch_meta["right"],
    ]

    if patch.sum() != 0:
        if patch.min() == 0:
            patch, patch_meta = shift_patch_inwards(patch, resized_arrays, patch_meta)

        if patch_meta is not None:
            return patch, patch_meta

    return None, None


def make_patches(
    resized_arrays: np.ndarray, patch_meta_list: List[Dict[str, int]]
) -> Tuple[List[np.ndarray], List[Dict[str, int]]]:
    """
    Creates image patches from a resized array using provided metadata,
    employing a thread pool for efficiency.

    Args:
        resized_arrays (np.ndarray): The array containing resized image data.
        patch_meta_list (List[Dict[str, int]]): Metadata for each patch, including position indices.

    Returns:
        Tuple[List[np.ndarray], List[Dict[str, int]]]: A list of processed patch
        arrays and their corresponding metadata.
    """
    patch_arrays = []
    patch_meta_list_subset = []
    patch_generator_threads = multiprocessing.cpu_count() * 2

    with ThreadPoolExecutor(max_workers=patch_generator_threads) as executor:
        results = executor.map(
            lambda p: process_patch(resized_arrays, p), patch_meta_list
        )
    for patch, meta in results:
        if patch is not None and meta not in patch_meta_list_subset:
            patch_arrays.append(patch)
            patch_meta_list_subset.append(meta)

    return patch_arrays, patch_meta_list_subset


def scene_to_no_data_mask(resized_arrays: np.ndarray) -> np.ndarray:
    """
    Creates a no-data mask from a resized image array. The mask is a boolean
    array indicating areas with no data.

    Args:
        resized_arrays (np.ndarray): Array of resized image data.

    Returns:
        np.ndarray: A boolean array representing the no-data mask.
    """
    nodata_mask = np.count_nonzero(resized_arrays, axis=0) >= 2

    return nodata_mask


def generate_patches(
    scene_settings: Settings,
) -> Tuple[List[np.ndarray], List[Dict], np.ndarray]:
    """
    Generates and processes patches from a scene based on specified settings. It
    involves reading required bands, creating virtual rasters, and generating
    patch metadata and nodata masks.

    Args:
        scene_settings (Settings): Configuration for the scene, including input image paths, VRT file path, tiling parameters, patch overlap, and required bands.

    Returns:
        Tuple[List[np.ndarray], List[Dict], np.ndarray]: A tuple containing a
        list of patch arrays, a list of patch metadata dictionaries, and a
        nodata mask for the scene.
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
    for index, band in enumerate(bands_to_patch_paths):
        open_data_list.append((band, index))

    pbar_inc = 33 / len(scene_settings.required_bands)

    size = int(10980 * scene_settings.scale_factor)

    resized_arrays = np.empty([13, size, size], dtype="uint16")

    with ThreadPool(scene_settings.processing_threads) as tp:
        # Create a partial function with fixed pbar argument

        open_and_resize_with_pbar = partial(
            open_and_resize,
            scene_progress_pbar=scene_settings.scene_progress_pbar,
            pbar_inc=pbar_inc,
            resized_arrays=resized_arrays,
        )

        list(tp.imap(open_and_resize_with_pbar, open_data_list))

    # build nodata mask in a separate thread to avoid blocking
    executor = futures.ThreadPoolExecutor(max_workers=1)
    nodata_mask_future = executor.submit(scene_to_no_data_mask, resized_arrays)

    # cut patches out of input data array
    patch_arrays, patch_metadata = make_patches(resized_arrays, patch_meta_list)

    # make sure the vrt is built before returning
    vrt_thread.join()
    mask = nodata_mask_future.result()
    executor.shutdown()

    return patch_arrays, patch_metadata, mask
