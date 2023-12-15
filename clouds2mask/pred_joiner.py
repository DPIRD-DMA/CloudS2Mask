from pathlib import Path
from typing import List

import numpy as np
import rasterio as rio

from .model_settings import Settings


def create_gradient_mask(scene_settings: Settings) -> np.ndarray:
    """
    Creates a gradient mask for blending overlapping image tiles.

    Args:
        scene_settings (Settings): An instance of Settings class containing:
            patch_size: The size of the image in pixels (assumes a square image). patch_overlap_px: The width of the gradient area.

    Returns:
        np.ndarray: An array representing the gradient mask.
    """
    if scene_settings.patch_overlap_px > 0:
        gradient_strength = 1
        gradient = (
            np.ones((scene_settings.patch_size, scene_settings.patch_size), dtype=int)
            * scene_settings.patch_overlap_px
        )
        gradient[:, : scene_settings.patch_overlap_px] = np.tile(
            np.arange(1, scene_settings.patch_overlap_px + 1),
            (scene_settings.patch_size, 1),
        )
        gradient[:, -scene_settings.patch_overlap_px :] = np.tile(
            np.arange(scene_settings.patch_overlap_px, 0, -1),
            (scene_settings.patch_size, 1),
        )
        gradient = gradient / scene_settings.patch_overlap_px
        rotated_gradient = np.rot90(gradient)
        combined_gradient = rotated_gradient * gradient

        combined_gradient = (combined_gradient * gradient_strength) + (
            1 - gradient_strength
        )
    else:
        combined_gradient = np.ones(
            (scene_settings.patch_size, scene_settings.patch_size), dtype=int
        )
    return combined_gradient


def export_geotiff(
    export_array: np.ndarray,
    scene_settings: Settings,
    vrt_meta: dict,
    nodata_mask: np.ndarray,
):
    """
    Exports a GeoTIFF file using provided data and settings.

    Parameters:
    export_array (np.ndarray): The array of data to be exported.
    scene_settings (Settings): Configuration settings for the export process.
    vrt_meta (dict): Metadata for the virtual raster (VRT) of the GeoTIFF.
    nodata_mask (np.ndarray): An array indicating no-data values.

    Note:
    The function modifies the `vrt_meta` dictionary in place with export metadata,
    writes the data to a GeoTIFF file specified in `scene_settings.cloud_mask_path`,
    and updates the progress bar in `scene_settings`.
    """
    export_meta = {
        "count": export_array.shape[0],
        "dtype": "uint8",
        "nodata": None,
        "driver": "GTiff",
        "compress": scene_settings.output_compression,
        "num_threads": "all_cpus",
    }
    vrt_meta.update(export_meta)

    scene_settings.scene_progress_pbar.desc = "Exporting mask"

    with rio.open(scene_settings.cloud_mask_path, "w", **vrt_meta) as dst:
        dst.write(export_array * nodata_mask)
    scene_settings.scene_progress_pbar.update(1)


def merge_overlapped_preds(
    preds_with_meta: List, scene_settings: Settings, nodata_mask: np.ndarray
) -> Path:
    """
    Merges overlapped predictions from segmented parts of a scene into a single prediction array
    and exports it as a GeoTIFF file.

    Parameters:
    preds_with_meta (List): A list of dictionaries containing predictions and metadata for each patch.
    scene_settings (Settings): Configuration settings for the merging and export processes.
    nodata_mask (np.ndarray): An array indicating no-data values.

    Returns:
    Path: Path to the exported GeoTIFF file.

    Raises:
    ValueError: If `preds_with_meta` is empty, which may indicate GPU memory exhaustion.

    Notes:
    - The function creates a gradient mask and merges predictions from each patch.
    - If `scene_settings.export_confidence` is True, it also tracks gradients and normalizes the merged predictions.
    - Updates the progress bar in `scene_settings` throughout the process.
    - Calls `export_geotiff` function to export the final merged prediction array as a GeoTIFF file.
    """

    scene_settings.scene_progress_pbar.desc = "Joining predictions"
    gradient_mask = create_gradient_mask(scene_settings)
    with rio.open(scene_settings.vrt_path) as vrt_src:
        vrt_meta = vrt_src.meta
    output_height = vrt_meta["height"]
    output_width = vrt_meta["width"]
    try:
        class_count = preds_with_meta[0]["patch_pred"].shape[0]
    except IndexError:
        raise ValueError(
            """Error: preds_with_meta list is empty, this normally means you have run
                    out of GPU memory, try lowering the batch size."""
        )

    merged_array = np.zeros([class_count, output_width, output_height], dtype="uint16")
    if scene_settings.export_confidence:
        grad_tracker = np.zeros([output_width, output_height], dtype="float32")
    else:
        grad_tracker = None

    pbar_inc = 32 / len(preds_with_meta)

    for pred_with_meta in preds_with_meta:
        pred_grad = (pred_with_meta["patch_pred"] * gradient_mask).astype("uint8")

        merged_array[
            :,
            pred_with_meta["top"] : pred_with_meta["bottom"],
            pred_with_meta["left"] : pred_with_meta["right"],
        ] += pred_grad

        if grad_tracker is not None:
            grad_tracker[
                pred_with_meta["top"] : pred_with_meta["bottom"],
                pred_with_meta["left"] : pred_with_meta["right"],
            ] += gradient_mask

        scene_settings.scene_progress_pbar.update(pbar_inc)

    if grad_tracker is not None:
        eps = 1e-8
        merged_array = np.where(
            np.logical_and(grad_tracker > 0, merged_array > 0),
            merged_array / (grad_tracker + eps),
            0,
        )

        merged_array = np.clip(merged_array, 0, 255).astype("uint8")

    export_array = np.argmax(merged_array, 0, keepdims=True)

    if scene_settings.export_confidence:
        export_array = np.vstack([export_array, merged_array])

    export_geotiff(export_array, scene_settings, vrt_meta, nodata_mask)

    return scene_settings.cloud_mask_path
