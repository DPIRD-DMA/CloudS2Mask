import queue
import threading
from concurrent import futures
from threading import Thread
from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm

from .model_settings import Settings
from .inference import run_inference
from .make_qml_files import (
    create_cloud_mask_classification_qml,
    create_cloud_mask_confidence_qml,
)
from .patch_generator import generate_patches
from .pred_joiner import merge_overlapped_preds


def export_overlapping_predictions(
    predictions_with_metadata: List[dict],
    scene_settings: Settings,
    batch_progress_bar: tqdm,
    nodata_mask_future: futures.Future,
) -> None:
    """
    Exports overlapping predictions, performs cleanup operations, and creates
    QML files if required.

    Args:
        predictions_with_metadata (List[dict]): List of predictions with
        corresponding metadata. scene_settings (Settings): Configuration
        settings for the current scene. batch_progress_bar (tqdm): Progress bar
        for batch processing. nodata_mask (np.ndarray): Nodata mask used in the
        processing.
    """
    # get the nodata mask which should be done by now
    nodata_mask = nodata_mask_future.result()

    merge_overlapped_preds(predictions_with_metadata, scene_settings, nodata_mask)
    scene_settings.temp_dir.cleanup()
    if scene_settings.make_qml_files:
        if scene_settings.export_confidence:
            create_cloud_mask_confidence_qml(scene_settings.cloud_mask_path)

        else:
            create_cloud_mask_classification_qml(scene_settings.cloud_mask_path)

    scene_progress_bar = scene_settings.scene_progress_pbar
    if scene_progress_bar.total is not None:
        scene_progress_bar.update(scene_progress_bar.total - scene_progress_bar.n)

    scene_progress_bar.close()
    batch_progress_bar.update()


def infer_and_save(
    patch_arrays: List[np.ndarray],
    patch_metadata: List[dict],
    scene_settings_local: Settings,
    batch_progress_bar: tqdm,
    nodata_mask_future: futures.Future,
) -> Thread:
    """
    Runs inference on patches, saves the results, and returns a thread of the
    save operation.

    Args:
        patch_batches (List[torch.Tensor]): List of input patches for inference.
        meta_batches (List[List[dict]]): Metadata associated with each patch.
        scene_settings_local (Settings): Scene-specific settings.
        batch_progress_bar (tqdm): Progress bar for batch processing.
        nodata_mask (np.ndarray): Nodata mask used in the processing.

    Returns:
        Thread: A thread object representing the save operation.
    """
    predictions_with_metadata = run_inference(
        patch_arrays, patch_metadata, scene_settings_local
    )
    save_thread = Thread(
        target=export_overlapping_predictions,
        args=[
            predictions_with_metadata,
            scene_settings_local,
            batch_progress_bar,
            nodata_mask_future,
        ],
    )
    save_thread.start()
    return save_thread


def create_progress_bar(scene_settings: Settings) -> tqdm:
    """
    Creates a progress bar based on the provided scene settings.

    Args:
        scene_settings (Settings): Settings for the current scene.

    Returns:
        tqdm: A progress bar object.
    """
    return (
        tqdm(total=100, leave=False, bar_format="{l_bar}{bar}{elapsed}")
        if not scene_settings.quiet
        else tqdm(display=False)
    )


def inference_worker(inference_queue, export_queue):
    """
    Processes tasks in the inference queue, runs inference, and pushes results
    to the export queue.

    Args:
        inference_queue (queue.Queue): Queue containing tasks for inference.
        export_queue (queue.Queue): Queue for storing inference results.
    """

    while True:
        task = inference_queue.get()
        if task is None:
            break
        result = infer_and_save(*task)
        export_queue.put(result)


def save_worker(export_queue, save_threads):
    """
    Processes results in the export queue, joins and appends them to a list of
    save threads.

    Args:
        export_queue (queue.Queue): Queue containing results from the inference
        worker. save_threads (List[threading.Thread]): List where completed save
        threads are appended.
    """
    while True:
        result = export_queue.get()
        if result is None:
            break
        save_thread = result
        save_threads.append(save_thread)
        save_thread.join()


def batch_process_scenes(scene_settings_batch):
    """
    Processes a batch of scenes in a multi-threaded manner, performing inference
    and saving results.

    Args:
        scene_settings_batch (List[Settings]): A list of settings for each scene
        in the batch.

    Returns:
        List[Path]: A list of Paths to the cloud mask files for each scene in
        the batch.
    """
    inference_queue = queue.Queue(maxsize=2)
    export_queue = queue.Queue(maxsize=3)

    save_threads = []
    cloud_mask_paths = []

    batch_progress_bar = (
        tqdm(
            total=len(scene_settings_batch),
            desc="Batch progress",
            unit="scenes",
            leave=False,
        )
        if not scene_settings_batch[0].quiet
        else tqdm(display=False)
    )

    inference_thread = threading.Thread(
        target=inference_worker, args=(inference_queue, export_queue)
    )
    inference_thread.start()

    save_thread = threading.Thread(
        target=save_worker, args=(export_queue, save_threads)
    )
    save_thread.start()

    for scene_settings in scene_settings_batch:
        scene_settings.scene_progress_pbar = create_progress_bar(scene_settings)
        scene_settings.mask_output_dir.mkdir(exist_ok=True, parents=True)
        patch_arrays, patch_metadata, nodata_mask_future = generate_patches(
            scene_settings
        )

        task = (
            patch_arrays,
            patch_metadata,
            scene_settings,
            batch_progress_bar,
            nodata_mask_future,
        )
        inference_queue.put(task)

        cloud_mask_paths.append(scene_settings.cloud_mask_path)

    inference_queue.put(None)
    export_queue.put(None)

    # Join the threads to ensure all tasks have completed before continuing
    inference_thread.join()
    save_thread.join()
    # if cuda is used, empty the cache
    if scene_settings_batch[0].pytorch_device.type == "cuda":
        torch.cuda.empty_cache()

    return cloud_mask_paths
