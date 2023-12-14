import queue
import threading
from threading import Thread
from typing import List
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from .model_settings import Settings
from .model_helpers import warmup_models
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
    mask: np.ndarray,
) -> None:
    """
    Processes predictions to handle overlapping areas, performs cleanup, and
    generates QML files based on settings.

    Args:
        predictions_with_metadata (List[dict]): Predictions and their metadata.
        scene_settings (Settings): Configuration for the scene, including paths
        and flags for QML file creation. batch_progress_bar (tqdm): Progress bar
        for tracking batch processing. mask (np.ndarray): Mask for identifying
        areas with no data.
    """

    merge_overlapped_preds(predictions_with_metadata, scene_settings, mask)

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
    mask: np.ndarray,
) -> Thread:
    """
    Performs inference on image patches, saves the results, and initiates a
    separate thread for the save operation.

    Args:
        patch_arrays (List[np.ndarray]): List of numpy arrays representing image
        patches for inference. patch_metadata (List[dict]): Metadata associated
        with each patch. scene_settings_local (Settings): Configuration settings
        specific to the current scene. batch_progress_bar (tqdm): Progress bar
        for monitoring batch processing. mask (np.ndarray): Mask used to
        identify areas with no data in the image processing.

    Returns:
        Thread: A thread object that handles the save operation.
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
            mask,
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
    Continuously processes tasks from the inference queue. It performs inference
    on each task and then places the results in the export queue.

    Args:
        inference_queue (queue.Queue): A queue containing tasks (data and
        settings) for inference. export_queue (queue.Queue): A queue where the
        results of the inference are stored for further processing.
    """

    while True:
        task = inference_queue.get()

        if task is None:
            break
        result = infer_and_save(*task)
        export_queue.put(result)
        inference_queue.task_done()


def save_worker(export_queue):
    """
    Continuously processes results from the export queue by joining each save
    thread. This ensures that each data save operation is completed before
    proceeding to the next.

    Args:
        export_queue (queue.Queue): Queue containing save thread objects from
        the inference worker.
    """
    while True:
        save_thread = export_queue.get()

        if save_thread is None:
            break

        save_thread.join()
        export_queue.task_done()


def batch_process_scenes(scene_settings_batch):
    """
    Processes a batch of scenes for cloud mask generation in a multi-threaded
    manner. It involves warming up models, generating patches, performing
    inference, and saving results.

    Args:
        scene_settings_batch (List[Settings]): A list of settings objects, each
        configuring the processing for a scene.

    Returns:
        List[Path]: A list of file paths, each corresponding to the cloud mask
        file generated for a scene in the batch.
    """

    warmup_thread = Thread(
        target=warmup_models,
        args=(scene_settings_batch[0],),
    )
    warmup_thread.start()

    if scene_settings_batch[0].processing_res == 10:
        inference_queue = queue.Queue(maxsize=1)
    else:
        inference_queue = queue.Queue(maxsize=2)
    export_queue = queue.Queue(maxsize=2)

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

    save_thread = threading.Thread(target=save_worker, args=([export_queue]))
    save_thread.start()

    for i, scene_settings in enumerate(scene_settings_batch):
        # hold up the queue if it's full to avoid excessive memory usage
        while inference_queue.full():
            time.sleep(0.1)
        scene_settings.scene_progress_pbar = create_progress_bar(scene_settings)
        scene_settings.mask_output_dir.mkdir(exist_ok=True, parents=True)
        patch_arrays, patch_metadata, mask = generate_patches(scene_settings)

        task = (
            patch_arrays,
            patch_metadata,
            scene_settings,
            batch_progress_bar,
            mask,
        )

        if warmup_thread.is_alive():
            warmup_thread.join()

        inference_queue.put(task)

        cloud_mask_paths.append(scene_settings.cloud_mask_path)
    # wait for inference to finish
    inference_queue.put(None)
    inference_thread.join()

    # if cuda is used, empty the cache
    if scene_settings_batch[0].pytorch_device.type == "cuda":
        torch.cuda.empty_cache()

    # wait for saving to finish
    export_queue.put(None)
    save_thread.join()

    return cloud_mask_paths
