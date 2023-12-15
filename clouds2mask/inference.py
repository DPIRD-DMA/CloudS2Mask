import concurrent.futures
import queue
from typing import List, Tuple, Union

import numpy as np
import torch

from .model_settings import Inf_Only_Settings, Settings
from .tta_helpers import update_active_indices, weighted_mean


def apply_augmentation(
    augmentations: Tuple,
    tensor_batch: torch.Tensor,
) -> torch.Tensor:
    """
    Applies specified image augmentations to the base patch tensors.

    Args:
        augmentations (Tuple): A tuple indicating the types of augmentations to apply.
        base_patch_tensors (torch.Tensor): The base image tensors to which augmentations are applied.

    Returns:
        torch.Tensor: The augmented image tensors.
    """
    rotation, h_flip = augmentations

    if h_flip:
        tensor_batch = torch.flip(tensor_batch, dims=[3])

    if rotation:
        tensor_batch = torch.rot90(tensor_batch, k=rotation, dims=(2, 3))

    return tensor_batch


def undo_augmentation(
    augmentations: Tuple,
    predictions: torch.Tensor,
) -> torch.Tensor:
    """
    Reverses the applied image augmentations.

    Args:
        augmentations (Tuple): A tuple indicating the types of augmentations to reverse.
        predictions (torch.Tensor): The augmented image tensors.

    Returns:
        torch.Tensor: The image tensors after reversing the augmentations.
    """
    rotation, h_flip = augmentations

    if rotation:
        predictions = torch.rot90(predictions, k=-rotation, dims=(2, 3))

    if h_flip:
        predictions = torch.flip(predictions, dims=[3])

    return predictions


@torch.jit.script  # type: ignore
def normalize_batch(
    x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return ((x / 32767) - mean) / std


def create_and_queue_batches(
    arrays_batches: List[List[np.ndarray]],
    scene_settings: Union[Settings, Inf_Only_Settings],
    batch_queue: queue.Queue,
) -> None:
    """
    The create_and_queue_batches function generates tensor batches and puts them
    in a queue.

    Args:
        arrays_batches (List[List[np.ndarray]]): List of numpy array batches.
        scene_settings (Settings): Scene settings.
        batch_queue (queue.Queue): A queue to store the tensor batches.
    """
    for patch_array_batch in arrays_batches:
        patch_tensor_batch = patch_list_to_tensor(patch_array_batch, scene_settings)

        batch_queue.put(patch_tensor_batch)


def consumer_batches_with_inf(
    meta_batches: List,
    queue: queue.Queue,
    preds_with_meta: List,
    scene_settings: Union[Settings, Inf_Only_Settings],
    pbar_inc: float,
) -> None:
    """
    The consumer_batches_with_inf function gets tensor batches from a queue and
    processes them.

    Args:
        meta_batches (List): List of metadata batches.
        queue (queue.Queue): A queue to get the tensor batches from.
        preds_with_meta (List): A list to store the predictions and metadata. scene_settings (Union[Settings,
        Inf_Only_Settings]): Scene settings.
        pbar_inc (float): Progress bar increment value.
    """
    for patch_meta in meta_batches:
        # this will block until an item is available
        patch_tensor_batch = queue.get()

        preds_np = get_preds(patch_tensor_batch, scene_settings)

        for patch_pred, meta in zip(preds_np, patch_meta):
            meta["patch_pred"] = patch_pred
            preds_with_meta.append(meta)

        scene_settings.scene_progress_pbar.update(pbar_inc)


def make_batches(
    patch_arrays: List[np.ndarray],
    patch_metadata: List[dict],
    scene_settings: Union[Settings, Inf_Only_Settings],
) -> Tuple[List[List[np.ndarray]], List[List[dict]]]:
    """
    Breaks the provided patch arrays and metadata into sublist batches.

    Args:
        patch_arrays (List[np.ndarray]): List of patch arrays.
        patch_metadata (List[dict]): List of metadata.
        scene_settings (Union[Settings, Inf_Only_Settings]): Scene settings.

    Returns:
        Tuple[List[List[np.ndarray]], List[List[dict]]]: Sublist batches of
        patch arrays and metadata.
    """
    # break the patch arrays into sublist batches
    patch_arrays_sublists = [
        patch_arrays[i : i + scene_settings.batch_size]
        for i in range(0, len(patch_arrays), scene_settings.batch_size)
    ]
    # break the patch meta into sublist batches
    patch_meta_sublists = [
        patch_metadata[i : i + scene_settings.batch_size]
        for i in range(0, len(patch_metadata), scene_settings.batch_size)
    ]

    return patch_arrays_sublists, patch_meta_sublists


def patch_list_to_tensor(
    patch_arrays: List[np.ndarray], scene_settings: Union[Settings, Inf_Only_Settings]
) -> torch.Tensor:
    """
    Converts a list of patch arrays to a tensor batch.

    Args:
        patch_arrays (List[np.ndarray]): List of patch arrays.
        scene_settings (Union[Settings, Inf_Only_Settings]): The settings for the scene.

    Returns:
        torch.Tensor: The tensor of patch arrays.
    """
    # convert the patch arrays to tensors, and move them to the correct device
    tensor_type = torch.float16 if scene_settings.fp16_mode else torch.float32

    patch_batch = torch.from_numpy(np.stack(patch_arrays).astype(np.float32)).to(
        device=scene_settings.pytorch_device, dtype=tensor_type
    )

    return patch_batch


def get_preds(
    patch_tensor_batch: torch.Tensor,
    scene_settings: Union[Settings, Inf_Only_Settings],
) -> np.ndarray:
    """
    Generates predictions for the provided image tensors.

    Args:
        patch_tensor_batch (torch.Tensor): The image tensors for which to generate predictions.
        scene_settings (Union[Settings, Inf_Only_Settings]): The settings for the scene.

    Returns:
        np.ndarray: The generated predictions for the image tensors.
    """
    active_indices = list(range((patch_tensor_batch.shape[0])))
    preds_mean = torch.zeros(
        [
            scene_settings.batch_size,
            4,
            patch_tensor_batch.shape[-2],
            patch_tensor_batch.shape[-1],
        ]
    )

    patch_tensors_norm = normalize_batch(
        patch_tensor_batch, scene_settings.mean, scene_settings.std
    )

    # keep track of the number of predictions made
    pred_count = 0
    for augs in scene_settings.ordered_augs:
        # if there are no active indices, break
        if len(active_indices) == 0:
            break

        patch_tensors_norm_aug = apply_augmentation(augs, patch_tensors_norm)

        with torch.no_grad():
            # Get predictions for each model
            for model in scene_settings.models:
                preds = model(patch_tensors_norm_aug[active_indices])
                # convert to probabilities
                preds = torch.nn.functional.softmax(preds, dim=1)

                preds = undo_augmentation(augs, preds)

                # update the mean predictions for the active indices
                preds_mean = weighted_mean(
                    preds_mean, preds, pred_count, active_indices
                )

                pred_count += 1
                # if there are no active indices, break
                if len(active_indices) == 0:
                    break

        active_indices = update_active_indices(preds_mean, active_indices)
    preds_mean = torch.mul(preds_mean, 255)

    preds_np = preds_mean.cpu().numpy().astype("uint8")

    return preds_np


def run_inference(
    patch_arrays: List[np.ndarray],
    patch_metadata: List[dict],
    scene_settings: Union[Settings, Inf_Only_Settings],
) -> list:
    """
    Performs inference on the provided image patches and returns the predictions
    with corresponding metadata.

    Args:
        patch_arrays (List[np.ndarray]): List of image patch arrays to perform inference on.
        patch_metadata (List[dict]): Metadata for each image patch.
        scene_settings (Union[Settings, Inf_Only_Settings]): The settings for the scene.

    Returns:
        list: List of dictionaries, each containing the prediction and
        corresponding metadata for an image patch.
    """

    # convert arrays and metadata to batches
    arrays_batches, meta_batches = make_batches(
        patch_arrays, patch_metadata, scene_settings
    )

    pbar_inc = 33 / len(arrays_batches)

    scene_settings.scene_progress_pbar.desc = (
        f"Running model ({scene_settings.pytorch_device.type})"
    )

    # Create a queue for batches
    batch_queue = queue.Queue(maxsize=2)

    preds_with_meta = []
    # run the producer and consumer in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(
            create_and_queue_batches, arrays_batches, scene_settings, batch_queue
        )

        executor.submit(
            consumer_batches_with_inf,
            meta_batches,
            batch_queue,
            preds_with_meta,
            scene_settings,
            pbar_inc,
        )
    scene_settings.scene_progress_pbar.update(pbar_inc)

    return preds_with_meta
