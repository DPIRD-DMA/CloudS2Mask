from typing import List, Tuple

import torch
import torch.nn.functional as F


def update_active_indices(
    preds_mean: torch.Tensor, active_indices: List[int], threshold: int = 10
) -> List[int]:
    """
    This function updates the list of active indices based on the threshold
    value provided. It considers an image active if the count of its isolated
    pixels, calculated from the mean of predictions, exceeds a dynamically
    calculated area modified threshold. This threshold is computed as a
    proportion of the image area based on the given pixel count threshold.

    Args:
        preds_mean (torch.Tensor): A tensor representing the current mean of the
        predictions
            for all images.
        active_indices (List[int]): A list of indices representing the images
        currently considered
            as active or requiring further processing.
        threshold (int, optional): The minimum count of isolated pixels an image
        should have to be
            considered active. This is used to calculate the area modified
            threshold for each image. Defaults to 10.

    Returns:
        List[int]: The updated list of active indices after applying the area
        modified threshold criteria.
    """
    iso_px = count_isolated_pixels(preds_mean, active_indices, 0)
    standard_pct_threshold = threshold / (512 * 512)
    area_modified_threshold = (
        preds_mean.shape[-2] * preds_mean.shape[-1]
    ) * standard_pct_threshold

    new_active_indices = [
        index
        for index, iso_count in enumerate(iso_px)
        if iso_count > area_modified_threshold
    ]

    return new_active_indices


def weighted_mean(
    mean_tensor: torch.Tensor,
    new_tensor: torch.Tensor,
    mean_count: int,
    active_indices: List[int],
) -> torch.Tensor:
    """
    Calculate the weighted mean of the input tensors.

    Args:
        mean_tensor (torch.Tensor): The mean tensor. new_tensor (torch.Tensor):
        The new tensor to be included in the mean calculation. mean_count (int):
        The current count of values used in calculating the mean. active_indices
        (List[int]): Indices of the images to process.

    Returns:
        torch.Tensor: The updated mean tensor.
    """
    if mean_count > 0:
        mean_tensor[active_indices] = (
            (mean_tensor[active_indices] * mean_count) + new_tensor
        ) / (mean_count + 1)
        return mean_tensor
    else:
        return new_tensor


@torch.jit.script  # type: ignore
def count_isolated_pixels(
    predictions: torch.Tensor, indices_to_keep: List[int], neighbour_threshold: int = 1
) -> torch.Tensor:
    """
    Count the number of isolated pixels in a batch of predictions.

    Args:
        predictions (torch.Tensor): The input predictions tensor.
        indices_to_keep (List[int]): Indices of the images to process.
        neighbour_threshold (int, optional): The maximum number of neighbours with the same class label to consider a pixel isolated. Defaults to 1 neighbour.

    Returns:
        List[int]: A list containing the count of isolated pixels for each image
        in the input predictions.
    """
    preds_tensor = predictions[indices_to_keep]
    # avoid a bug in pytorch 2.0 that causes random outputs when using argmax on mps
    # https://github.com/pytorch/pytorch/issues/92311
    if preds_tensor.device.type == "mps":
        _, argmax_tensor = torch.max(preds_tensor.float(), dim=1)
    else:
        argmax_tensor = torch.argmax(preds_tensor.float(), dim=1)

    # Pad the argmax_tensor
    argmax_tensor_padded = F.pad(
        argmax_tensor, (1, 1, 1, 1), mode="constant", value=0.0
    )

    # Calculate the number of neighbors with the same class label
    neighbor_count = (
        (argmax_tensor_padded[:, :-2, :-2] == argmax_tensor).int()
        + (argmax_tensor_padded[:, 1:-1, :-2] == argmax_tensor).int()
        + (argmax_tensor_padded[:, 2:, :-2] == argmax_tensor).int()
        + (argmax_tensor_padded[:, :-2, 2:] == argmax_tensor).int()
        + (argmax_tensor_padded[:, 1:-1, 2:] == argmax_tensor).int()
        + (argmax_tensor_padded[:, 2:, 2:] == argmax_tensor).int()
        + (argmax_tensor_padded[:, :-2, 1:-1] == argmax_tensor).int()
        + (argmax_tensor_padded[:, 2:, 1:-1] == argmax_tensor).int()
    )

    # Identify pixels with a number of neighbors less than or equal to the threshold
    tracker = neighbor_count <= neighbour_threshold

    # Sum the tracker tensor to count the number of isolated pixels
    remaining_isolated_counts = tracker.sum(dim=[1, 2], dtype=torch.int32)

    # Initialize a tensor to store the isolated pixel counts for all images
    full_isolated_counts = torch.zeros(
        predictions.shape[0], device=remaining_isolated_counts.device, dtype=torch.int32
    )

    # Update the counts for the images specified in indices_to_keep
    full_isolated_counts[indices_to_keep] = remaining_isolated_counts
    return full_isolated_counts


def get_tta_options(tta_max_depth: int) -> List[Tuple[int]]:
    """
    Generates a list of test-time augmentation (TTA) options based on the
    provided settings.

    Args:
        scene_settings (Settings): A settings object containing the maximum number of TTA options (tta_max_depth) to include in the output list.
    """

    ordered_augs = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)]

    return ordered_augs[: tta_max_depth + 1]
