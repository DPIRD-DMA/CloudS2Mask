from pathlib import Path
from typing import List

import torch
from fastai.vision.models.unet import DynamicUnet
from torch import device as Device

from .model_settings import Settings


def load_models(
    model_paths: List[Path], pytorch_device: torch.device
) -> List[DynamicUnet]:
    """
    Load a list of models from file paths.

    Args:
        model_paths (List[Path]): A list of model file paths.
        pytorch_device (torch.device): The device to which the model is loaded.

    Returns:
        List[DynamicUnet]: A list of models.
    """

    models = []
    for model_path in model_paths:
        model = torch.load(model_path, map_location=pytorch_device)
        model.eval()
        models.append(model)
    return models


def warmup_models(settings: Settings) -> None:
    """
    Warm up the models by passing a tensor through them.

    Args:
        settings (Settings): The settings object.
    """
    dummy_input = torch.zeros(
        (1, 13, settings.patch_size, settings.patch_size),
        device=settings.pytorch_device,
        dtype=torch.float16 if settings.fp16_mode else torch.float32,
    )

    for model in settings.models:
        model(dummy_input)


def fp16_available(
    pytorch_device: torch.device,
    models: List[DynamicUnet],
    patch_size: int = 500,
) -> bool:
    """
    Check if the given device supports FP16 (half-precision) computations.

    Args:
        device (torch.device): The device to check.
        models (List[DynamicUnet]): A list of models.
        patch_size (int, optional): The patch size to use for the test. Defaults to 500.

    Returns:
        bool: True if the device supports FP16 computations, False otherwise.
    """

    try:
        # try passing a half precision tensor through the model
        model = models[0].half()
        model(
            torch.zeros((1, 13, patch_size, patch_size), device=pytorch_device).half()
        )
        return True
    except RuntimeError as e:
        print(f"FP16 NOT supported on {pytorch_device}, using FP32.")
        return False


def find_models(
    model_dir: Path, processing_res: int, model_ensembling: bool = False
) -> List[Path]:
    """
    Finds models in a given directory based on processing resolution.

    This function searches for model files with .pkl extension in the provided
    directory, and then filters these models based on the processing resolution
    provided.

    Args:
        model_dir (Path): The directory where to search for model files.
        processing_res (int): The processing resolution to use for filtering the model files.
        model_ensembling (bool, optional): If True, returns all the models that match the processing resolution. If False, returns the first model that matches the processing resolution. Defaults to False.

    Returns:
        List[Path]: A list containing the paths of the model files found.

    Raises:
        Exception: If no model can be found based on the processing resolution.
    """
    # find all models in the model directory
    all_models = list(model_dir.glob("*.pkl"))
    # sort the models by name so that the order is consistent
    all_models.sort()
    model_paths = []
    # filter the models based on the processing resolution
    for model_file_path in all_models:
        if model_file_path.name.split("_")[-1] == f"{processing_res}m.pkl":
            model_paths.append(model_file_path)
            # if we are not ensembling, we only need one model
            if not model_ensembling:
                return model_paths
    # if we are ensembling, we need at least two models
    return model_paths


def default_device() -> Device:
    """
    Determines the best available device for computation.

    This function checks if CUDA or MPS (Metal Performance Shaders) are
    available on the current machine, in that order. If neither are available,
    it defaults to using the CPU.

    Returns:
        torch.device: The device to be used for computation. This can be a CUDA
        device, MPS device or CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # type: ignore
        return torch.device("mps")
    return torch.device("cpu")
