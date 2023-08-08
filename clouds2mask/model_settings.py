from dataclasses import dataclass, fields
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple

from fastai.vision.models.unet import DynamicUnet
from torch import Tensor, device
from tqdm.auto import tqdm


@dataclass
class Settings:
    """
    Represents the Settings for the CloudsS2Mask.

    Attributes:
        scene_name (str): Name of the scene.
        sent_safe_dir (Path): Directory of the .SAFE file.
        mask_output_dir (Path): Directory where mask output will be stored.
        cloud_mask_path (Path): Path to the cloud mask.
        vrt_path (Path): Path to the .vrt file.
        make_qml_files (bool): If True, makes QML files.
        patch_overlap_px (int): Overlap between patches in pixels.
        batch_size (int): Size of the batch to be used in the model.
        tta_max_depth (int): Maximum depth for test time augmentation.
        processing_threads (int): Number of threads for processing.
        temp_dir (TemporaryDirectory): Temporary directory for operations.
        pytorch_device (device): PyTorch device to be used for computations.
        fp16_mode (bool): If True, uses 16-bit precision, else uses 32-bit.
        required_bands (List[str]): List of required bands.
        processing_level (str): The level of processing for the imagery.
        processing_res (int): Resolution for the imagery processing.
        scale_factor (float): Scale factor for image processing.
        patch_size (int): Size of the image patch.
        export_confidence (bool): If True, exports the confidence level.
        mean (Tensor): Mean tensor for normalization.
        std (Tensor): Standard deviation tensor for normalization.
        output_compression (Optional[str]): Type of output compression (if any).
        quiet (bool): If True, suppresses print statements.
        models (List[DynamicUnet]): List of models to be used for inference.
        ordered_augs (List[Tuple[int]]): List of ordered augmentations to apply.
        scene_progress_pbar (tqdm): Progress bar for the scene processing.
    """

    scene_name: str
    sent_safe_dir: Path
    mask_output_dir: Path
    cloud_mask_path: Path
    vrt_path: Path
    make_qml_files: bool
    patch_overlap_px: int
    batch_size: int
    tta_max_depth: int
    processing_threads: int
    temp_dir: TemporaryDirectory
    pytorch_device: device
    fp16_mode: bool
    required_bands: List[str]
    processing_level: str
    processing_res: int
    scale_factor: float
    patch_size: int
    export_confidence: bool
    mean: Tensor
    std: Tensor
    output_compression: Optional[str]
    quiet: bool
    models: List[DynamicUnet]
    ordered_augs: List[Tuple[int]]
    b10_size: float
    scene_progress_pbar: tqdm = tqdm(disable=True)

    def __repr__(self):
        table = "{:<20} {:<15}\n".format("Attribute", "Value")
        table += "=" * 35
        table += "\n"

        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, TemporaryDirectory):
                value = value.name  # Get the path of the temporary directory

            elif isinstance(value, list) and all(
                isinstance(i, DynamicUnet) for i in value
            ):
                value = "\n".join([str(i.layers[0][0]) for i in value])
            elif isinstance(value, Tensor):
                # Flatten the tensor, convert it to a list and join it into a
                # single line string
                value = ", ".join(map(str, value.flatten().tolist()))

            table += "{:<20} {:<15}\n".format(field.name, str(value))

        return table


@dataclass
class Inf_Only_Settings:
    """
    Represents the Settings specific to inference only.

    Attributes:
        processing_res (int): Resolution for the imagery processing.
        processing_level (str): The level of processing for the imagery.
        mean (Tensor): Mean tensor for normalization.
        std (Tensor): Standard deviation tensor for normalization.
        scene_progress_pbar (tqdm): Progress bar for the scene processing.
        pytorch_device (device): PyTorch device to be used for computations.
        batch_size (int): Size of the batch to be used in the model.
        tta_max_depth (int): Maximum depth for test time augmentation.
        fp16_mode (bool): If True, uses 16-bit precision, else uses 32-bit.
        models (List[DynamicUnet]): List of models to be used for inference.
        ordered_augs (List[Tuple[int]]): List of ordered augmentations to apply.
    """

    processing_res: int
    processing_level: str
    mean: Tensor
    std: Tensor
    scene_progress_pbar: tqdm
    pytorch_device: device
    batch_size: int
    tta_max_depth: int
    fp16_mode: bool
    models: List[DynamicUnet]
    ordered_augs: List[Tuple[int]]

    def __repr__(self):
        table = "{:<20} {:<15}\n".format("Attribute", "Value")
        table += "=" * 35
        table += "\n"

        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, TemporaryDirectory):
                value = value.name  # Get the path of the temporary directory

            elif isinstance(value, list) and all(
                isinstance(i, DynamicUnet) for i in value
            ):
                value = "\n".join([str(i.layers[0][0]) for i in value])
            elif isinstance(value, Tensor):
                # Flatten the tensor, convert it to a list and join it into a
                # single line string
                value = ", ".join(map(str, value.flatten().tolist()))

            table += "{:<20} {:<15}\n".format(field.name, str(value))

        return table
