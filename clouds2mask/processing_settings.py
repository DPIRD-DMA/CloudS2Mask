from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple, Union

import torch
from torch import Tensor, device
from tqdm.auto import tqdm
import warnings

from .download_model_weights import download_model_weights
from .model_helpers import (
    default_device,
    find_models,
    fp16_available,
    load_models,
)
from .model_settings import Inf_Only_Settings, Settings
from .tta_helpers import get_tta_options


def get_normalization_stats(
    pytorch_device: device,
    fp16_mode: bool,
    processing_level: str,
    required_bands: List[str],
) -> Tuple[Tensor, Tensor]:
    """
    Return normalization statistics (mean and standard deviation) for the
    required bands based on the processing level.

    Args:
        pytorch_device (torch.device): The device to which the tensor will be
        moved. fp16_mode (bool): If true, the returned tensors will be in fp16
        format. processing_level (str): The processing level, currently supports
        "L1C" only. required_bands (List[str]): List of band names for which
        normalization statistics are required.

    Returns:
        tuple: A tuple containing the mean and standard deviation tensors, each
        of shape (1, num_bands, 1, 1) where num_bands is the length of
        required_bands. These tensors are moved to the device specified by
        pytorch_device, and their datatype is fp16 if fp16_mode is True,
        otherwise fp32.

    Raises:
        Exception: If the processing_level is not "L1C".
    """

    if processing_level == "L1C":
        L1C_mean = {
            "B01": 0.072623697227855,
            "B02": 0.06608867585127501,
            "B03": 0.061940767467830685,
            "B04": 0.06330473795822207,
            "B05": 0.06858655023065205,
            "B06": 0.08539433443008514,
            "B07": 0.09401670610922229,
            "B08": 0.09006412206990828,
            "B8A": 0.09915093732164396,
            "B09": 0.035429756513690985,
            "B10": 0.003632839439909688,
            "B11": 0.06855744750648961,
            "B12": 0.0486043830034996,
        }

        L1C_std = {
            "B01": 0.020152047138155018,
            "B02": 0.022698212883948143,
            "B03": 0.023073879486441455,
            "B04": 0.02668270641026416,
            "B05": 0.0263763340626224,
            "B06": 0.027439342904551974,
            "B07": 0.02896087163616576,
            "B08": 0.028661147214616267,
            "B8A": 0.0301365958005653,
            "B09": 0.013482676031864258,
            "B10": 0.0019204000834290252,
            "B11": 0.023938917594669776,
            "B12": 0.020069414811037536,
        }

        mean_subset = torch.tensor([L1C_mean[i] for i in required_bands])
        std_subset = torch.tensor([L1C_std[i] for i in required_bands])

    else:
        raise Exception("Processing level must be L1C")
    mean = mean_subset.view(1, len(required_bands), 1, 1).to(pytorch_device)
    std = std_subset.view(1, len(required_bands), 1, 1).to(pytorch_device)

    if fp16_mode:
        mean = mean.half()
        std = std.half()
    return mean, std


def get_b10_size(sent_safe_dir: Path) -> float:
    """
    Return the size of the B10 band file in bytes.

    Args:
        sent_safe_dir (Path): The directory of the .SAFE file.


    Returns:
        float: The size of the B10 band file in bytes.

    """
    # find the B10 file
    B10_file = next((sent_safe_dir / "GRANULE").rglob("*_B10.jp2"))
    return B10_file.stat().st_size


def create_settings(
    output_dir: Path,
    sent_safe_dirs: Union[List[Path], List[str]],
    output_compression: Union[str, None] = "LZW",
    batch_size: int = 10,
    pytorch_device: Union[torch.device, str] = default_device(),
    processing_threads: int = 2,
    required_bands: List[str] = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    ],
    keep_vrt: bool = True,
    make_qml_files: bool = True,
    tta_max_depth: int = 0,
    processing_res: int = 10,
    export_confidence: bool = False,
    quiet: bool = False,
    model_ensembling: bool = False,
    patch_size: int = 500,
    fp16_mode: Union[bool, None] = None,
    patch_overlap_px: int = 128,
    optimise_scene_order: bool = True,
) -> List[Settings]:
    """
    Creates a list of settings objects for CloudS2Mask for each directory in
    sent_safe_dirs.

    Parameters
    ----------
    output_dir : Path
        The directory where outputs will be saved.
    sent_safe_dirs : List[Path], List[str]
        List of directories of the .SAFE files.
    output_compression : str, optional
        The type of output compression (if any).
    model_names : List[str], optional
        List of names of models to use.
    model_dir : Path, optional
        Directory where the models are stored.
    batch_size : int, optional
        The size of the batch to be used in model.
    pytorch_device : torch.device or str, optional
        PyTorch device to be used for computations.
    processing_threads : int, optional
        Number of threads for processing.
    required_bands : List[str], optional
        List of required bands.
    keep_vrt : bool, optional
        If True, keeps the .vrt file.
    tta_max_depth : int, optional
        Maximum depth for test time augmentation.
    processing_res : int, optional
        The resolution for the imagery processing.
    export_confidence : bool, optional
        If True, exports the confidence level.
    quiet : bool, optional
        If True, suppresses progress bar and other non-essential output.
    model_ensembling : bool, optional
        To enable model ensembling or not
    patch_size : int, optional
        The size of the patch.
    fp16_mode : bool, optional
        If True, use 16-bit precision for the model, else use 32-bit.
    patch_overlap_px : int, optional
        The overlap size in pixels between patches.
    optimise_scene_order : bool, optional
        If True, optimises the scene order for processing.

    Returns
    -------
    List[Settings]
        List of settings objects for each directory in sent_safe_dirs.
    """
    if processing_res not in [10, 20]:
        raise Exception("processing_res must be 10 or 20")

    if len(sent_safe_dirs) == 0:
        raise Exception("No .SAFE directories provided")

    if tta_max_depth < 0 or tta_max_depth > 7:
        raise Exception("tta_max_depth must be between 0 and 7")

    if patch_overlap_px >= patch_size:
        raise Exception("patch_overlap_px must be less than patch_size")

    if patch_overlap_px < 0:
        raise Exception("patch_overlap_px must be greater than or equal to 0")

    if patch_size < 256:
        warnings.warn(
            f"{patch_size} is a very small patch size, this may cause accuracy issues"
        )

    download_model_weights()
    # if device has been set to a string make it a pytorch device
    pytorch_device = torch.device(pytorch_device)

    model_dir = Path(__file__).resolve().parent / "models"

    model_paths = find_models(
        model_dir=model_dir,
        processing_res=processing_res,
        model_ensembling=model_ensembling,
    )
    # load the models
    models = load_models(model_paths=model_paths, pytorch_device=pytorch_device)

    # if fp16_mode is not set then check if we can use it
    if fp16_mode is None:
        fp16_mode = fp16_available(
            pytorch_device=pytorch_device,
            models=models,
            patch_size=patch_size,
        )
    # if fp16_mode is True the convert the models to half precision
    for index, model in enumerate(models):
        if fp16_mode:
            models[index] = model.half()
        else:
            models[index] = model.float()

    # build output folder path
    output_folder_name = f"CloudS2Mask {processing_res}m"
    if model_ensembling:
        output_folder_name += " model ensembling"
    # if we are using adaptive test time augmentation add it to the output folder name
    if tta_max_depth > 0:
        output_folder_name += f" ATTA {tta_max_depth}"

    if processing_res == 10:
        scale_factor = 1.0
    elif processing_res == 20:
        scale_factor = 0.5
    else:
        raise Exception("res must be int 10 or 20")

    # create a TTA augmentation list
    ordered_augs = get_tta_options(tta_max_depth)

    scene_settings_list = []
    for sent_safe_dir in sent_safe_dirs:
        # convert to path if string
        sent_safe_dir = Path(sent_safe_dir)

        # the name of the .SAFE folder
        scene_name = sent_safe_dir.name

        # the processing level of the imagery
        processing_level = scene_name.split("_")[1][-3:]

        mask_output_dir = output_dir / output_folder_name / scene_name

        # if we are keeping the vrt place it in the mask output folder
        temp_dir = TemporaryDirectory()
        working_dir = Path(temp_dir.name)

        if keep_vrt:
            vrt_path = mask_output_dir / (scene_name + ".vrt")
        else:
            vrt_path = working_dir / (scene_name + ".vrt")

        mean, std = get_normalization_stats(
            pytorch_device, fp16_mode, processing_level, required_bands
        )
        if optimise_scene_order:
            b10_size = get_b10_size(sent_safe_dir)
        else:
            b10_size = 0.0

        scene_settings_list.append(
            Settings(
                scene_name=scene_name,
                sent_safe_dir=sent_safe_dir,
                mask_output_dir=mask_output_dir,
                cloud_mask_path=mask_output_dir
                / (str(scene_name) + "_CloudS2Mask.tif"),
                vrt_path=vrt_path,
                make_qml_files=make_qml_files,
                patch_overlap_px=patch_overlap_px,
                batch_size=batch_size,
                tta_max_depth=tta_max_depth,
                processing_threads=processing_threads,
                temp_dir=temp_dir,
                pytorch_device=pytorch_device,
                fp16_mode=fp16_mode,
                required_bands=required_bands,
                processing_level=processing_level,
                processing_res=processing_res,
                scale_factor=scale_factor,
                patch_size=patch_size,
                export_confidence=export_confidence,
                output_compression=output_compression,
                mean=mean,
                std=std,
                quiet=quiet,
                models=models,
                ordered_augs=ordered_augs,
                b10_size=b10_size,
            )
        )
    # sort the scenes by B10 file size, to optimise processing order
    if optimise_scene_order:
        scene_settings_list.sort(key=lambda x: x.b10_size, reverse=True)
    return scene_settings_list


def create_inf_only_settings(
    processing_res: int,
    pytorch_device: Union[torch.device, str] = default_device(),
    processing_level: str = "L1C",
    model_ensembling: bool = False,
    batch_size: int = 2,
    tta_max_depth: int = 1,
    fp16_mode: Union[bool, None] = None,
    required_bands: List[str] = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    ],
) -> Inf_Only_Settings:
    """
    Create the settings for inference-only mode.

    Parameters
    ----------
    processing_res : int
        The resolution for the imagery processing.
    processing_level : str, optional
        The processing level of the imagery. Default is "L1C".
    model_count : int, optional
        Number of models to use. Default is 1.
    batch_size : int, optional
        The size of the batch to be used in model. Default is 2.
    tta_max_depth : int, optional
        Maximum depth for test time augmentation. Default is 1.
    tta_min_depth : int, optional
        Minimum depth for test time augmentation. Default is 1.
    tta_early_stop : bool, optional
        Whether to stop early during test time augmentation. Default is True.
    fp16_mode : bool, optional
        If True, use 16-bit precision for the model, else use 32-bit. If None,
        will be determined automatically.
    required_bands : List[str], optional
        List of required bands.

    Returns
    -------
    Inf_Only_Settings
        The inference-only settings object.
    """
    if tta_max_depth < 0 or tta_max_depth > 7:
        raise Exception("tta_max_depth must be between 0 and 7")

    download_model_weights()
    scene_progress_pbar = tqdm(
        total=33,
        smoothing=0,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )
    if type(pytorch_device) != torch.device:
        pytorch_device = torch.device(pytorch_device)

    model_dir = Path(__file__).resolve().parent / "models"
    model_paths = find_models(
        model_dir=model_dir,
        processing_res=processing_res,
        model_ensembling=model_ensembling,
    )
    # load the models
    models = load_models(model_paths=model_paths, pytorch_device=pytorch_device)

    # if fp16_mode is not set then check if we can use it
    if fp16_mode is None:
        fp16_mode = fp16_available(
            pytorch_device=pytorch_device,
            models=models,
        )
    # if fp16_mode is True the convert the models to half precision
    for index, model in enumerate(models):
        if fp16_mode:
            models[index] = model.half()
        else:
            models[index] = model.float()
    # create a TTA augmentation list
    ordered_augs = get_tta_options(tta_max_depth)

    mean, std = get_normalization_stats(
        processing_level=processing_level,
        fp16_mode=fp16_mode,
        pytorch_device=pytorch_device,
        required_bands=required_bands,
    )

    return Inf_Only_Settings(
        processing_res=processing_res,
        processing_level=processing_level,
        mean=mean,
        std=std,
        scene_progress_pbar=scene_progress_pbar,
        pytorch_device=pytorch_device,
        batch_size=batch_size,
        tta_max_depth=tta_max_depth,
        fp16_mode=fp16_mode,
        models=models,
        ordered_augs=ordered_augs,
    )
