# original code from
# https://github.com/12rambau/rio-vrt/blob/main/rio_vrt/vrt.py

import xml.etree.cElementTree as ET
from pathlib import Path
from typing import List, Tuple
from xml.dom import minidom

import rasterio as rio

from .model_settings import Settings
from .make_qml_files import create_vrt_qml


def _add_source_content(
    Source: ET.Element,
    src: rio.DatasetReader,
    type: str,
    xoff: str,
    yoff: str,
    res: float,
) -> None:
    """
    Adds the content of a source file to the provided XML element.

    Args:
        Source (ET.Element): XML element to add source content.
        src (rio.DatasetReader): Rasterio DatasetReader object of the source file.
        type (str): Data type of the source file.
        xoff (str): The x-offset of the source content.
        yoff (str): The y-offset of the source content.
        res (float): Resolution of the source content.

    Raises:
        ValueError: If the given resolution is not supported.
    """
    width, height = str(src.width), str(src.height)
    blockx = str(src.profile.get("blockxsize", ""))
    blocky = str(src.profile.get("blockysize", ""))

    attr = {"RasterXSize": width, "RasterYSize": height, "DataType": type}

    if blockx and blocky:
        attr["BlockXSize"], attr["BlockYSize"] = blockx, blocky

    ET.SubElement(Source, "SourceProperties", attr)

    attr = {"xOff": "0", "yOff": "0", "xSize": width, "ySize": height}
    ET.SubElement(Source, "SrcRect", attr)

    if res == 10:
        width, height = str(5490 * 2), str(5490 * 2)
    elif res == 20:
        width, height = str(5490), str(5490)
    else:
        raise ValueError(f"Resolution {res} is not supported")

    attr = {"xOff": xoff, "yOff": yoff, "xSize": width, "ySize": height}
    ET.SubElement(Source, "DstRect", attr)


def _extract_file_information(files: List[Path]):
    """
    Extracts spatial information from each file.

    Args:
        files (List[Path]): List of file paths.

    Returns:
        tuple: A tuple containing lists of the left, bottom, right, and top
        bounds, as well as the x and y resolution for each file.
    """
    (
        left_,
        bottom_,
        right_,
        top_,
    ) = (
        [],
        [],
        [],
        [],
    )

    for file in files:
        with rio.open(file) as f:
            left_.append(f.bounds.left)
            right_.append(f.bounds.right)
            top_.append(f.bounds.top)
            bottom_.append(f.bounds.bottom)

    return left_, bottom_, right_, top_


def _calculate_spatial_extend(
    left_: list, bottom_: list, right_: list, top_: list, res: Tuple[float, float]
) -> Tuple:
    """
    Calculates the spatial extent of the dataset.

    Args:
        left_ (list): List of left bounds.
        bottom_ (list): List of bottom
        bounds. right_ (list): List of right bounds. top_ (list): List of top bounds.
        res (Tuple[float, float]): A tuple representing the resolution.

    Returns:
        tuple: A tuple containing the affine transformation, total width, and
        total height.
    """
    left = min(float(l) for l in left_)
    bottom = min(float(b) for b in bottom_)
    right = max(float(r) for r in right_)
    top = max(float(t) for t in top_)
    xres, yres = res

    transform = rio.Affine.from_gdal(left, float(xres), 0, top, 0, -float(yres))
    total_width = round((right - left) / xres)
    total_height = round((top - bottom) / yres)

    return transform, total_width, total_height


def build_vrt(
    scene_settings: Settings,
    files: List[Path],
) -> None:
    """
    Creates a Virtual Raster (VRT) file from multiple files.

    Args:
        scene_settings Settings: Settings object with scene information.
        (List[Path]): List of rasterio readable files.
        res (Tuple[float, float],


    Returns:
        Path: The path to the VRT file.

    Raises:
        ValueError: If the CRS of any file doesn't match the global one.
    """
    # Read global information from the first file
    with rio.open(files[0]) as f:
        crs = f.crs

    # Ensure all files have the same CRS
    for file in files:
        with rio.open(file) as f:
            if f.crs != crs:
                raise ValueError(
                    f'The CRS ({f.crs}) from file "{file}" doesn\'t match the global one ({crs})'
                )

    left_, bottom_, right_, top_ = _extract_file_information(files)
    transform, total_width, total_height = _calculate_spatial_extend(
        left_,
        bottom_,
        right_,
        top_,
        (scene_settings.processing_res, scene_settings.processing_res),
    )

    # Start the tree
    attr = {"rasterXSize": str(total_width), "rasterYSize": str(total_height)}
    VRTDataset = ET.Element("VRTDataset", attr)
    ET.SubElement(VRTDataset, "SRS").text = crs.wkt
    ET.SubElement(VRTDataset, "GeoTransform").text = ", ".join(
        [str(i) for i in transform.to_gdal()]
    )
    ET.SubElement(VRTDataset, "OverviewList", {"resampling": "nearest"}).text = "2 4 8"

    for i, file in enumerate(files, start=1):
        attr = {"dataType": "UInt16", "band": str(i)}
        VRTRasterBands = ET.SubElement(VRTDataset, "VRTRasterBand", attr)
        ET.SubElement(VRTRasterBands, "NoDataValue").text = "0"

        ComplexSource = ET.SubElement(VRTRasterBands, "ComplexSource")

        ET.SubElement(ComplexSource, "SourceFilename", attr).text = str(file.absolute())

        ET.SubElement(ComplexSource, "SourceBand").text = "1"

        with rio.open(file) as src:
            _add_source_content(
                ComplexSource,
                src,
                "UInt16",
                str(
                    abs(
                        round(
                            (src.bounds.left - transform.c)
                            / scene_settings.processing_res
                        )
                    )
                ),
                str(
                    abs(
                        round(
                            (src.bounds.top - transform.f)
                            / scene_settings.processing_res
                        )
                    )
                ),
                scene_settings.processing_res,
            )

        ET.SubElement(ComplexSource, "NODATA").text = str(0)

    # Write the file
    scene_settings.vrt_path.resolve().write_text(
        minidom.parseString(ET.tostring(VRTDataset).decode("utf-8"))
        .toprettyxml(indent="  ")
        .replace("&quot;", '"')
    )

    if scene_settings.make_qml_files:
        create_vrt_qml(scene_settings.vrt_path)
