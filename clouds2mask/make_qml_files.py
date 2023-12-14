from pathlib import Path


def create_vrt_qml(vrt_path: Path) -> None:
    """
    Creates a QML file for visualizing a Virtual Raster (VRT) file in QGIS.

    Args:
        vrt_path (Path): The path to the VRT file.

    Notes:
        The generated QML file can be used in QGIS to visualize the VRT file.
        The QML file defines rendering properties, including color mapping and
        transparency.
    """
    qml_path = vrt_path.with_suffix(".qml")
    qml_content = """
    <!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
    <qgis maxScale="1e+08" minScale="0" hasScaleBasedVisibilityFlag="0" styleCategories="AllStyleCategories">
        <pipe>
            <rasterrenderer nodataColor="" redBand="4" greenBand="3" alphaBand="-1" opacity="1" blueBand="2" type="multibandcolor">
                <rasterTransparency/>
                <minMaxOrigin>
                    <limits>StdDev</limits>
                    <extent>UpdatedCanvas</extent>
                    <statAccuracy>Estimated</statAccuracy>
                    <cumulativeCutLower>0.02</cumulativeCutLower>
                    <cumulativeCutUpper>0.98</cumulativeCutUpper>
                    <stdDevFactor>2</stdDevFactor>
                </minMaxOrigin>
                <redContrastEnhancement>
                    <algorithm>StretchToMinimumMaximum</algorithm>
                </redContrastEnhancement>
                <greenContrastEnhancement>
                    <algorithm>StretchToMinimumMaximum</algorithm>
                </greenContrastEnhancement>
                <blueContrastEnhancement>
                    <algorithm>StretchToMinimumMaximum</algorithm>
                </blueContrastEnhancement>
            </rasterrenderer>
        </pipe>
    </qgis>
    """
    with open(qml_path, "w") as qml_file:
        qml_file.write(qml_content)


def create_cloud_mask_classification_qml(cloud_mask_path: Path) -> None:
    """
    Creates a QML file for visualizing a cloud mask file in QGIS.

    Args:
        cloud_mask_path (Path): The path to the cloud mask file.

    Notes:
        The generated QML file can be used in QGIS to visualize the cloud mask
        file. The QML file defines rendering properties, including transparency
        and color mapping for different types of clouds and their shadows.
    """
    qml_path = cloud_mask_path.with_suffix(".qml")
    qml_content = """
    <!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'> <qgis
    maxScale="0" hasScaleBasedVisibilityFlag="0"
    styleCategories="AllStyleCategories" minScale="1e+08">
        <pipe>
            <rasterrenderer nodataColor="" type="paletted" opacity="0.5"
            band="1">
                <rasterTransparency/> <minMaxOrigin>
                    <limits>None</limits> <extent>WholeRaster</extent>
                    <statAccuracy>Estimated</statAccuracy>
                    <cumulativeCutLower>0.02</cumulativeCutLower>
                    <cumulativeCutUpper>0.98</cumulativeCutUpper>
                    <stdDevFactor>2</stdDevFactor>
                </minMaxOrigin> <colorPalette>
                    <paletteEntry alpha="0" color="#cfd940" label="Clear"
                    value="0"/> <paletteEntry alpha="255" color="#03fcf8"
                    label="Thick Cloud" value="1"/> <paletteEntry alpha="255"
                    color="#ff5ffb" label="Thin Cloud" value="2"/>
                    <paletteEntry alpha="255" color="#f2ff00" label="Shadow"
                    value="3"/>
                </colorPalette>
            </rasterrenderer>
        </pipe>
    </qgis>
    """
    with open(qml_path, "w") as qml_file:
        qml_file.write(qml_content)


def create_cloud_mask_confidence_qml(cloud_mask_path: Path) -> None:
    """
    Creates a QML file for visualizing a cloud mask file in QGIS, taking into
    account the confidence level of the mask.

    Args:
        cloud_mask_path (Path): The path to the cloud mask file.

    Notes:
        The generated QML file can be used in QGIS to visualize the cloud mask
        file. The QML file defines rendering properties, including transparency
        and color mapping, and it uses multiband color rendering for
        representing different confidence levels in the cloud mask.
    """
    qml_path = cloud_mask_path.with_suffix(".qml")
    qml_content = """
    <!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'> <qgis
    version="3.28.2-Firenze" minScale="1e+08" maxScale="0"
    hasScaleBasedVisibilityFlag="0" styleCategories="AllStyleCategories">
        <pipe>
            <provider>
                <resampling enabled="false" maxOversampling="2"
                zoomedInResamplingMethod="nearestNeighbour"
                zoomedOutResamplingMethod="nearestNeighbour"/>
            </provider> <rasterrenderer type="multibandcolor" redBand="2"
            greenBand="3" blueBand="4" alphaBand="-1" opacity="0.5"
            nodataColor="">
                <rasterTransparency/> <minMaxOrigin>
                    <limits>MinMax</limits> <extent>WholeRaster</extent>
                    <statAccuracy>Estimated</statAccuracy>
                    <cumulativeCutLower>0.02</cumulativeCutLower>
                    <cumulativeCutUpper>0.98</cumulativeCutUpper>
                    <stdDevFactor>2</stdDevFactor>
                </minMaxOrigin>
            </rasterrenderer> <brightnesscontrast brightness="0" contrast="0"
            gamma="1"/> <huesaturation grayscaleMode="0" invertColors="0"
            colorizeOn="0" colorizeRed="255" colorizeGreen="128"
            colorizeBlue="128" colorizeStrength="100" saturation="0"/>
            <rasterresampler maxOversampling="2"/>
        </pipe>
    </qgis>
    """
    with open(qml_path, "w") as qml_file:
        qml_file.write(qml_content)
