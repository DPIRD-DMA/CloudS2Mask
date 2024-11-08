# Welcome to CloudS2Mask ☁️

**Note, CloudS2Mask is being replaced by [OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask).**

**OmniCloudMask has improved accuracy and imagery compatibility over CloudS2Mask, and should be a drop in replacement in most cases.**

![GitHub](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/version-1.1.0-blue)
![Last Commit](https://img.shields.io/github/last-commit/DPIRD-DMA/CloudS2Mask)

## 💡 About  
CloudS2Mask is an open-source Python library that efficiently segments clouds and cloud shadows in Sentinel-2 imagery using state-of-the-art deep learning techniques.

[Check out the paper](https://doi.org/10.1016/j.rse.2024.114122) for details and benchmarks.

## 🎯 Features  
* High-precision cloud and cloud shadow segmentation for Sentinel-2 L1C imagery.
* Rapid processing: Approximately 2.2 seconds per scene at 20m resolution (RTX 4090, AMD Ryzen 9 5950X).
* Compatibility with both GPU and non-GPU systems.
* Supported on Linux, Windows, and macOS.

## 🚀 Installation  
**Windows Users with NVIDIA GPUs**: Before installing CloudS2Mask, ensure you've installed [PyTorch with CUDA support](https://pytorch.org/get-started/locally/), then follow the steps below.

**Mac and Linux Users**: You can proceed with the installation commands below.

To install within a fresh environment.
```bash
# Create and activate new conda environment using python 3.9 or above
conda create -n clouds2mask python=3.9
conda activate clouds2mask

# If using windows install PyTorch using the command found here
# https://pytorch.org/get-started/locally/

# Install CloudS2Mask
pip install clouds2mask
```
To install within an existing environment you can do one of the following.

Make sure your using **Python 3.9 or above**.

Install using pip.
```console
pip install clouds2mask
```
Or manually.
```console
git clone https://github.com/DPIRD-DMA/CloudS2Mask
cd CloudS2Mask
pip install -q .
cd ..
```

## 💻 Usage  
Here's a simple demonstration of how to use CloudS2Mask:

All you need to do is pass a list of Sentinel-2 level 1C 'SAFE' directories to CloudS2Mask. 

[![Colab_Button]][Link]

[Link]: https://colab.research.google.com/drive/10zyZWCPaGDUO6PKNsyKyxcXIfvkoP2xK?usp=sharing 'Try CloudS2Mask In Colab'

[Colab_Button]: https://img.shields.io/badge/Try%20in%20Colab-grey?style=for-the-badge&logo=google-colab


```python
from pathlib import Path
from clouds2mask import (
    create_settings,
    batch_process_scenes,
)

output_dir = Path("./outputs")
l1c_folders_path = Path("/path/to/your/S2_l1c_SAFE/folders")
l1c_folders = list(l1c_folders_path.glob("*.SAFE"))


scene_settings = create_settings(
    sent_safe_dirs=l1c_folders,
    output_dir=output_dir,
    processing_res=20,
)

paths_to_masks = batch_process_scenes(scene_settings)
```
## ⚙️ Performance Tuning
CloudsS2Mask offers a range of performance and accuracy options, here are some examples,

Settings for high accuracy inference:

```python
scene_settings = create_settings(
    sent_safe_dirs=l1c_folders,
    output_dir=output_dir,
    batch_size=32,
    tta_max_depth=2,
    processing_res=10,
    model_ensembling=True,
)
```
Settings for fast inference:
```python
scene_settings = create_settings(
    sent_safe_dirs=l1c_folders,
    output_dir=output_dir,
    batch_size=32,
    processing_res=20,
    output_compression=None,
)
```
Settings for CPU inference:
```python
scene_settings = create_settings(
    sent_safe_dirs=l1c_folders,
    output_dir=output_dir,
    batch_size=1,
    processing_res=20,
)
```
CloudS2Mask will try to auto detect acceleration cards such as NVIDIA GPUs or Apple MPS, but you can also manually specify them like this:
```python
scene_settings = create_settings(
    sent_safe_dirs=l1c_folders,
    output_dir=output_dir,
    pytorch_device='MPS',
)
```

## 👏 Contributing  
We welcome all contributions! Feel free to open an issue or submit a pull request.

## 📄 License  
This project is licensed under the MIT License - please refer to the LICENSE file for more details.

## 📝 Contact  
For support, bug reporting, or to contribute, feel free to reach out at nicholas.wright@dpird.wa.gov.au.

## 📚 Citation
If you use this work, please cite:

```bibtex
@article{WRIGHT2024114122,
    title = {CloudS2Mask: A novel deep learning approach for improved cloud and cloud shadow masking in Sentinel-2 imagery},
    author = {Nicholas Wright and John M.A. Duncan and J. Nik Callow and Sally E. Thompson and Richard J. George},
    journal = {Remote Sensing of Environment},
    volume = {306},
    pages = {114122},
    year = {2024},
    issn = {0034-4257},
    doi = {https://doi.org/10.1016/j.rse.2024.114122},
    url = {https://www.sciencedirect.com/science/article/pii/S0034425724001330}
}
```