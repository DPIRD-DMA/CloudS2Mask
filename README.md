# Welcome to CloudS2Mask ☁️

![GitHub](https://img.shields.io/badge/License-MIT-green)

## 💡 About  
CloudS2Mask is an open-source Python library that efficiently segments clouds and cloud shadows in Sentinel-2 imagery using state-of-the-art deep learning techniques. Benchmarks with the CloudSEN12 Dataset show a 17% error reduction compared to the most competent existing methods.

## 🎯 Features  
* High-precision cloud and cloud shadow segmentation for Sentinel-2 L1C imagery.
* Rapid processing: Approximately 2.2 seconds per scene at 20m resolution (RTX 4090, AMD Ryzen 9 5950X).
* Compatibility with both GPU and non-GPU systems.

## 🚀 Installation  
To install CloudS2Mask, clone this repository and proceed with manual installation.
```console
git clone https://github.com/DPIRD-DMA/CloudS2Mask
cd CloudS2Mask
pip install -q .
```

## 💻 Usage  
Here's a simple demonstration of how to use CloudS2Mask:

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

Settings for high accuracy GPU inference:

```python
scene_settings = create_settings(
    sent_safe_dirs=l1c_folders,
    output_dir=output_dir,
    batch_size=32,
    tta_max_depth=7,
    processing_res=10,
    model_ensembling=True,
)
```
Settings for fast GPU inference:
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
    batch_size=2,
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