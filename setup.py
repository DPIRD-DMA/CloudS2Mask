from setuptools import find_packages, setup

setup(
    name="CloudS2Mask",
    version="0.1.0",  # Update this for new versions
    description="Python library for cloud and cloud shadow segmentation in Sentinel-2 L1C imagery",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Nick Wright",
    author_email="nicholas.wright@dpird.wa.gov.au",
    url="https://github.com/DPIRD-DMA/CloudS2Mask",
    python_requires=">=3.7",
    packages=find_packages(),
    install_requires=["fastai", "timm", "tqdm", "rasterio"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
