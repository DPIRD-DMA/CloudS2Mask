from setuptools import find_packages, setup

setup(
    name="clouds2mask",
    version="0.1.7",
    description="""Python library for cloud and cloud shadow segmentation in Sentinel-2
    L1C imagery""",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Nick Wright",
    author_email="nicholas.wright@dpird.wa.gov.au",
    url="https://github.com/DPIRD-DMA/CloudS2Mask",
    python_requires=">=3.7",
    packages=find_packages(),
    install_requires=["fastai>=2.7", "timm>=0.9", "tqdm>=4.0", "rasterio>=1.3"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    package_data={"clouds2mask": ["model_download_links.csv"]},
)
