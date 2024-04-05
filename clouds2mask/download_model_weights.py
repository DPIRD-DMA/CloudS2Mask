from pathlib import Path

import pandas as pd
import gdown


def download_file_from_google_drive(file_id: str, destination: Path) -> None:
    """
    Downloads a file from Google Drive and saves it at the given destination using gdown.

    Args:
        file_id (str): The ID of the file on Google Drive.
        destination (Path): The local path where the file should be saved.
    """
    # Construct the Google Drive URL
    url = f"https://drive.google.com/uc?id={file_id}"
    print(url)

    # Use gdown to download the file
    gdown.download(url, str(destination), quiet=False)


def get_model_download_links() -> pd.DataFrame:
    """
    Returns a DataFrame containing the model download links.

    Returns:
        DataFrame: A DataFrame containing the file names and corresponding Google Drive IDs.
    """
    return pd.read_csv(Path(__file__).resolve().parent / "model_download_links.csv")


def download_model_weights() -> None:
    """
    Downloads the model weights from Google Drive and saves them locally.
    """
    df = get_model_download_links()

    for _, row in df.iterrows():
        file_id = str(row["google_drive_id"])
        model_dir = Path(__file__).resolve().parent / "models"
        destination = model_dir / str(row["file_name"])

        # Only download the file if it doesn't exist already
        if not destination.exists():
            model_dir.mkdir(exist_ok=True)
            print(f"Downloading {row['file_name']} to {destination}...")
            download_file_from_google_drive(file_id, destination)

        # If it exists, check if its size is less than or equal to 1 MB
        elif destination.stat().st_size <= 1024 * 1024:
            model_dir.mkdir(exist_ok=True)
            print(f"Downloading {row['file_name']} to {destination}...")
            download_file_from_google_drive(file_id, destination)
