import shutil
import tempfile
from pathlib import Path
from typing import Union

import pandas as pd
import requests
from requests import Response
from tqdm.auto import tqdm


def download_file_from_google_drive(file_id: str, destination: Path) -> None:
    """
    Downloads a file from Google Drive and saves it at the given destination.

    Args:
        file_id (str): The ID of the file on Google Drive.
        destination (Path): The local path where the file should be saved.
    """
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    with requests.Session() as session:
        response = session.get(URL, params={"id": file_id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {"id": file_id, "confirm": token}
            response = session.get(URL, params=params, stream=True)

        save_response_content(response, destination)


def get_confirm_token(response: Response) -> Union[str, None]:
    """
    Extracts the confirmation token from the response cookies.

    Args:
        response (Response): The response received from Google Drive.

    Returns:
        str or None: The confirmation token if found, otherwise None.
    """
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def save_response_content(response: Response, destination: Path) -> None:
    """
    Saves the content of a response to the destination path, with a progress bar.

    Args:
        response (Response): The response received from Google Drive.
        destination (Path): The local path where the content should be saved.
    """
    CHUNK_SIZE = 32768

    # Get the total file size
    file_size = int(response.headers.get("Content-Length", 0))

    # Initialize a progress bar
    progress_bar = tqdm(
        total=file_size, unit="iB", unit_scale=True, desc=destination.name, leave=False
    )

    # Create a temporary file in the same directory as the destination
    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent) as f:
        temp_path = Path(f.name)

        for chunk in response.iter_content(CHUNK_SIZE):
            # Filter out keep-alive new chunks
            if chunk:
                f.write(chunk)
                # Update the progress bar manually
                progress_bar.update(len(chunk))

    progress_bar.close()

    if file_size != 0 and progress_bar.n != file_size:
        # Delete the temporary file if an error occurred
        temp_path.unlink()
        raise Exception("Something went wrong while downloading the model weights.")
    else:
        # Move the temporary file to the final destination
        shutil.move(str(temp_path), str(destination))


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
