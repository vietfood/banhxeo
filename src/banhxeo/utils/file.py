import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

from banhxeo.utils import progress_bar
from banhxeo.utils.logging import default_logger

USER_AGENT = "vietfood/banhxeo"


def download_archive(source: Optional[str], url: str, archive_file_path: Path):
    """Download an archive file from a given URL or Google Drive.

    Args:
        source (Optional[str]): Source type, e.g., 'drive' for Google Drive, or None for direct URL.
        url (str): The URL to download the archive from.
        archive_file_path (Path): The local path where the archive will be saved.

    Raises:
        Any exception raised by gdown or urllib.request on download failure.
    """
    if source == "drive":
        import gdown

        gdown.download(
            url,
            archive_file_path.as_posix(),
            quiet=False,
            user_agent=USER_AGENT,
        )
    else:
        # Copy from: https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py#L27
        with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        ) as response:
            with (
                archive_file_path.open(mode="wb") as file,
                progress_bar(
                    total=response.length,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {archive_file_path.name}",
                ) as pbar,
            ):
                while chunk := response.read(1024 * 32):
                    file.write(chunk)
                    pbar.update(len(chunk))


def check_md5(md5: Optional[str], name: str, archive_file_path: Path):
    """Check the MD5 checksum of a file, optionally prompting the user if not provided.

    Args:
        md5 (Optional[str]): The expected MD5 checksum. If None, user is prompted to continue.
        name (str): Name of the dataset (for logging).
        archive_file_path (Path): Path to the archive file to check.

    Raises:
        ValueError: If the user aborts or if the MD5 does not match.
    """
    if not md5:
        default_logger.warning(
            f"MD5 checksum not provided for dataset {name}. Extraction is potentially unsafe."
        )
        accept = input("Do you want to continue (yes/no): ").strip().lower()
        if accept != "yes":
            if archive_file_path.is_file():
                archive_file_path.unlink()
                default_logger.warning(f"Removed {str(archive_file_path)}")
            raise ValueError(
                f"Extraction aborted by user for dataset {str(archive_file_path)}."
            )
    else:
        default_logger.info(f"Verifying MD5 for {str(archive_file_path)}...")

        with archive_file_path.open(mode="rb") as f:
            file_hash = hashlib.md5()
            while chunk := f.read(8192):
                file_hash.update(chunk)
            archive_md5 = file_hash.hexdigest()

        if archive_md5 != md5:
            archive_file_path.unlink()
            default_logger.warning(f"Removed corrupted file {str(archive_file_path)}")
            raise ValueError(
                f"MD5 mismatch for {str(archive_file_path)}. Expected {md5}, got {archive_md5}. File may be corrupted."
            )

        default_logger.info("MD5 checksum verified.")


def extract_archive(
    ext: str,
    archive_file_path: Path,
    dataset_base_path: Path,
    extracted_data_dir_path: Path,
):
    """Extract an archive file to a target directory.

    Args:
        ext (str): Archive extension, e.g., 'zip', 'tar.gz', or others supported by shutil.
        archive_file_path (Path): Path to the archive file.
        dataset_base_path (Path): Base directory to extract files into.
        extracted_data_dir_path (Path): Path to the extracted data directory (for logging).

    Raises:
        Any exception raised by zipfile, tarfile, or shutil on extraction failure.
    """
    if ext == "zip":
        with zipfile.ZipFile(
            archive_file_path,
            "r",
        ) as zip:
            for member in progress_bar(
                iterable=zip.infolist(),
                total=len(zip.infolist()),
                desc=f"Extracting {archive_file_path.name}",
            ):
                zip.extract(path=dataset_base_path, member=member)
    elif ext == "tar.gz":
        with tarfile.open(archive_file_path, "r") as tar:
            for member in progress_bar(
                iterable=tar.getmembers(),
                total=len(tar.getmembers()),
                desc=f"Extracting {archive_file_path.name}",
            ):
                tar.extract(path=dataset_base_path, member=member)
    else:
        default_logger.warning(
            f"Use shutil to extract {extracted_data_dir_path.name}, this will be very slow"
        )
        shutil.unpack_archive(
            filename=archive_file_path,
            extract_dir=dataset_base_path,
        )


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
