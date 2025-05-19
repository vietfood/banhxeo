import hashlib
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

from banhxeo.utils import progress_bar
from banhxeo.utils.logging import DEFAULT_LOGGER

USER_AGENT = "vietfood/banhxeo"


def download_archive(source: Optional[str], url: str, archive_file_path: Path):
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
    if not md5:
        DEFAULT_LOGGER.warning(
            f"MD5 checksum not provided for dataset {name}. Extraction is potentially unsafe."
        )
        accept = input("Do you want to continue (yes/no): ").strip().lower()
        if accept != "yes":
            if archive_file_path.is_file():
                archive_file_path.unlink()
                DEFAULT_LOGGER.warning(f"Removed {archive_file_path.name}")
            raise ValueError(
                f"Extraction aborted by user for dataset {archive_file_path.name}."
            )
    else:
        DEFAULT_LOGGER.info(f"Verifying MD5 for {archive_file_path.name}...")

        with archive_file_path.open(mode="rb") as f:
            file_hash = hashlib.md5()
            while chunk := f.read(8192):
                file_hash.update(chunk)
            archive_md5 = file_hash.hexdigest()

        if archive_md5 != md5:
            archive_file_path.unlink()
            DEFAULT_LOGGER.warning(f"Removed corrupted file {archive_file_path.name}")
            raise ValueError(
                f"MD5 mismatch for {archive_file_path.name}. Expected {md5}, got {archive_md5}. File may be corrupted."
            )

        DEFAULT_LOGGER.info("MD5 checksum verified.")


def extract_archive(
    ext: str,
    archive_file_path: Path,
    dataset_base_path: Path,
    extracted_data_dir_path: Path,
):
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
        DEFAULT_LOGGER.warning(
            f"Use shutil to extract {extracted_data_dir_path.name}, this will be very slow"
        )
        shutil.unpack_archive(
            filename=archive_file_path,
            extract_dir=dataset_base_path,
        )
