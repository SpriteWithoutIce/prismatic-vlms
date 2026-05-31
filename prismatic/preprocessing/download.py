"""
download.py

Utility functions for downloading and extracting various datasets to (local) disk.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, TypedDict
from zipfile import BadZipFile, ZipFile

import requests
from PIL import Image
from rich.progress import BarColumn, DownloadColumn, MofNCompleteColumn, Progress, TextColumn, TransferSpeedColumn
from tqdm import tqdm

from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Dataset Registry w/ Links ===
# fmt: off
DatasetComponent = TypedDict(
    "DatasetComponent",
    {"name": str, "extract": bool, "extract_type": str, "url": str, "do_rename": bool},
    total=False
)

DATASET_REGISTRY: Dict[str, List[DatasetComponent]] = {
    # === LLaVa v1.5 Dataset(s) ===

    # Note =>> This is the full suite of datasets included in the LLaVa 1.5 "finetuning" stage; all the LLaVa v1.5
    #          models are finetuned on this split. We use this dataset for all experiments in our paper.
    "llava-laion-cc-sbu-558k": [
        {
            "name": "chat.json",        # Contains the "chat" traces :: {"human" => <prompt>, "gpt" => <caption>}
            "extract": False,
            "url": "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json",
            "do_rename": True,
        },
        {
            "name": "images",           # Contains the LLaVa Processed Images (jpgs, 224x224 resolution)
            "extract": True,
            "extract_type": "directory",
            "url": "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip",
            "do_rename": False,
        }
    ],

    "llava-v1.5-instruct": [
        {
            "name": "llava_v1_5_mix665k.json",
            "extract": False,
            "url": (
                "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json"
            ),
            "do_rename": True,
        },
        {
            "name": "coco/train2017",       # Visual Instruct Tuning images are all sourced from COCO Train 2017
            "extract": True,
            "extract_type": "directory",
            "url": "http://images.cocodataset.org/zips/train2017.zip",
            "do_rename": True,
        },
        {
            "name": "gqa/images",
            "extract": True,
            "extract_type": "directory",
            "url": "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
            "do_rename": True,
        },
        {
            "name": "ocr_vqa/images",
            "extract": True,
            "extract_type": "directory",
            "url": "https://huggingface.co/datasets/qnguyen3/ocr_vqa/resolve/main/ocr_vqa.zip",
            "do_rename": True,
        },
        {
            "name": "textvqa/train_images",
            "extract": True,
            "extract_type": "directory",
            "url": "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
            "do_rename": True,
        },
        {
            "name": "vg/VG_100K",
            "extract": True,
            "extract_type": "directory",
            "url": "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
            "do_rename": True,
        },
        {
            "name": "vg/VG_100K_2",
            "extract": True,
            "extract_type": "directory",
            "url": "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
            "do_rename": True,
        },
    ]
}
# fmt: on


def convert_to_jpg(image_dir: Path) -> None:
    """Handling for OCR-VQA Images specifically; iterates through directory, converts all GIFs/PNGs."""
    overwatch.info(f"Converting all Images in `{image_dir}` to JPG")

    for image_fn in tqdm(list(image_dir.iterdir())):
        if image_fn.suffix in {".jpg", ".jpeg"} or (jpg_fn := image_dir / f"{image_fn.stem}.jpg").exists():
            continue

        if image_fn.suffix == ".gif":
            gif = Image.open(image_fn)
            gif.seek(0)
            gif.convert("RGB").save(jpg_fn)
        elif image_fn.suffix == ".png":
            Image.open(image_fn).convert("RGB").save(jpg_fn)
        else:
            raise ValueError(f"Unexpected image format `{image_fn.suffix}`")


def is_valid_zip(archive_path: Path) -> bool:
    """Return whether a zip archive can be opened and passes CRC checks."""
    try:
        with ZipFile(archive_path) as zf:
            return zf.testzip() is None
    except (BadZipFile, OSError):
        return False


def get_remote_file_metadata(url: str) -> tuple[int | None, bool]:
    """Return `(content_length, supports_range_requests)` for a remote file when available."""
    response = requests.head(url, allow_redirects=True)
    try:
        response.raise_for_status()

        expected_bytes = response.headers.get("content-length")
        expected_bytes = None if expected_bytes is None else int(expected_bytes)
        supports_range_requests = "bytes" in response.headers.get("accept-ranges", "").lower()
        return expected_bytes, supports_range_requests
    finally:
        response.close()


def download_with_progress(url: str, download_dir: Path, chunk_size_bytes: int = 1024) -> Path:
    """Utility function for downloading files from the internet, with a handy Rich-based progress bar."""
    overwatch.info(f"Downloading {(dest_path := download_dir / Path(url).name)} from `{url}`", ctx_level=1)
    partial_path = dest_path.with_suffix(f"{dest_path.suffix}.part")
    expected_bytes, supports_range_requests = get_remote_file_metadata(url)

    def file_looks_complete(path: Path) -> bool:
        size_matches = expected_bytes is None or path.stat().st_size == expected_bytes
        zip_is_valid = dest_path.suffix != ".zip" or is_valid_zip(path)
        return size_matches and zip_is_valid

    if dest_path.exists():
        if file_looks_complete(dest_path):
            return dest_path

        existing_size = dest_path.stat().st_size
        if expected_bytes is not None and existing_size < expected_bytes and not partial_path.exists():
            overwatch.info(f"Existing file `{dest_path}` appears incomplete; resuming from it.", ctx_level=1)
            dest_path.rename(partial_path)
        else:
            overwatch.info(f"Existing file `{dest_path}` appears invalid; re-downloading.", ctx_level=1)
            dest_path.unlink()

    if partial_path.exists():
        partial_size = partial_path.stat().st_size

        if expected_bytes is not None and partial_size > expected_bytes:
            overwatch.info(f"Removing oversized partial download `{partial_path}`", ctx_level=1)
            partial_path.unlink()
        elif expected_bytes is not None and partial_size == expected_bytes:
            if file_looks_complete(partial_path):
                partial_path.rename(dest_path)
                return dest_path

            overwatch.info(f"Existing partial download `{partial_path}` is corrupted; re-downloading.", ctx_level=1)
            partial_path.unlink()

    resume_from = partial_path.stat().st_size if partial_path.exists() else 0
    request_headers = {"Range": f"bytes={resume_from}-"} if resume_from > 0 and supports_range_requests else {}
    if resume_from > 0 and request_headers:
        overwatch.info(f"Resuming download from byte {resume_from}", ctx_level=1)

    response = requests.get(url, stream=True, headers=request_headers)
    try:
        response.raise_for_status()

        write_mode = "ab" if resume_from > 0 and response.status_code == 206 else "wb"
        if resume_from > 0 and response.status_code != 206:
            overwatch.info("Remote host does not support resume for this file; restarting full download.", ctx_level=1)
            resume_from = 0

        # Download w/ Transfer-Aware Progress
        #   => Reference: https://github.com/Textualize/rich/blob/master/examples/downloader.py
        with Progress(
            TextColumn("[bold]{task.description} - {task.fields[fname]}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            transient=True,
        ) as dl_progress:
            response_bytes = response.headers.get("content-length")
            response_bytes = None if response_bytes is None else int(response_bytes)
            total_bytes = expected_bytes if expected_bytes is not None else None
            if total_bytes is None and response_bytes is not None:
                total_bytes = resume_from + response_bytes

            dl_tid = dl_progress.add_task("Downloading", fname=dest_path.name, total=total_bytes, completed=resume_from)
            with open(partial_path, write_mode) as f:
                for data in response.iter_content(chunk_size=chunk_size_bytes):
                    if not data:
                        continue
                    dl_progress.advance(dl_tid, f.write(data))
    finally:
        response.close()

    partial_path.rename(dest_path)
    return dest_path


def extract_with_progress(archive_path: Path, download_dir: Path, extract_type: str, cleanup: bool = False) -> Path:
    """Utility function for extracting compressed archives, with a handy Rich-based progress bar."""
    assert archive_path.suffix == ".zip", "Only `.zip` compressed archives are supported for now!"
    overwatch.info(f"Extracting {archive_path.name} to `{download_dir}`", ctx_level=1)

    # Extract w/ Progress
    with Progress(
        TextColumn("[bold]{task.description} - {task.fields[aname]}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        MofNCompleteColumn(),
        transient=True,
    ) as ext_progress:
        with ZipFile(archive_path) as zf:
            ext_tid = ext_progress.add_task("Extracting", aname=archive_path.name, total=len(members := zf.infolist()))
            extract_path = Path(zf.extract(members[0], download_dir))
            if extract_type == "file":
                assert len(members) == 1, f"Archive `{archive_path}` with extract type `{extract_type} has > 1 member!"
            elif extract_type == "directory":
                for member in members[1:]:
                    zf.extract(member, download_dir)
                    ext_progress.advance(ext_tid)
            else:
                raise ValueError(f"Extract type `{extract_type}` for archive `{archive_path}` is not defined!")

    # Cleanup (if specified)
    if cleanup:
        archive_path.unlink()

    return extract_path


def download_extract(dataset_id: str, root_dir: Path) -> None:
    """Download all files for a given dataset (querying registry above), extracting archives if necessary."""
    os.makedirs(download_dir := root_dir / "download" / dataset_id, exist_ok=True)

    # Download Files => Single-Threaded, with Progress Bar
    dl_tasks = [d for d in DATASET_REGISTRY[dataset_id] if not (download_dir / d["name"]).exists()]
    for dl_task in dl_tasks:
        dl_path = download_with_progress(dl_task["url"], download_dir)

        # Extract Files (if specified) --> Note (assumes ".zip" ONLY!)
        if dl_task["extract"]:
            dl_path = extract_with_progress(dl_path, download_dir, dl_task["extract_type"])
            dl_path = dl_path.parent if dl_path.is_file() else dl_path

        # Rename Path --> dl_task["name"]
        if dl_task["do_rename"]:
            shutil.move(dl_path, download_dir / dl_task["name"])
