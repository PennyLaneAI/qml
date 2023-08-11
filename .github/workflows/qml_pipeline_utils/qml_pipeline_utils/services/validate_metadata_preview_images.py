import os
import json
from pathlib import Path
from typing import List, Union, TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from pathlib import Path


def validate_metadata_preview_images(metadata_files: List[str], sphinx_build_directory: Path):
    working_dir = Path(os.getcwd())

    sphinx_build_directory = working_dir / sphinx_build_directory \
        if not sphinx_build_directory.is_absolute() \
        else sphinx_build_directory

    for metadata_file in metadata_files:
        if not metadata_file.startswith("/"):
            metadata_file_path = Path(working_dir) / metadata_file
        else:
            metadata_file_path = Path(metadata_file)

        with metadata_file_path.open() as fh:
            metadata = json.load(fh)

        preview_images = metadata.get("previewImages", [])

        image_uris = [
            image["uri"]
            for image in preview_images
        ]

        for image_uri in tqdm(image_uris, ascii=True, desc=metadata_file):
            if not image_uri.startswith("/"):
                print(f"  Metadata file {metadata_file} contains either remote image uri or non-absolute path "
                      f"for previewImage {image_uri}. Remote image paths cannot be validated at this time. Skipping.")
                continue

            uri_path = sphinx_build_directory / image_uri[1:]

            assert uri_path.exists(), f"Could not find previewImage {image_uri}"

    return None



