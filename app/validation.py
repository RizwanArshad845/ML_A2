"""Task 4.2: upload validation.

Checks, in order:
  1. a file was actually sent
  2. its extension is an accepted image type          -> 415
  3. it is not empty
  4. it really decodes as an image (not just named .jpg) -> 422
  5. the decoded format matches the accepted types      -> 415
The size limit is enforced earlier by Flask (MAX_CONTENT_LENGTH)  -> 413

The file is read into memory only. Nothing is written to disk.
"""
import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

# The model was trained only on JPEG and PNG images (see the Task 1 audit).
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}


class UploadError(Exception):
    def __init__(self, title: str, message: str, status: int = 400):
        super().__init__(message)
        self.title = title
        self.message = message
        self.status = status


@dataclass
class Upload:
    filename: str   # sanitized, safe to store and display
    raw: bytes
    size: int
    sha256: str
    width: int
    height: int
    format: str


def validate_upload(file) -> Upload:
    if file is None or not file.filename:
        raise UploadError("No file selected", "Choose an image file before pressing Analyze.")

    # secure_filename removes path parts and odd characters, e.g. "../../a b.png" -> "a_b.png".
    safe_name = secure_filename(file.filename) or "upload"
    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        shown = f".{ext}" if ext else "Files without an extension"
        raise UploadError("Unsupported file type",
                          f"{shown} is not accepted. Please upload a JPG or PNG image.", 415)

    raw = file.read()
    if not raw:
        raise UploadError("Empty file", "The uploaded file is empty.", 400)

    try:
        with Image.open(BytesIO(raw)) as im:
            im.verify()  # checks the file structure
        with Image.open(BytesIO(raw)) as im:
            im.load()    # full decode, catches truncated files
            fmt, (width, height) = im.format, im.size
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError,
            Image.DecompressionBombError):
        raise UploadError("Image could not be read",
                          f"'{safe_name}' looks like an image but could not be decoded. "
                          "It may be corrupted or not really an image.", 422)

    if fmt not in ALLOWED_FORMATS:
        raise UploadError("Unsupported file type",
                          f"The file content is {fmt}, not JPG or PNG.", 415)

    return Upload(filename=safe_name, raw=raw, size=len(raw),
                  sha256=hashlib.sha256(raw).hexdigest(), width=width, height=height,
                  format=fmt)
