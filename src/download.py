"""
Utilities for downloading and verifying pre-trained embedding models.
All functions include error handling and detailed console feedback.
"""

import gzip
import logging
import os
import shutil
from typing import Optional
import zipfile

import gdown

logger = logging.getLogger(__name__)

# Word2Vec Source
W2V_URL = "https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM"
# Word2Vec mirror
W2V_URL_MIRROR = (
    "https://github.com/mmihaltz/word2vec-GoogleNews-vectors/"
    "raw/master/GoogleNews-vectors-negative300.bin.gz"
)
# Expected file sizes for word2vec Google News vectors (in bytes)
# Verify and update size in case of mismatch or manual download
W2V_SIZE = 3_644_258_522      # ~3.39 GB (uncompressed .bin)
W2V_GZ_SIZE = 1_647_046_227   # ~1.53 GB (compressed .gz)
# GloVe Source
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
# Expected file sizes for GloVe 6B (in bytes)
# Verify and update size in case of mismatch or manual download
GLOVE_ZIP_SIZE = 862_182_613  # ~822 MB (full zip archive)
GLOVE_TXT_SIZES = {
    "6B.50d": 171_350_079,    # ~163 MB
    "6B.100d": 347_116_733,   # ~331 MB
    "6B.200d": 693_432_828,   # ~661 MB
    "6B.300d": 1_037_962_819  # ~989 MB
}
QUESTIONS_URL = "http://download.tensorflow.org/data/questions-words.txt"
QUESTIONS_FILE = "questions-words.txt"


def verify_file_size(
    file_path: str,
    expected_size: int,
    strict: bool = False
) -> bool:
    """Check file size with optional strict mode."""
    if not os.path.exists(file_path):
        return False
    actual_size = os.path.getsize(file_path)

    if actual_size == 0:
        logger.warning(f"File is empty: {file_path}")
        return False

    if actual_size != expected_size:
        diff_pct = abs(actual_size - expected_size) / expected_size * 100
        if strict or diff_pct > 5.0:
            logger.warning(
                f"Size mismatch ({diff_pct:.1f}%): "
                f"expected {expected_size:,} bytes, "
                f"got {actual_size:,} bytes"
            )
            if strict:
                return False
    return True


def extract_gzip(gz_path: str, bin_path: str) -> bool:
    """Extract .gz file to uncompressed binary format."""
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(bin_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    except (gzip.BadGzipFile, OSError) as e:
        logger.error(f"Extraction failed: {e}")
        return False


def download_word2vec_model(
    force_download: bool = False,
    data_dir: str = "data"
) -> Optional[str]:
    """
    Download GoogleNews pre-trained Word2Vec model (vectors-negative300).
    Implements smart caching: uses existing .bin or .gz files if valid.
    """
    os.makedirs(data_dir, exist_ok=True)

    gz_path = os.path.join(data_dir, "GoogleNews-vectors-negative300.bin.gz")
    bin_path = os.path.join(data_dir, "GoogleNews-vectors-negative300.bin")

    # Clean cache when force_download is requested
    # force_download: If True, ignores existing files and re-downloads
    if force_download:
        for path, name in [(bin_path, "binary"), (gz_path, "compressed")]:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Removed cached {name} file: {path}")

    # Check if valid uncompressed binary already exists
    if os.path.exists(bin_path):
        if verify_file_size(bin_path, W2V_SIZE, strict=False):
            logger.info(f"Word2Vec binary exists: {bin_path}")
            try:
                with open(bin_path, 'rb') as f:
                    f.read(1024)
                return bin_path
            except Exception as e:
                logger.warning(
                    "File exists but appears corrupted: "
                    f"{e}. Re-downloading..."
                )

    # Check if valid compressed archive exists and extract it
    if os.path.exists(gz_path):
        actual_gz_size = os.path.getsize(gz_path)
        size_diff_pct = abs(actual_gz_size - W2V_GZ_SIZE) / W2V_GZ_SIZE * 100

        if verify_file_size(gz_path, W2V_GZ_SIZE, strict=False):
            logger.info(
                f"Found compressed file (size OK), extracting: {gz_path}"
            )
            if extract_gzip(gz_path, bin_path):
                if verify_file_size(bin_path, W2V_SIZE, strict=False):
                    logger.info(f"Extraction complete: {bin_path}")
                    return bin_path
                else:
                    logger.warning(
                        "Extracted file has incorrect size. "
                        "Re-downloading..."
                    )
        else:
            if size_diff_pct > 5.0:
                logger.info(
                    f"Re-downloading due to significant size mismatch "
                    f"({size_diff_pct:.1f}%)..."
                )
                os.remove(gz_path)
            else:
                logger.info(
                    f"Size slightly differs ({size_diff_pct:.1f}%) "
                    f"but proceeding with extraction..."
                )
                if extract_gzip(gz_path, bin_path):
                    if verify_file_size(bin_path, W2V_SIZE, strict=False):
                        logger.info(f"Extraction complete: {bin_path}")
                        return bin_path
                    else:
                        logger.warning(
                            "Extracted file has incorrect size. "
                            "Re-downloading..."
                        )

    # Verify sufficient disk space before download (~4.5 GB recommended)
    # +0.5 GB buffer
    required_space = W2V_SIZE + W2V_GZ_SIZE + 500_000_000
    free_space = shutil.disk_usage(data_dir).free
    if free_space < required_space:
        logger.error(
            f"Insufficient disk space. Need ~{required_space / 1024**3:.1f} "
            f"GB, have {free_space / 1024**3:.1f} GB available."
        )
        return None

    # Download from Google Drive
    logger.info(
        "Downloading Word2Vec model (GoogleNews-vectors-negative300.bin.gz)..."
    )
    logger.info(
        f"This file is ~{W2V_GZ_SIZE / 1024**3:.1f} GB compressed, "
        f"~{W2V_SIZE / 1024**3:.1f} GB uncompressed. "
        "May take several minutes."
    )

    try:
        gdown.download(W2V_URL, gz_path, quiet=False)
    except Exception as e:
        logger.warning(f"Download failed: {e}. Trying mirror...")
        try:
            gdown.download(W2V_URL_MIRROR, gz_path, quiet=False)
        except Exception as mirror_e:
            logger.error(f"Mirror download failed: {mirror_e}")
            logger.info(
                f"Download manually: {W2V_URL} or {W2V_URL_MIRROR}"
            )
            logger.info(f"Save as: {gz_path}")
            return None

    # Verify downloaded .gz file size before extraction
    if not os.path.exists(gz_path):
        logger.error("Downloaded file not found.")
        return None

    gz_size = os.path.getsize(gz_path)
    logger.info(f"Downloaded compressed file size: {gz_size / 1024**3:.2f} GB")

    if not verify_file_size(gz_path, W2V_GZ_SIZE, strict=False):
        diff_pct = abs(gz_size - W2V_GZ_SIZE) / W2V_GZ_SIZE * 100
        logger.warning(
            f"Size mismatch ({diff_pct:.1f}%): "
            f"expected {W2V_GZ_SIZE:,} bytes, "
            f"got {gz_size:,} bytes, proceeding anyway"
        )

    # Extract the archive
    logger.info("Extracting compressed file...")
    if not extract_gzip(gz_path, bin_path):
        return None

    logger.info(f"Extraction complete: {bin_path}")

    # Final verification of uncompressed file
    if verify_file_size(bin_path, W2V_SIZE, strict=False):
        logger.info(f"Word2Vec (GoogleNews) ready: {bin_path}")
        return bin_path
    else:
        actual = os.path.getsize(bin_path)
        diff_pct = abs(actual - W2V_SIZE) / W2V_SIZE * 100
        logger.warning(
            f"Size mismatch after extraction ({diff_pct:.1f}%): "
            f"expected {W2V_SIZE:,} bytes, got {actual:,} bytes"
            "File may still be usable."
        )
        return bin_path


def get_glove_txt_path(data_dir: str, version: str) -> str:
    """Generate expected path for a specific GloVe .txt file."""
    return os.path.join(data_dir, f"glove.{version}.txt")


def verify_glove_txt(data_dir: str, version: str) -> bool:
    """Check if a specific GloVe .txt file exists and has correct size."""
    txt_path = get_glove_txt_path(data_dir, version)
    expected = GLOVE_TXT_SIZES.get(version)
    if expected is None:
        raise ValueError(f"Unknown GloVe version: {version}")
    return verify_file_size(txt_path, expected, strict=False)


def download_glove_model(
    version: str = "6B.100d",
    force_download: bool = False,
    data_dir: str = "data",
    keep_zip: bool = False
) -> Optional[str]:
    """
    Download Stanford GloVe pre-trained vectors (6B tokens, 400K vocab).

    Available versions:
    - 6B.50d  : 50-dimensional vectors (171 MB)
    - 6B.100d : 100-dimensional vectors (347 MB) — recommended
    - 6B.200d : 200-dimensional vectors (694 MB)
    - 6B.300d : 300-dimensional vectors (1.04 GB)

    Implements smart caching: uses existing .txt if valid,
    reuses .zip if valid and extraction needed.
    """
    expected_txt_size = GLOVE_TXT_SIZES.get(version)
    if expected_txt_size is None:
        logger.error(f"Unknown GloVe version: {version}")
        logger.info("Available: 6B.50d, 6B.100d, 6B.200d, 6B.300d")
        return None

    os.makedirs(data_dir, exist_ok=True)

    zip_path = os.path.join(data_dir, "glove.6B.zip")
    txt_path = get_glove_txt_path(data_dir, version)

    # Clean cache if force_download requested
    if force_download:
        for path in [txt_path, zip_path]:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Removed cached file: {path}")

    # Check if target .txt already exists and is valid
    if os.path.exists(txt_path):
        if verify_file_size(txt_path, expected_txt_size, strict=False):
            logger.info(f"GloVe {version} already exists: {txt_path}")
            # Verify file readability
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    sample = f.read(1024)
                    if not sample.strip():
                        raise ValueError("File appears empty")
                return txt_path
            except Exception as e:
                logger.warning(
                    f"File exists but is unreadable: {e}. "
                    "Re-downloading..."
                )
                if os.path.exists(txt_path):
                    os.remove(txt_path)
        else:
            actual = os.path.getsize(txt_path)
            logger.warning(
                f"Size mismatch for {version}: "
                f"expected {expected_txt_size:,} bytes, "
                f"got {actual:,} bytes"
            )
            # Clean corrupted file
            if os.path.exists(txt_path):
                os.remove(txt_path)

    # Check if zip archive exists and is valid
    zip_valid = (
        os.path.exists(zip_path)
        and verify_file_size(zip_path, GLOVE_ZIP_SIZE, strict=False)
    )

    if zip_valid and not force_download:
        logger.info(f"Found valid zip archive: {zip_path}")
        # Try to extract only the needed file
        if extract_glove_single_file(zip_path, version, data_dir):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    sample = f.read(1024)
                    if not sample.strip():
                        raise ValueError("Extracted file appears empty")
            except Exception as e:
                logger.warning(
                    f"Extracted file is unreadable: {e}. Re-downloading..."
                )
                if os.path.exists(txt_path):
                    os.remove(txt_path)
            else:
                if verify_glove_txt(data_dir, version):
                    logger.info(f"Extracted {version} from existing zip.")
                    # If keep_zip=True, keep .zip archive after extraction
                    # (default False — delete to save space)
                    if not keep_zip:
                        os.remove(zip_path)
                        logger.info(
                            "Removed zip archive "
                            "(use keep_zip=True to retain)."
                        )
                    return txt_path
        else:
            logger.warning("Extraction failed, will re-download full archive.")
    else:
        if os.path.exists(zip_path):
            logger.warning(
                "Zip archive has incorrect size or corrupted. "
                "Re-downloading..."
            )
            os.remove(zip_path)

    # Check disk space before download
    required_space = GLOVE_ZIP_SIZE + expected_txt_size + 200_000_000
    free_space = shutil.disk_usage(data_dir).free
    if free_space < required_space:
        logger.warning(
            "Insufficient disk space. Need ~"
            f"{required_space / 1024**3:.1f} GB, "
            f"have {free_space / 1024**3:.1f} GB available."
        )
        return None

    # Download full zip archive
    logger.info(f"Downloading GloVe {version} (full archive: glove.6B.zip)...")
    logger.info(
        f"This file is ~{GLOVE_ZIP_SIZE / 1024**2:.0f} MB compressed, "
        "contains all 4 vector sizes."
    )

    try:
        gdown.download(GLOVE_URL, zip_path, quiet=False)
    except Exception as e:
        logger.error(f"\nDownload failed: {e}")
        logger.info(f"Download manually: {GLOVE_URL}")
        logger.info(f"Save as: {zip_path}")
        return None

    # Verify downloaded zip
    if not os.path.exists(zip_path):
        logger.error("Downloaded zip file not found.")
        return None

    zip_size = os.path.getsize(zip_path)
    logger.info(f"Downloaded zip size: {zip_size / 1024**3:.2f} GB")

    if not verify_file_size(zip_path, GLOVE_ZIP_SIZE, strict=False):
        logger.warning(
            f"Zip size mismatch: expected {GLOVE_ZIP_SIZE} bytes, "
            f"got {zip_size} bytes, proceeding anyway"
        )

    # Extract the specific version file
    if not extract_glove_single_file(zip_path, version, data_dir):
        return None

    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            sample = f.read(1024)
            if not sample.strip():
                raise ValueError("Extracted file appears empty")
    except Exception as e:
        logger.warning(f"Extracted file is unreadable: {e}. Re-downloading...")
        if os.path.exists(txt_path):
            os.remove(txt_path)
        return None

    # Verify extracted .txt file
    if verify_glove_txt(data_dir, version):
        logger.info(f"GloVe ({version}) ready: {txt_path}")

        # Cleanup zip unless requested otherwise
        if not keep_zip and os.path.exists(zip_path):
            os.remove(zip_path)
            logger.info("Removed zip archive (use keep_zip=True to retain).")

        return txt_path
    else:
        actual = os.path.getsize(txt_path) if os.path.exists(txt_path) else 0
        logger.warning(
            f"Size mismatch for {version}: "
            f"expected {expected_txt_size:,} bytes, got {actual:,} bytes"
        )
        return None


def extract_glove_single_file(
        zip_path: str, version: str, data_dir: str
) -> bool:
    """Extract a single GloVe .txt file from the zip archive."""
    txt_filename = f"glove.{version}.txt"

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Check if file exists in archive
            if txt_filename not in zf.namelist():
                logger.error(f"{txt_filename} not found in zip archive.")
                return False

            logger.info(f"Extracting {txt_filename}...")
            zf.extract(txt_filename, data_dir)

        return True

    except zipfile.BadZipFile:
        logger.error("Corrupted zip file.")
        return False
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def download_analogy_test_set(data_dir: str = "data") -> Optional[str]:
    """
    Download Google Analogy Test Set (questions-words.txt).
    Contains 19,544 analogy questions (8,869 semantic + 10,675 syntactic).
    Source: Tomas Mikolov et al., 2013
    (Efficient Estimation of Word Representations...)
    """
    os.makedirs(data_dir, exist_ok=True)
    dest = os.path.join(data_dir, QUESTIONS_FILE)

    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        logger.info(f"Analogy test set already exists: {dest}")
        return dest

    logger.info("Downloading Google Analogy Test Set (questions-words.txt)...")
    logger.info("Source: Tomas Mikolov et al. (2013)")
    logger.info("Contains 19,544 questions (semantic + syntactic)")

    try:
        gdown.download(QUESTIONS_URL, dest, quiet=False)
        logger.info(f"\nDownload complete: {dest}")

        if os.path.getsize(dest) < 500_000:
            logger.warning(
                "Warning: file size is small "
                f"({os.path.getsize(dest):,} bytes)"
            )
            return None

        return dest
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info(f"Try manual download: {QUESTIONS_URL}")
        logger.info(f"Save as: {dest}")
        return None
