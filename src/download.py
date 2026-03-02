"""
Utilities for downloading and verifying pre-trained embedding models.
All functions include error handling and detailed console feedback.
"""

import gzip
import os
import shutil
import zipfile
from typing import Optional

import gdown

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
        print(f"File is empty: {file_path}")
        return False

    if actual_size != expected_size:
        diff_pct = abs(actual_size - expected_size) / expected_size * 100
        if strict or diff_pct > 5.0:
            print(
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
        print(f"Extraction failed: {e}")
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
                print(f"Removed cached {name} file: {path}")

    # Check if valid uncompressed binary already exists
    if os.path.exists(bin_path):
        if verify_file_size(bin_path, W2V_SIZE, strict=False):
            print(f"Word2Vec binary exists: {bin_path}")
            try:
                with open(bin_path, 'rb') as f:
                    f.read(1024)
                return bin_path
            except Exception as e:
                print(
                    "File exists but appears corrupted: "
                    f"{e}. Re-downloading..."
                )

    # Check if valid compressed archive exists and extract it
    if os.path.exists(gz_path):
        actual_gz_size = os.path.getsize(gz_path)
        size_diff_pct = abs(actual_gz_size - W2V_GZ_SIZE) / W2V_GZ_SIZE * 100

        if verify_file_size(gz_path, W2V_GZ_SIZE, strict=False):
            print(f"Found compressed file (size OK), extracting: {gz_path}")
            if extract_gzip(gz_path, bin_path):
                if verify_file_size(bin_path, W2V_SIZE, strict=False):
                    print(f"Extraction complete: {bin_path}")
                    return bin_path
                else:
                    print(
                        "Extracted file has incorrect size. "
                        "Re-downloading..."
                    )
        else:
            if size_diff_pct > 5.0:
                print(
                    f"Re-downloading due to significant size mismatch "
                    f"({size_diff_pct:.1f}%)..."
                )
                os.remove(gz_path)
            else:
                print(
                    f"Size slightly differs ({size_diff_pct:.1f}%) "
                    f"but proceeding with extraction..."
                )
                if extract_gzip(gz_path, bin_path):
                    if verify_file_size(bin_path, W2V_SIZE, strict=False):
                        print(f"Extraction complete: {bin_path}")
                        return bin_path
                    else:
                        print(
                            "Extracted file has incorrect size. "
                            "Re-downloading..."
                        )

    # Verify sufficient disk space before download (~4.5 GB recommended)
    # +0.5 GB buffer
    required_space = W2V_SIZE + W2V_GZ_SIZE + 500_000_000
    free_space = shutil.disk_usage(data_dir).free
    if free_space < required_space:
        print(f"Insufficient disk space. Need ~{required_space / 1024**3:.1f} "
              f"GB, have {free_space / 1024**3:.1f} GB available.")
        return None

    # Download from Google Drive
    print(
        "Downloading Word2Vec model (GoogleNews-vectors-negative300.bin.gz)..."
    )
    print(
        f"This file is ~{W2V_GZ_SIZE / 1024**3:.1f} GB compressed, "
        f"~{W2V_SIZE / 1024**3:.1f} GB uncompressed. "
        "May take several minutes."
    )

    try:
        gdown.download(W2V_URL, gz_path, quiet=False)
    except Exception as e:
        print(f"Download failed: {e}. Trying mirror...")
        try:
            gdown.download(W2V_URL_MIRROR, gz_path, quiet=False)
        except Exception as mirror_e:
            print(f"Mirror download failed: {mirror_e}")
            print(
                f"Download manually: {W2V_URL} or {W2V_URL_MIRROR}"
            )
            print(f"Save as: {gz_path}")
            return None

    # Verify downloaded .gz file size before extraction
    if not os.path.exists(gz_path):
        print("Downloaded file not found.")
        return None

    gz_size = os.path.getsize(gz_path)
    print(f"Downloaded compressed file size: {gz_size / 1024**3:.2f} GB")

    if not verify_file_size(gz_path, W2V_GZ_SIZE, strict=False):
        diff_pct = abs(gz_size - W2V_GZ_SIZE) / W2V_GZ_SIZE * 100
        print(
            f"Size mismatch ({diff_pct:.1f}%): "
            f"expected {W2V_GZ_SIZE:,} bytes, "
            f"got {gz_size:,} bytes, proceeding anyway"
        )

    # Extract the archive
    print("Extracting compressed file...")
    if not extract_gzip(gz_path, bin_path):
        return None

    print(f"Extraction complete: {bin_path}")

    # Final verification of uncompressed file
    if verify_file_size(bin_path, W2V_SIZE, strict=False):
        print(f"Word2Vec (GoogleNews) ready: {bin_path}")
        return bin_path
    else:
        actual = os.path.getsize(bin_path)
        diff_pct = abs(actual - W2V_SIZE) / W2V_SIZE * 100
        print(
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
        print(f"Unknown GloVe version: {version}")
        print("Available: 6B.50d, 6B.100d, 6B.200d, 6B.300d")
        return None

    os.makedirs(data_dir, exist_ok=True)

    zip_path = os.path.join(data_dir, "glove.6B.zip")
    txt_path = get_glove_txt_path(data_dir, version)

    # Clean cache if force_download requested
    if force_download:
        for path in [txt_path, zip_path]:
            if os.path.exists(path):
                os.remove(path)
                print(f"Removed cached file: {path}")

    # Check if target .txt already exists and is valid
    if os.path.exists(txt_path):
        if verify_file_size(txt_path, expected_txt_size, strict=False):
            print(f"GloVe {version} already exists: {txt_path}")
            # Verify file readability
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    sample = f.read(1024)
                    if not sample.strip():
                        raise ValueError("File appears empty")
                return txt_path
            except Exception as e:
                print(
                    f"File exists but is unreadable: {e}. "
                    "Re-downloading..."
                )
                if os.path.exists(txt_path):
                    os.remove(txt_path)
        else:
            actual = os.path.getsize(txt_path)
            print(
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
        print(f"Found valid zip archive: {zip_path}")
        # Try to extract only the needed file
        if extract_glove_single_file(zip_path, version, data_dir):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    sample = f.read(1024)
                    if not sample.strip():
                        raise ValueError("Extracted file appears empty")
            except Exception as e:
                print(f"Extracted file is unreadable: {e}. Re-downloading...")
                if os.path.exists(txt_path):
                    os.remove(txt_path)
            else:
                if verify_glove_txt(data_dir, version):
                    print(f"Extracted {version} from existing zip.")
                    # If keep_zip=True, keep .zip archive after extraction
                    # (default False — delete to save space)
                    if not keep_zip:
                        os.remove(zip_path)
                        print(
                            "Removed zip archive "
                            "(use keep_zip=True to retain)."
                        )
                    return txt_path
        else:
            print("Extraction failed, will re-download full archive.")
    else:
        if os.path.exists(zip_path):
            print(
                "Zip archive has incorrect size or corrupted. "
                "Re-downloading..."
            )
            os.remove(zip_path)

    # Check disk space before download
    required_space = GLOVE_ZIP_SIZE + expected_txt_size + 200_000_000
    free_space = shutil.disk_usage(data_dir).free
    if free_space < required_space:
        print(
            "Insufficient disk space. Need ~"
            f"{required_space / 1024**3:.1f} GB, "
            f"have {free_space / 1024**3:.1f} GB available."
        )
        return None

    # Download full zip archive
    print(f"Downloading GloVe {version} (full archive: glove.6B.zip)...")
    print(
        f"This file is ~{GLOVE_ZIP_SIZE / 1024**2:.0f} MB compressed, "
        "contains all 4 vector sizes."
    )

    try:
        gdown.download(GLOVE_URL, zip_path, quiet=False)
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print(f"Download manually: {GLOVE_URL}")
        print(f"Save as: {zip_path}")
        return None

    # Verify downloaded zip
    if not os.path.exists(zip_path):
        print("Downloaded zip file not found.")
        return None

    zip_size = os.path.getsize(zip_path)
    print(f"Downloaded zip size: {zip_size / 1024**3:.2f} GB")

    if not verify_file_size(zip_path, GLOVE_ZIP_SIZE, strict=False):
        print(
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
        print(f"Extracted file is unreadable: {e}. Re-downloading...")
        if os.path.exists(txt_path):
            os.remove(txt_path)
        return None

    # Verify extracted .txt file
    if verify_glove_txt(data_dir, version):
        print(f"GloVe ({version}) ready: {txt_path}")

        # Cleanup zip unless requested otherwise
        if not keep_zip and os.path.exists(zip_path):
            os.remove(zip_path)
            print("Removed zip archive (use keep_zip=True to retain).")

        return txt_path
    else:
        actual = os.path.getsize(txt_path) if os.path.exists(txt_path) else 0
        print(
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
                print(f"{txt_filename} not found in zip archive.")
                return False

            print(f"Extracting {txt_filename}...")
            zf.extract(txt_filename, data_dir)

        return True

    except zipfile.BadZipFile:
        print("Corrupted zip file.")
        return False
    except Exception as e:
        print(f"Extraction failed: {e}")
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
        print(f"Analogy test set already exists: {dest}")
        return dest

    print("Downloading Google Analogy Test Set (questions-words.txt)...")
    print("Source: Tomas Mikolov et al. (2013)")
    print("Contains 19,544 questions (semantic + syntactic)")

    try:
        gdown.download(QUESTIONS_URL, dest, quiet=False)
        print(f"\nDownload complete: {dest}")

        if os.path.getsize(dest) < 500_000:
            print(
                "Warning: file size is small "
                f"({os.path.getsize(dest):,} bytes)"
            )
            return None

        return dest
    except Exception as e:
        print(f"Download failed: {e}")
        print(f"Try manual download: {QUESTIONS_URL}")
        print(f"Save as: {dest}")
        return None
