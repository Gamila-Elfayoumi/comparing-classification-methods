"""Download pinned public study inputs. Files are intentionally gitignored."""
from pathlib import Path
import hashlib
import io
import zipfile
import requests

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "ham/HAM10000_metadata.csv": "https://www.kaggle.com/api/v1/datasets/download/kmader/skin-cancer-mnist-ham10000?file_name=HAM10000_metadata.csv",
    "ham/hmnist_8_8_RGB.csv": "https://www.kaggle.com/api/v1/datasets/download/kmader/skin-cancer-mnist-ham10000?file_name=hmnist_8_8_RGB.csv",
    "blood/blood_cell_anomaly_detection.csv": "https://www.kaggle.com/api/v1/datasets/download/alitaqishah/blood-cell-anomaly-detection-2025?file_name=blood_cell_anomaly_detection.csv",
}

def main():
    for name, url in SOURCES.items():
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            print(f"Downloading {name}", flush=True)
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with path.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                members = [x for x in archive.namelist() if not x.endswith("/")]
                if len(members) != 1 or Path(members[0]).name != path.name:
                    raise ValueError(f"Unexpected archive contents for {name}: {members}")
                raw = archive.read(members[0])
            path.write_bytes(raw)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{name}\t{path.stat().st_size}\t{digest}")

if __name__ == "__main__":
    main()
