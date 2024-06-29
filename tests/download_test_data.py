import os

import requests


def download_file(url, local_filename):
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                # if chunk:
                f.write(chunk)


url_template = (
    "https://datashare.biochem.mpg.de/s/GOiZGGOhrHzS54M/download?path=%2F&files={}"
)
raw_dir = "../nbs_tests/test_data"
test_files = [
    "02112022_Zeno1_TiHe_DIAMA_HeLa_200ng_EVO5_01.wiff",
    "02112022_Zeno1_TiHe_DIAMA_HeLa_200ng_EVO5_01.wiff2",
    "02112022_Zeno1_TiHe_DIAMA_HeLa_200ng_EVO5_01.wiff.scan",
    "iRT.raw",
    "iRT_DIA.raw",
    "multinotch.raw",
]

if __name__ == "__main__":
    for test_file in test_files:
        print(f"Downding {test_file}...")
        download_file(url_template.format(test_file), os.path.join(raw_dir, test_file))
