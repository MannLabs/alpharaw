import functools
import ssl

import certifi
from alphabase.tools.data_downloader import DataShareDownloader

# On Windows, Python < 3.12 concatenates every certificate of the system store into a single
# blob and hands it to OpenSSL at once, so one malformed certificate in the store makes
# ssl.create_default_context() fail with "ASN1: NOT_ENOUGH_DATA" (fixed in CPython 3.12,
# which loads certificates individually). Using certifi's bundle bypasses the system store.
ssl._create_default_https_context = functools.partial(
    ssl.create_default_context, cafile=certifi.where()
)

raw_dir = "../nbs_tests/test_data"
url_template = (
    "https://datashare.biochem.mpg.de/public.php/dav/files/0lJqqAQQcTd9QNB/{}"
)
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
        DataShareDownloader(url_template.format(test_file), raw_dir).download()
