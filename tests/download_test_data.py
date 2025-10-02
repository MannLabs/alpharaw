from alphabase.tools.data_downloader import DataShareDownloader

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
