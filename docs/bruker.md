# Handling Bruker TimsTOF raw data 

This document describes how AlphaRaw handles Bruker TimsTOF raw data using an implementation that was originally part of [AlphaTims](https://github.com/MannLabs/alphatims).

## How it works

The basic workflow looks as follows:

* Read data from a [Bruker `.d` folder](#bruker-raw-data).
* Convert data to a [TimsTOF object in Python](#timstof-objects-in-python) and optionally store them as a persistent [HDF5 file](https://www.hdfgroup.org/solutions/hdf5/).
* Use Python's [slicing mechanism](#slicing-timstof-objects) to retrieve data from this object e.g. for visualisation.

Also checkout:

* The [ALphaTims paper](https://doi.org/10.1016/j.mcpro.2021.100149) for a complete overview.
* The [presentation](https://datashare.biochem.mpg.de/s/JlVKCvLHdQjsVZU) at [ISCB](https://www.iscb.org/ismbeccb2021) for a brief video.

### Bruker raw data

Bruker stores TimsTOF raw data in a `.d` folder. The two main files in this folder are `analysis.tdf` and `analysis.tdf_bin`.

The `analysis.tdf` file is an SQL database, in which all metadata are stored together with summarised information. This includes the `Frames` table, wherein information about each individual TIMS cycle is summarised including the retention time, the number of scans (i.e. a single TOF push is related to a single ion mobility value), the summed intensity and the total number of ions that have hit the detector. More details about individual scans of the frames are available in the `PasefFrameMSMSInfo` (for PASEF acquisition) or `DiaFrameMsMsWindows` (for diaPASEF acquisition) tables. This includes quadrupole and collision settings of the frame/scan combinations.

The `analysis.tdf_bin` file is a binary file that contains the number of detected ions per individual scan, all detector arrival times and their intensity values. These values are grouped and compressed per frame (i.e. TIMS cycle), thereby allowing fast appendage during online acquisition.

### TimsTOF objects in Python

First it reads relevant metadata from the `analysis.tdf` SQL database and creates a Python object of the `bruker.TimsTOF` class. Next, it reads the summary information from the `Frames` table and creates three empty arrays:

* An empty `tof_indices` array, in which all TOF arrival times of each individual detector hit will be stored. Its size is determined by summing the number of detector hits for all frames.
* An empty `intensities` array of the same size, in which all intensity values of each individual detector hit will be stored.
* An empty `tof_indptr` array, that will store the number of detector hits per scan. Its size is equal to `(frame_max_index + 1) * scans_max_index + 1`. It includes one additional frame to compensate for the fact that Bruker arrays are 1-indexed, while Python uses 0-indexing. The final `+1` is because this array will be converted to an offset array, similar to the index pointer array of a [compressed sparse row matrix](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_%28CSR,_CRS_or_Yale_format%29). Typical values are `scans_max_index = 1000` and `frame_max_index = gradient_length_in_seconds * 10`, resulting in approximately `len(tof_indptr) = 10000 * gradient_length_in_seconds`.

After reading the `PasefFrameMSMSInfo` or `DiaFrameMsMsWindows` table from the `analysis.tdf` SQL database, four arrays are created:

* A `quad_indptr` array that indexes the `tof_indptr` array. Each element points to an index of the `tof_indptr` where the voltage on the quadrupole and collision cell is adjusted. For PASEF acquisitions, this is typically 20 times per MSMS frame (turning on and off a value for 10 precursor selections) and once per change from an MS (precursor) frame to an MSMS (fragment) frame. For diaPASEF, this is typically twice to 10 times per frame and with a repetitive pattern over the frame cycle. This results in an array of approximately `len(quad_indptr) = 100 * gradient_length_in_seconds`. As with the `tof_indptr` array, this array is converted to an offset array with size `+1`.
* A `quad_low_values` array of `len(quad_indptr) - 1`. This array stores the lower m/z boundary that is selected with the quadrupole. For precursors without quadrupole selection, this value is set to -1.
* A `quad_high_values` array, similar to `quad_low_values`.
* A `precursor_indices` array of `len(quad_indptr) - 1`. For PASEF this array stores the index of the selected precursor. For diaPASEF, this array stores the `WindowGroup` of the fragment frame. A value of 0 indicates an MS1 ion (i.e. precursor) without quadrupole selection.

After processing this summarising information from the `analysis.tdf` SQL database, the actual raw data from the `analysis.tdf_bin` binary file is read and stored in the empty `tof_indices`, `intensities` and `tof_indptr` arrays.

Finally, three arrays are defined that allow quick translation of `frame_`, `scan_` and `tof_indices` to `rt_values`, `mobility_values` and `mz_values` arrays.
* The `rt_values` array is read read directly from the `Frames` table in `analysis.tdf` and has a length equal to `frame_max_index + 1`. Note that an empty zeroth frame with `rt = 0` is created to make Python's 0-indexing compatible with Bruker's 1-indexing.
* The `mobility_values` array is defined by using the function `tims_scannum_to_oneoverk0` from `timsdata.dll` on the first frame and typically has a length of `1000`.
* Similarly, the `mz_values` array is defined by using the function `tims_index_to_mz` from `timsdata.dll` on the first frame. Typically this has a length of `400000`.

All these arrays can be loaded into memory, taking up roughly twice as much RAM as the `.d` folder on disk. This increase in RAM memory is mainly due to the compression used in the `analysis.tdf_bin` file. The HDF5 file can also be compressed so that its size is roughly halved and thereby has the same size as the Bruker `.d` folder, but (de)compression reduces accession times by 3-6 fold.

### Slicing TimsTOF objects

Once a Python TimsTOF object is available, it can be loaded into memory for ultrafast accession. Accession of the `data` object is done by simple Python slicing such as e.g. `selected_ion_indices = data[frame_selection, scan_selection, quad_selection, tof_selection]`. This slicing returns a `pd.DataFrame` for subsequent analysis. The columns of this dataframe contain all information for all selected ions, i.e. `frame`, `scan`, `precursor` and `tof` indices and `rt`, `mobility`, `quad_low`, `quad_high`, `mz` and `intensity` values. See the [tutorial jupyter notebook](nbs/tutorial.ipynb) for usage examples.
