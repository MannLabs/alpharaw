AlphaRaw
================

[![Default installation and tests](https://github.com/MannLabs/alpharaw/actions/workflows/pip_installation.yml/badge.svg)](https://github.com/MannLabs/alpharaw/actions/workflows/pip_installation.yml)
[![Publish on PyPi and release on GitHub](https://github.com/MannLabs/alpharaw/actions/workflows/publish_and_release.yml/badge.svg)](https://github.com/MannLabs/alpharaw/actions/workflows/publish_and_release.yml)
[![pypi](https://img.shields.io/pypi/v/alpharaw)](https://pypi.org/project/alpharaw)
[![pip downloads](https://img.shields.io/pypi/dm/alpharaw?color=blue&label=pip%20downloads)](https://pypi.org/project/alpharaw)
![Python](https://img.shields.io/pypi/pyversions/alpharaw)
[![Documentation Status](https://readthedocs.org/projects/alpharaw/badge/?version=latest)](https://alpharaw.readthedocs.io/en/latest/?badge=latest)

## About

An open-source Python package of the AlphaPept ecosystem from the [Mann
Labs at the Max Planck Institute of
Biochemistry](https://www.biochem.mpg.de/mann) to unify raw MS data
accession and storage. To enable all hyperlinks in this document, please
view it at [GitHub](https://github.com/MannLabs/alpharaw).

- [**About**](#about)
- [**License**](#license)
- [**Installation**](#installation)
  - [**Pip installer**](#pip)
  - [**Developer installer**](#developer)
- [**Usage**](#usage)
  - [**Python and jupyter notebooks**](#python-and-jupyter-notebooks)
- [**Troubleshooting**](#troubleshooting)
- [**Citations**](#citations)
- [**How to contribute**](#how-to-contribute)
- [**Changelog**](#changelog)

------------------------------------------------------------------------

## License

AlphaRaw was developed by the [Mann Labs at the Max Planck Institute of
Biochemistry](https://www.biochem.mpg.de/mann) and is freely available
with an [Apache License](LICENSE.txt). External Python packages
(available in the [requirements](requirements) folder) have their own
licenses, which can be consulted on their respective websites.

------------------------------------------------------------------------

## Installation

Pythonnet must be installed to access Thermo or Sciex raw data.

#### For Windows

Pythonnet will be automatically installed via pip.

#### For Linux

1.  Install Mono from mono-project website [Mono
    Linux](https://www.mono-project.com/download/stable/#download-lin).
    NOTE, the installed mono version should be at least 6.10, which
    requires you to add the ppa to your trusted sources!
2.  Install pythonnet with `pip install pythonnet`.

#### For MacOS including M1/M2 platform

1.  Install [brew](https://brew.sh).
2.  Install mono: `brew install mono`.
3.  If the pseudo mono folder `/Library/Frameworks/Mono.framework/Versions` does not exist, create it by running `sudo mkdir -p /Library/Frameworks/Mono.framework/Versions`.
4.  Link homebrew mono to pseudo mono folder: `sudo ln -s /opt/homebrew/Cellar/mono/6.12.0.182 /Library/Frameworks/Mono.framework/Versions/Current`. Here, `6.12.0.182` is the brew-installed mono version, please check your installed version. Navigate to `/Library/Frameworks/Mono.framework/Versions` and run `ls -l` to verify that the link `Current` points to `/opt/homebrew/Cellar/mono/6.12.0.182`. If `Current` points to a different installation and/or `/opt/homebrew/Cellar/mono/6.12.0.182` is referenced by a different link, delete the corresponding links and run `sudo ln -s /opt/homebrew/Cellar/mono/6.12.0.182 Current`.     
5.  Install pythonnet: `pip install pythonnet`.

------------------------------------------------------------------------

AlphaRaw can be installed and used on all major operating systems
(Windows, macOS and Linux). There are three different types of
installation possible:

- [**Pip installer:**](#pip) Choose this installation if you want to use
  AlphaRaw as a Python package in an existing Python 3.8 environment
  (e.g. a Jupyter notebook).
- [**Developer installer:**](#developer) Choose this installation if you
  are familiar with CLI tools, [conda](https://docs.conda.io/en/latest/)
  and Python. This installation allows access to all available features
  of AlphaRaw and even allows to modify its source code directly.
  Generally, the developer version of AlphaRaw outperforms the
  precompiled versions which makes this the installation of choice for
  high-throughput experiments.

### Pip

AlphaRaw can be installed in an existing Python 3.8 environment with a
single `bash` command. *This `bash` command can also be run directly
from within a Jupyter notebook by prepending it with a `!`*:

``` bash
pip install alpharaw
```

Installing AlphaRaw like this avoids conflicts when integrating it in
other tools, as this does not enforce strict versioning of dependancies.
However, if new versions of dependancies are released, they are not
guaranteed to be fully compatible with AlphaRaw. While this should only
occur in rare cases where dependencies are not backwards compatible, you
can always force AlphaRaw to use dependancy versions which are known to
be compatible with:

``` bash
pip install "alpharaw[stable]"
```

NOTE: You might need to run `pip install pip --upgrade` before installing
AlphaRaw like this. Also note the double quotes `"`.

For those who are really adventurous, it is also possible to directly
install any branch (e.g. `@development`) with any extras
(e.g. `#egg=alpharaw[stable,development-stable]`) from GitHub with e.g.

``` bash
pip install "git+https://github.com/MannLabs/alpharaw.git@development#egg=alpharaw[stable,development-stable]"
```

### Developer

AlphaRaw can also be installed in editable (i.e. developer) mode with a
few `bash` commands. This allows to fully customize the software and
even modify the source code to your specific needs. When an editable
Python package is installed, its source code is stored in a transparent
location of your choice. While optional, it is advised to first (create
and) navigate to e.g. a general software folder:

``` bash
mkdir ~/folder/where/to/install/software
cd ~/folder/where/to/install/software
```

***The following commands assume you do not perform any additional `cd`
commands anymore***.

Next, download the AlphaRaw repository from GitHub either directly or
with a `git` command. This creates a new AlphaRaw subfolder in your
current directory.

``` bash
git clone https://github.com/MannLabs/alpharaw.git
```

For any Python package, it is highly recommended to use a separate
[conda virtual environment](https://docs.conda.io/en/latest/), as
otherwise *dependancy conflicts can occur with already existing
packages*.

``` bash
conda create --name alpharaw python=3.9 -y
conda activate alpharaw
```

Finally, AlphaRaw and all its [dependancies](requirements) need to be
installed. To take advantage of all features and allow development (with
the `-e` flag), this is best done by also installing the [development
dependencies](requirements/requirements_development.txt) instead of only
the [core dependencies](requirements/requirements.txt):

``` bash
pip install -e "./alpharaw[development]"
```

By default this installs loose dependancies (no explicit versioning),
although it is also possible to use stable dependencies
(e.g. `pip install -e "./alpharaw[stable,development-stable]"`).

***By using the editable flag `-e`, all modifications to the [AlphaRaw
source code folder](alpharaw) are directly reflected when running
AlphaRaw. Note that the AlphaRaw folder cannot be moved and/or renamed
if an editable version is installed.***

------------------------------------------------------------------------

## Usage

- [**Python**](#python-and-jupyter-notebooks)

NOTE: The first time you use a fresh installation of AlphaRaw, it is
often quite slow because some functions might still need compilation on
your local operating system and architecture. Subsequent use should be a
lot faster.

### Python and Jupyter notebooks

AlphaRaw can be imported as a Python package into any Python script or
notebook with the command `import alpharaw`.

A brief [Jupyter notebook tutorial](nbs/tutorial.ipynb) on how to use
the API is also present in the [nbs folder](nbs).

------------------------------------------------------------------------

## Troubleshooting

In case of issues, check out the following:

- [Issues](https://github.com/MannLabs/alpharaw/issues): Try a few
  different search terms to find out if a similar problem has been
  encountered before
- [Discussions](https://github.com/MannLabs/alpharaw/discussions): Check
  if your problem or feature requests has been discussed before.

------------------------------------------------------------------------

## Citations

There are currently no plans to draft a manuscript.

------------------------------------------------------------------------

## How to contribute

If you like this software, you can give us a
[star](https://github.com/MannLabs/alpharaw/stargazers) to boost our
visibility! All direct contributions are also welcome. Feel free to post
a new [issue](https://github.com/MannLabs/alpharaw/issues) or clone the
repository and create a [pull
request](https://github.com/MannLabs/alpharaw/pulls) with a new branch.
For an even more interactive participation, check out the
[discussions](https://github.com/MannLabs/alpharaw/discussions) and the
[the Contributors License Agreement](misc/CLA.md).

------------------------------------------------------------------------

## Changelog

See the [HISTORY.md](HISTORY.md) for a full overview of the changes made
in each version.
