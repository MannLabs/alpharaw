#!/bin/bash
### Install the package with a given type in a defined conda environment with a define python version,
### and call it to check if it works
### example usage:
### ./pip_install.sh stable my_env 3.9
set -e -u

INSTALL_TYPE=$1 # stable, loose, etc..
ENV_NAME=${2:-alpharaw}
PYTHON_VERSION=${3:-3.9}
DOTNET_RUNTIME=${4:-mono} # mono | coreclr | netfx; selects the .NET runtime backend


# Mono is installed here for the "mono" backend (.NET Framework DLLs, e.g. Sciex
# Clearcore2 and the Thermo netfx build). For "coreclr" the .NET runtime is
# installed separately (see the CI's "Install .NET runtime" step); "netfx" uses
# Windows' built-in .NET Framework and needs no extra package.
if [ "$DOTNET_RUNTIME" = "mono" ]; then
  conda create -n $ENV_NAME python=$PYTHON_VERSION mono -y
else
  conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi

if [ "$INSTALL_TYPE" = "loose" ]; then
  INSTALL_STRING=""
else
  INSTALL_STRING="[${INSTALL_TYPE}]"
fi

# print pip environment for reproducibility
conda run -n $ENV_NAME --no-capture-output pip freeze

# conda 'run' vs. 'activate', cf. https://stackoverflow.com/a/72395091
conda run -n $ENV_NAME --no-capture-output pip install -e "../.$INSTALL_STRING"
conda run -n $ENV_NAME --no-capture-output python -c "import alpharaw"
