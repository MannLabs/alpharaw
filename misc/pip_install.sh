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


case "$DOTNET_RUNTIME" in
  mono)
    # .NET Framework DLLs (e.g. Sciex Clearcore2, Thermo netfx build) run on Mono.
    conda create -n $ENV_NAME python=$PYTHON_VERSION mono -y
    ;;
  coreclr)
    # Cross-platform .NET runtime for the mono-free Thermo (.NET 8) build.
    conda create -n $ENV_NAME python=$PYTHON_VERSION -c conda-forge dotnet-runtime -y
    ;;
  *)
    # netfx: Windows' built-in .NET Framework needs no extra runtime package.
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    ;;
esac

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
