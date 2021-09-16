#!bash

# Initial cleanup
rm -rf dist
rm -rf build
cd ../..
rm -rf dist
rm -rf build

# Creating a conda environment
conda create -n alpharaw_installer python=3.8 -y
conda activate alpharaw_installer

# Creating the wheel
python setup.py sdist bdist_wheel

# Setting up the local package
cd release/one_click_linux_gui
# Make sure you include the required extra packages and always use the stable or very-stable options!
pip install "../../dist/alpharaw-0.0.1-py3-none-any.whl[stable]"

# Creating the stand-alone pyinstaller folder
pip install pyinstaller==4.2
pyinstaller ../pyinstaller/alpharaw.spec -y
conda deactivate

# If needed, include additional source such as e.g.:
# cp ../../alpharaw/data/*.fasta dist/alpharaw/data
# WARNING: this probably does not work!!!!

# Wrapping the pyinstaller folder in a .deb package
mkdir -p dist/AlphaRaw_gui_installer_linux/usr/local/bin
mv dist/AlphaRaw dist/AlphaRaw_gui_installer_linux/usr/local/bin/AlphaRaw
mkdir dist/AlphaRaw_gui_installer_linux/DEBIAN
cp control dist/AlphaRaw_gui_installer_linux/DEBIAN
dpkg-deb --build --root-owner-group dist/AlphaRaw_gui_installer_linux/
