#!bash

# Initial cleanup
rm -rf dist
rm -rf build
FILE=AlphaRaw.pkg
if test -f "$FILE"; then
  rm AlphaRaw.pkg
fi
cd ../..
rm -rf dist
rm -rf build

# Creating a conda environment
conda create -n alpharawinstaller python=3.8 -y
conda activate alpharawinstaller

# Creating the wheel
python setup.py sdist bdist_wheel

# Setting up the local package
cd release/one_click_macos_gui
pip install "../../dist/alpharaw-0.0.1-py3-none-any.whl[stable]"

# Creating the stand-alone pyinstaller folder
pip install pyinstaller==4.2
pyinstaller ../pyinstaller/alpharaw.spec -y
conda deactivate

# If needed, include additional source such as e.g.:
# cp ../../alpharaw/data/*.fasta dist/alpharaw/data

# Wrapping the pyinstaller folder in a .pkg package
mkdir -p dist/alpharaw/Contents/Resources
cp ../logos/alpha_logo.icns dist/alpharaw/Contents/Resources
mv dist/alpharaw_gui dist/alpharaw/Contents/MacOS
cp Info.plist dist/alpharaw/Contents
cp alpharaw_terminal dist/alpharaw/Contents/MacOS
cp ../../LICENSE.txt Resources/LICENSE.txt
cp ../logos/alpha_logo.png Resources/alpha_logo.png
chmod 777 scripts/*

pkgbuild --root dist/alpharaw --identifier de.mpg.biochem.alpharaw.app --version 0.3.0 --install-location /Applications/AlphaRaw.app --scripts scripts AlphaRaw.pkg
productbuild --distribution distribution.xml --resources Resources --package-path AlphaRaw.pkg dist/alpharaw_gui_installer_macos.pkg
