rm -rf _build
conda env remove -n alpharawdocs
conda create -n alpharawdocs python=3.10 -y
# conda create -n alphatimsinstaller python=3.10
conda activate alpharawdocs
# call conda install git -y
# call pip install 'git+https://github.com/MannLabs/alphatims.git#egg=alphatims[gui]' --use-feature=2020-resolver
# brew install freetype
pip install '../.[development]'
make html
conda deactivate
