rm -rf _build
conda env remove -n alpharawdocs -y
conda create -n alpharawdocs python=3.11 -y

conda activate alpharawdocs

pip install '../.[development]'
make html
conda deactivate
