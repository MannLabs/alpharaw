conda create -n alpharaw python=3.8 -y
conda activate alpharaw
pip install -e '../.[development]'
alpharaw
conda deactivate
