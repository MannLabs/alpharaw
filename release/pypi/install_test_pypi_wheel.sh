conda create -n alpharaw_pip_test python=3.9 -y
conda activate alpharaw_pip_test
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "alpharaw[stable]"
python -c "import alpharaw"
conda deactivate
