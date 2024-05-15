INCLUDED_NBS=$(find ./nbdev_nbs -name "*.ipynb")
python -m pytest --nbmake $(echo $INCLUDED_NBS)
