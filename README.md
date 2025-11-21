# Projet de recherche étude de biodiversité



## Create necessary packages


First, create the environment with an appropriate version of Python

```bash
conda create -n mrcnn python=3.7
conda activate mrcnn
```

Then, install the packages

```bash
`pip install -r requirements.txt`
```

Install pycocotools sous Windows

```bash
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

Finally, install the git repositories of the Mask_RCNN
```bash
pip install git+https://github.com/TECOLOGYxyz/Mask_RCNN.git
```
