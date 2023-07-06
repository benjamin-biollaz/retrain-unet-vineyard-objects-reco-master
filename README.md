# Vineyard aerial detection: from binary to multi-label image segmentation
This repository holds the code for my Bachelor thesis in information technology at HES-SO Valais/Wallis. I received a binary image segmentation model that identify vine lines only as input. My goal is to extend the capabilities of this model to recognize other objects. The repository contain two branches:
- master: The predictions and training are done by two different models (i.e., one for vine lines, another for the rest)
- retrain: The predictions and training are done on a single model.

## Contents details

### Datasets folder

It contains the images and masks of the different sets.

- check_augmentation_online folder : it contains the patches with matching errors, the lists of all generated patches, a python script and the control excel file.
  

- other folders : each folder contains the images before cutting and a subfolder for the creation of patches.

### Images folder

It contains the image of the architecture of the U-Net model used.

### Ratio_gsd_surface folder

It contains the excel file for calculating the ratios and surfaces covered by test image.

### Results folder

It contains the vine prediction images generated with the validation script.

### Retrain folder

It contains the saved terminal of some iterations.

### Statistiques folder

It contains details of pixel predictions, performance metrics, all retraining.

### Weights folder

It contains the weights of all the models generated (including the initial model).

### Binary script

It allows a binary transformation of the prediction image.

### Config script

It contains some parameters.

### Requirements file

It contains the requirements to setup the environment (_see below_).

### Retrain script

It allows to retrain the model with the possibility to change the parameters inside.

### Statistiques script

It allows to calculate performance metrics with a customized threshold.

### Unet_model script

It contains the construction of the U-Net model.

### Unet_validation script

It allows to test the performance of the model and to generate the prediction image.

Run the validation script :

```bash
python unet_validation.py -f <image_to_validate> -c <cut_size> -g <gt_size> --stats --cmpx <new_data_resolution>
```

With:

- <image_to_validate>: relative path to the image
  

- <cut_size> : patches size for the detection (in this project, using cut_size = 144)
  

- <gt_size> : patches size of the labels (in this project, using gt_size = 144)
  

- _stats_ : to be used to display in the terminal the performance metrics with threshold at 0
  

- <new_data_resolution> : ground sampling distance of the image (if not specified, a ratio based on the height of the image is used)

## Setup of the environment
Before using one of the scripts, here are the instructions (_provided by the DUDE-Lab_) to setup  a virtual environment.

- Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Be familiar with the conda commands using the [CONDA CHEAT SHEET](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)
- Download and install [PyCharm](https://www.jetbrains.com/pycharm/)

### Clone the environment 
(_only if the project is accessed from the [private GitLab repository](http://lxhdude.hevs.ch/gitlab/dude-lab/projects/students/aurore-pittet/retrain-unet-vine-lines-reco)_)

```bash
git clone <url in gitlab to clone> <project name>
cd <project name>
```

### Create the new virtual environnement with conda

```bash
conda create --name <conda_env_name> python=3.7.9
conda activate <conda_env_name>
```

### Install the requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```





