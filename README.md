# VNO_submission
The code for the Vandermonde Neural Operators (VNO) submission to NeurIPS 2023. This code presents the implementations of VNOs for several numerical experiments; Burgers' Equation, shear layer, surface-level specific humidity, and flow past an airfoil. 

PyTorch version 1.12.1+cu113 was used in these experiments. Information on installing this version is availabe at [PyTorch](https://pytorch.org/get-started/previous-versions/). Experiments were run on a NVidia GeForce RTX_3090 with 24 GB of memory. Running on other platforms may require adjustment of the batch size.  

## Burgers' Equation
* To run this experiment, modify the variable on line 163 ``data_dist = 'uniform', 'conexp',`` or `` rand'``, corresponding to one of the data sets. 
* Also modify the variable on line 166 ``PATH`` to lead to the path of the data. 

Run the experiment with the command ``python3 vno_burgers.py`` in the terminal.

## Shear Layer
* Modify variable on line 94 ``file_path`` to lead to the data for this experiment. 
* If you would like to load a model, set ``configs['load_model']`` to ``True``, and modify the variable on line 93 ``model_path`` to lead to this model. 
* If you would like to save a model, set ``configs['save_model']`` to ``True``, and modify the path in line 197 within the ``torch.save()`` to the location where you would like to save this. 

The file may be run from the terminal with the command ``python3 vno_shearflow.py``.

## Surface-Level Specific Humidity
* Modify variable on line 94 ``file_path`` to lead to the data for this experiment. 
* If you would like to load a model, set ``configs['load_model']`` to ``True``, and modify the variable on line 93 ``model_path`` to lead to this model. 
* If you would like to save a model, set ``configs['save_model']`` to ``True``, and modify the path in line 202 within the ``torch.save()`` to the location where you would like to save this. 

The file may be run from the terminal with the command ``python3 vno_earth.py``.

## Flow Past Airfoil
This directory contains two files, one which computes the Vandermonde matrices online (``nvno_naca_airfoil.py``) and one which precomputes the Vandermonde matrices and loads them in the data loader (``nvno_pre_naca_airfoil.py``). The paper only presents results for the online matrix computation method. We include the second method for the curious reader. 
To run the VNO for this experiment, several modifications to the file ``nvno_naca_airfoil.py`` are required. 
* Modify variable on line 57 ``file_path`` to lead to the data for this experiment. 
* If you would like to load a model, set ``configs['load_model']`` to ``True``, and modify the variable on line 58 ``model_path`` to lead to this model. 
* If you would like to save a model, set ``configs['save_model']`` to ``True``, and modify the path in line 132 within the ``torch.save()`` to the location where you would like to save this. 

The file may be run from the terminal with the command ``python3 nvno_naca_airfoil.py``.

## Datasets
Methods to download the data for the *Burgers*', *Shear Layer*, and *Surface-Level Specific Humidity* experiments are provided in the **download_data** directory. It is necessary to have ``wget`` installed. 

To download the data for the *Burgers*' experiment, execute the command ``python3 burgers_download.py``. This will download a compressed directory. This directory can be extracted from the ``tar.gz`` file by executing ``tar -xf burgers.tar.gz``. The directory will contain three files, with the name format ``<distribution>_burgers_data_R10.mat``. The three distributions are ``uniform`` (the original, equispaced, uniform distribution), ``conexp`` (the contracting-expanding distribution), and ``rand`` (the random distribution). 

To download the data for the *Shear Layer* experiment, execute the command ``python3 shearlayer_download.py``. This will download a compressed directory. This directory can be extracted from the ``tar.gz`` file by executing ``tar -xf ddsl.tar.gz``. This will result in a new directory containing all the files for this experiment. Please note, each instance is stored as a separate file, i.e. there are 2048 files within this directory.

To download the data for the *Surface-Level Specific Humidity* experiment, execute the shell script ``./humidity_download.sh``. If this is not already executable, run ``chmod 777 humidity_download.sh``. Please note, this data is provided publicly by [NASA EarthData](https://www.earthdata.nasa.gov/). It is required to have an account, which may be set at the linked website. The username and password will be requested when executing this script.  Additionally, please note that all files will be downloaded into the directory where this script is executed. We recommend to set up a directory for this data in a location suitable for large data storage and execute this script there. 

The data for the *Flow Past Airfoil* experiment are provided by [Li et al., *Fourier Neural Operator with Learned Deformations for PDEs on General Topologies*](https://arxiv.org/abs/2207.05209) and are available in this [Google Drive](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8).

## Models
All models can be downloaded by executing the file in the directory **download_models**. Execute the command ``python3 models_download.py``. This will download a compressed directory. This directory can be extracted from the ``tar.gz`` file by executing ``tar -xf VNO_models.tar.gz``.
