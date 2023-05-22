# VNO_submission
The code for the Vandermonde Neural Operators (VNO) submission to NeurIPS 2023. This code presents the implementations of VNOs for several numerical experiments; Burgers' Equation, shear layer, surface-level specific humidity, and flow past an airfoil. 

PyTorch version 1.12.1+cu113 was used in these experiments. Information on installing this version is found here https://pytorch.org/get-started/previous-versions/. Experiments were run on a NVidia GeForce RTX_3090 with 24 GB of memory. Running on other platforms may require adjustment of the batch size.  

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
