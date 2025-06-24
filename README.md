# README

## Overview

This repository contains the implementation used in the paper titled "Randomized-MLP Regularization Improves Domain Adaptation and Interpretability in DINOv2".
You will find here the required code to fine-tune DINOv2 model using RMLPs, as well as the linear heads we used for downstream tasks as well as the ViT-UNet hybrid. 
We only include the code for learning depth estimation task on natural images, but the remaining ones are analogous.

You can install the dependencies you need by cloning this repository and running 
```
cd rmlp
pip install -r requirements.txt
```

## Usage

If you want to use RMLPs to fine-tune DINOv2-small backbone on a given dataset, you need to provide a suitable loader.
The loader included in this repository will train on ImageNet-1K and is tailored for that specific data structure,
but the code will work as long as you respect the class attributes and the specified data types in the documentation.
To do so, you need to run

```
python3 contrastive_learning.py /path/to/data /path/to/config.yaml
```

Please make sure to adjust the config.yaml file to your needs. 
Provided values were the ones used in our paper.

For training for depth estimation, you can run

```
python3 dst_depth.py /path/to/your/dst.yaml
```
The script will train the task and evaluate on the metrics you specified in the script.
Same as with the backbone's configuration file, make sure to change the values to your needs.
Values in provided file are the ones used in the our paper.


# License

This repository is licensed under the CC BY 4.0 License. See the LICENSE file for more details.


