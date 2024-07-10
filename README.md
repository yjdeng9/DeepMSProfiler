
# DeepMSProfiler

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

Welcome to ``DeepMSProfiler``, an innovative data analysis tool focused on liquid chromatography-mass spectrometry (LC-MS) data. It harnesses the potential of deep learning to process complex data from different diseases and generate unique disease features.

# Contents
- [Overview](#overview)
- [System Requirements](#System-Requirements)
- [Installation Guide](#Installation-Guide)
- [Usage Guide](#usage-guide)
- [Demo](#Demo)
- [License](#license)
- [Citation](#Citation)
- [Issues](https://github.com/yjdeng9/DeepMSProfiler/issues)


  
# Overview

Unlike traditional metabolomics data analysis tools, ``DeepMSProfiler`` is a tool for mining global features from raw metabolomics data. It takes raw metabolomics data from different disease groups as input and provides three main outputs:
1. Sample disease type labels.
2. Heatmaps depicting the correlation of different metabolite signals with diseases.
3. Disease-associated metabolite-protein network plots.

![image](https://github.com/yjdeng9/DeepMSProfiler/assets/130525414/1ce3d2e8-60eb-40bf-afa8-ffb5dddb6b25)



## Why DeepMSProfiler?

DeepMSProfiler stands out due to the following advantages:
- Superior analysis accuracy.
- Increased efficiency.
- User-friendliness.
- Suitable for both experts and beginners.



# System Requirements

Ensure that your system meets these requirements before proceeding with installation and usage.


## Hardware Requirements

The ``DeepMSProfiler`` package is designed to support development in environments with CUDA capability(`CUDA VERSION=10.1`), but it is also compatible with CPU-only environments. The hardware requirements are as follows:

- **Standard Computer:** The meta package requires a standard computer with sufficient RAM to support in-memory operations.


## Software Requirements

The ``DeepMSProfiler`` package development version has been tested on CentOS 7 but is also compatible with Windows environments. It is essential to ensure that the Python environment and dependencies are properly installed. The software requirements are as follows:

- **Operating System Compatibility:** CentOS 7 (tested), Windows (compatible)
- **Python Version:** Python >= 3.6
- **TensorFlow Version:** TensorFlow == 2.2.0
- **Keras Version:** Keras == 2.3.1

### Python Dependencies

Please refer to the `requirements.txt` script for installing Python dependencies.



# Installation Guide

**Install from PyPI:**
```
pip install DeepMSProfiler
```

**Install from source code:**
```
git clone https://github.com/yjdeng9/DeepMSProfiler
cd DeepMSProfiler
bash install_dependencies.sh
```
**Install time:**
  <10 minutes



# Usage Guide
   
You can run ``DeepMSProfiler`` using the following command:

```bash
python mainRun.py -data ../example/data/ -label ../example/label.txt -out ../jobs -run_train -run_pred -run_feature
```
We provide a pre-trained model based on 859 serum metabolomics samples (210 healthy individuals, 323 lung nodules, 326 lung cancer) for academic use. Please contact the author for access.

**Command Line Arguments:**

- **Data Options**:
  -    `-data`: Specifies the path to the raw metabolomics data. Default is `../example/data/`.
  -    `-label`: Specifies the path to the sample disease type labels file. Default is `../example/label.txt`.
  -    `-out`: Specifies the directory where the output results will be saved. Default is `../jobs`.

- **Run Options**:

  -    `-run_train`: Initiates the training process (Boolean, default is `False`).
  -    `-run_pred`: Initiates the prediction process (Boolean, default is `False`).
  -    `-run_feature`: Initiates the feature extraction process (Boolean, default is `True`).

- **Model Parameters**:

  -    `-arch`: Specifies the model architecture, e.g., `'DenseNet121'`.
  -    `-pretrain`: Specifies the path to the pre-trained model, default is `None`.
  -    `-lr`: Sets the learning rate, default is 1e-4.
  -    `-opt`: Specifies the optimizer, e.g., `'adam'`.
  -    `-batch`: Sets the batch size, default is `8`.
  -    `-epoch`: Sets the number of training epochs, default is `2`.
  -    `-run`: Specifies the number of runs, default is `10`.

- **Other Options**:

  -    `-models`: Specifies the models to be used for prediction, e.g., `'use_old'`.
  -    `-mode`: Specifies the mode, e.g., `'ensemble'`.
  -    `-boost`: Enables boosting mode (Boolean).
  -    `-plot_auc`: Plots the AUC curve (Boolean).
  -    `-plot_cm`: Plots the confusion matrix (Boolean).

Please adjust and expand this explanation based on the specific parameters and usage of your project. This example will help users understand how to run your project and utilize different command-line arguments.

# Demo

The demo data can be downloaded from `example` dir or **[Baidu Netdisk](https://pan.baidu.com/s/14v82CMsFZwcTI13iWaTWxA):** `https://pan.baidu.com/s/14v82CMsFZwcTI13iWaTWxA`, **Passward:** `acaa`

The demo files are in. npy format. If you upload a file in. mzML format, and the script will automatically convert to. npy format automatically.

**Run with Demo Data (PyPI Version)**
```bash
from DeepMSProfiler import *


run_train(datalist_path='DeepMSProfiler/example/datalist.txt',data_dir='DeepMSProfiler/example/data',
          job_dir='DeepMSProfiler/example/out/jobs007',epoch=2)
run_predict(job_dir = 'DeepMSProfiler/example/out/jobs007',plot_auc=True,plot_cm=True)
run_feature(job_dir='DeepMSProfiler/example/out/jobs007')
show_feature(job_dir='DeepMSProfiler/example/out/jobs007',mode='ensemble')
```

**Run with Demo Data (Github Source Code Version)**
```bash
python mainRun.py -data ../example/data/ -label ../example/label.txt -out ../jobs -run_train -run_pred -run_feature
```

**Demo Log**
```
[INFO] Start in 2024-02-23 09:38:15
[...]
2024-02-23 09:38:16.529642: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1247] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 5072 MB memory) -> physical GPU (device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:02:00.0, compute capability: 6.0)
2024-02-23 09:38:16.532211: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1247] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 6799 MB memory) -> physical GPU (device: 1, name: Tesla P100-PCIE-16GB, pci bus id: 0000:03:00.0, compute capability: 6.0)
2024-02-23 09:40:14.107222: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2024-02-23 09:40:15.990060: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
[INFO] run in linux platform!
[INFO] Start in ../jobs/jobs007! [2024-02-23 09:38:15]
   FilePath   Label Dataset
0   C01.npy  cancer    test
1   C02.npy  cancer   train
2   C03.npy  cancer   train
3   C04.npy  cancer   train
4   C05.npy  cancer   train
5   C06.npy  cancer   train
6   C07.npy  cancer   train
7   C08.npy  cancer   train
8   C09.npy  cancer    test
9   H01.npy  health    test
10  H02.npy  health   train
11  H03.npy  health    test
12  H04.npy  health   train
13  H05.npy  health    test
14  H06.npy  health   train
15  H07.npy  health   train
16  H08.npy  health   train
17  H09.npy  health   train
18  N01.npy  nodule   train
19  N02.npy  nodule   train
20  N03.npy  nodule   train
21  N04.npy  nodule    test
22  N05.npy  nodule   train
23  N06.npy  nodule   train
24  N07.npy  nodule   train
25  N08.npy  nodule   train
26  N09.npy  nodule   train
[INFO] Step1 Done! [2024-02-23 09:38:16]
[INFO] run model with args: DenseNet121 adam lr 1.0e-04 0
[INFO] run model with args: DenseNet121 adam lr 1.0e-04 1
[INFO] run model with args: DenseNet121 adam lr 1.0e-04 2
[INFO] run model with args: DenseNet121 adam lr 1.0e-04 3
[INFO] run model with args: DenseNet121 adam lr 1.0e-04 4
[INFO] run model with args: DenseNet121 adam lr 1.0e-04 5
[INFO] run model with args: DenseNet121 adam lr 1.0e-04 6
[INFO] run model with args: DenseNet121 adam lr 1.0e-04 7
[INFO] run model with args: DenseNet121 adam lr 1.0e-04 8
[INFO] run model with args: DenseNet121 adam lr 1.0e-04 9
[...]
[INFO] Step2 Train Done! [2024-02-23 10:03:39]
[...]
[INFO] Step2 Pred Done! [2024-02-23 10:09:42]
[INFO] Heatmap Shape:  (6, 341, 341)
[INFO] Save Heatmap to:  ../jobs/jobs007/feature_results/ensemble_RISE.npy
[INFO] Step2 Feature Done! [2024-02-23 10:25:30]
[INFO] All Done! [2024-02-23 10:25:30]
[INFO] End in 2024-02-23 10:25:30
```

**Run with Pretrain Model**
```bash
python mainRun.py -data ../example/all_data/ -label ../example/all_label.txt -out ../jobs -run_pred -pretrain ../example/pretrain_model -plot_cm
```

![image](https://github.com/yjdeng9/DeepMSProfiler/blob/main/example/result/confusion_matrix.svg)

**Show Feature Heatmaps**
```bash
python mainRun.py -data ../example/data/ -label ../example/label.txt -out ../jobs -run_feature
```
After `run_feature` ,the heatmaps were saved in `../jobs/jobs007/feature_results/ensemble_RISE.npy`, so we can then show the feature heatmaps for different classes.
```bash
python showFeature.py
```
![image](https://github.com/yjdeng9/DeepMSProfiler/blob/main/example/result/ensemble_RISE.png)

# License

This project is licensed under the Apache License, Version 2.0 and is open for any academic use. Papers related to this project will be submitted, please cite for use and contact the author for data acquisition.

**Yongjie Deng - dengyj9@mail2.sysu.edu.cn**

**Weizhong Li - liweizhong@mail.sysu.edu.cn**



# Citation

Paper is in the submission process.

For usage of the package and associated manuscript, please cite:
    **An explainable end-to-end deep learning method for direct analysis of mass spectrometry data to reveal disease-specific metabolic profiles.** Yongjie Deng, Weizhong Li and et al.





