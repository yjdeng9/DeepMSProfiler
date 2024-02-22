
# DeepMSProfiler

Welcome to DeepMSProfiler, an innovative data analysis tool focused on liquid chromatography-mass spectrometry (LC-MS) data. It harnesses the potential of deep learning to process complex data from different diseases and generate unique disease features.
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


- [Overview](#overview)
- [Documentation](#documentation)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#Demo)
- [License](#license)
- [Citation](#Citation)
- [Issues](https://github.com/yjdeng9/DeepMSProfiler/issues)
- 
# overview

Unlike traditional metabolomics data analysis tools, DeepMSProfiler is a tool for mining global features from raw metabolomics data. It takes raw metabolomics data from different disease groups as input and provides three main outputs:
1. Sample disease type labels.
2. Heatmaps depicting the correlation of different metabolite signals with diseases.
3. Disease-associated metabolite-protein network plots.



## Why DeepMSProfiler?

DeepMSProfiler stands out due to the following advantages:
- Superior analysis accuracy.
- Increased efficiency.
- User-friendliness.
- Suitable for both experts and beginners.

## Getting Started

To get started with DeepMSProfiler, follow these steps:
1. Clone this repository to your local machine.
2. Install the required dependencies (install_dependencies.sh).
3. Run the main script (mainRun.py) with your input data with parameters changing if you need.

# Demo

You can run DeepMSProfiler using the following command:

```bash
python mainRun.py -data ../example/data/ -label ../example/label.txt -out ../jobs -run_train -run_pred -run_feature
```
We provide a pre-trained model based on 859 serum metabolomics samples (210 healthy individuals, 323 lung nodules, 326 lung cancer) for academic use.

**Command Line Arguments:**

    -data: Specifies the path to the raw metabolomics data. Default is ../example/data/.
    -label: Specifies the path to the sample disease type labels file. Default is ../example/label.txt.
    -out: Specifies the directory where the output results will be saved. Default is ../jobs.

Run Options:

    -run_train: Initiates the training process (Boolean, default is False).
    -run_pred: Initiates the prediction process (Boolean, default is False).
    -run_feature: Initiates the feature extraction process (Boolean, default is True).

Model Parameters:

    -arch: Specifies the model architecture, e.g., 'DenseNet121'.
    -pretrain: Specifies the path to the pre-trained model, default is None.
    -lr: Sets the learning rate, default is 1e-4.
    -opt: Specifies the optimizer, e.g., 'adam'.
    -batch: Sets the batch size, default is 8.
    -epoch: Sets the number of training epochs, default is 2.
    -run: Specifies the number of runs, default is 10.

Other Options:

    -models: Specifies the models to be used for prediction, e.g., 'use_old'.
    -mode: Specifies the mode, e.g., 'ensemble'.
    -boost: Enables boosting mode (Boolean).
    -plot_auc: Plots the AUC curve (Boolean).
    -plot_cm: Plots the confusion matrix (Boolean).

Please adjust and expand this explanation based on the specific parameters and usage of your project. This example will help users understand how to run your project and utilize different command-line arguments.

## Contributing

We welcome contributions to this project! If you would like to contribute, please follow the guidelines outlined in the project's README file. Fork the repository, make your changes, and submit a pull request with a clear explanation of the modifications you've made.


# License

Copyright by Yongjie Deng. 

This project is licensed under the Apache License, Version 2.0 and is open for any academic use. Papers related to this project will be submitted, please cite for use and contact the author for data acquisition.


# Citation

Paper is in the submission process.

For usage of the package and associated manuscript, please cite 

**An explainable end-to-end deep learning method for direct analysis of mass spectrometry data to reveal disease-specific metabolic profiles**

Yongjie Deng, Weizhong Li and et al.

![image](https://github.com/yjdeng9/DeepMSProfiler/assets/130525414/1ce3d2e8-60eb-40bf-afa8-ffb5dddb6b25)



## Contact

Yongjie Deng - dengyj9@mail2.sysu.edu.cn

Weizhong Li - liweizhong@mail.sysu.edu.cn


