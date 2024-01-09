# MT 3D inversion with a deep learning framework

Deep learning (DL) based algorithm to realize magnetotelluric (MT) 3D inversion 
* The goal of this study is to realize 3D inversion with deep learning algorithm and its practical application of magnetotelluric data, instead of traditional interative processing.
* Because of some limitation, current uploaded code is for testing the trained CNN model with synthetic models only. We will update code with more details in the future.
  
### Requirements:

* Install python 3.8/newer and anaconda packages: conda install keras, tensorflow, scikit-image, opencv, tqdm, pandas, numpy, seaborn, mtpy, etc. libraries; also need download ModEM package for forward modelling.

### Instruction:
    1. _1_modelpred.py is for test the model prediction with validation datasets
    2. updating...

### Hardware tested: 

* HP-7920 workstation: 56core CPU; 64G memory; one Nvidia Quadro P5000 GPU.

### Contact: 

* jon.liu@nrcan-rncan.gc.ca

### License

 * Unless otherwise noted, the source code of this project is covered under Crown Copyright, Government of Canada, and is distributed under the MIT Licence.

 * The Canada wordmark and related graphics associated with this distribution are protected under trademark law and copyright law. No permission is granted to use them outside the parameters of the Government of Canada's corporate identity program. For more information, see Federal identity requirements.

### Acknowledgments:

* MT3D_CNN used open source codes and library from github, google, kaggle, and open-sourced geophysical inversion packages mtpy and ModEM. Please cite the related references in your publications.

Synthetic model testing - slice and 3d view of true models and predictions:
![Figure_8](https://github.com/Jon-GSC/MT3D_CNN/assets/39324742/c047c636-7e29-4c18-8aac-1033c6041dae)
![Figure_5](https://github.com/Jon-GSC/MT3D_CNN/assets/39324742/4d9d8c07-51e2-4204-9e48-57a1e358d416)

<span style="display:block;text-align:center">![Synthetic_Cubic_20230810100007](https://github.com/Jon-GSC/MT3D_CNN/assets/39324742/7e3935c1-1056-4c55-9a65-3f84ba1cdeff)</span>
![Synthetic_Cubic_20230810100010](https://github.com/Jon-GSC/MT3D_CNN/assets/39324742/72337c28-c67a-498a-9204-b32f0197bb3b)

