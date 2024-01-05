# MT 3D inversion with a deep learning framework

Deep learning (DL) based algorithm to realize magnetotelluric (MT) 3D inversion 
* The goal of this study is to realize 3D inversion with deep learning algorithm and its practical application of magnetotelluric data, instead of traditional interative processing.
* Because of some limitation, current uploaded code is for testing the trained CNN model with synthetic models only. We will update code with more details in the future.
  
### Requirements:

* Install python 3.8/newer and anaconda packages: conda install keras, tensorflow, scikit-image, opencv, tqdm, pandas, numpy, seaborn, mtpy, etc. libraries; also need download ModEM package for forward modelling.

### Instruction:
    1. MT3D_unet_1.py is for generating datasets
    2. updating...

Synthetic testing - model and prediction
![Figure_8](https://github.com/Jon-GSC/MT3D_CNN/assets/39324742/c047c636-7e29-4c18-8aac-1033c6041dae)
![Figure_5](https://github.com/Jon-GSC/MT3D_CNN/assets/39324742/4d9d8c07-51e2-4204-9e48-57a1e358d416)
![Synthetic_Cubic_20230810100007](https://github.com/Jon-GSC/MT3D_CNN/assets/39324742/7e3935c1-1056-4c55-9a65-3f84ba1cdeff)
![Synthetic_Cubic_20230810100010](https://github.com/Jon-GSC/MT3D_CNN/assets/39324742/72337c28-c67a-498a-9204-b32f0197bb3b)

