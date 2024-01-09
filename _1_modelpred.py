'''---------------------------------------------------------------------------------------------------------------------
# ---concise version, 2024, Geological Survey of Canada---

# language: Python 3.8

# OBJECTIVE: This code is used to load a CNN model for MT-3D inversion.

# ACKNOWLEDGEMENTS: Github, kaggle, google, ModEM, MTPy, etc. for sharing open source code and library.

# INTEREST CONFLICT: None

'''
#-----------------------------------------------------------------------------------------------------------------------
import os
import time
import warnings
import shutil
import numpy as np
import pandas as pd
import subprocess
import pickle
import matplotlib.pyplot as plt
import pyvista as pv
import mtpy.modeling.modem as modem

from itertools import cycle, permutations, product
from datetime import datetime
from mtpy.modeling.modem import Data as DataEM
from joblib import Parallel, delayed, dump, load
from tqdm import tqdm_notebook
from mtpy.core.mt import MT
from mtpy.modeling.modem import PlotResponse
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d, interp2d, griddata, interpn, RegularGridInterpolator, LinearNDInterpolator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.layers.merge import concatenate
from keras.engine.training import Model
from keras.layers import Input,Flatten,Dense,Reshape
from tensorflow.keras import optimizers
from itools import u_net_3D_1 as un
warnings.filterwarnings('ignore')   # ignore warning, such as divide by zero, 2023-04-11
modem_path = '.../ModEM/f90/Mod3DMT'
root_path = os.getcwd()
edi_file = root_path+'/plc002.edi'
#-----------------------------------------------------------------------------------------------------------------------

fold_train = 'train_unet_OS'
RMIN, RMAX = 0, 3
n_f = 32
n_x, n_y, n_z = 17, 17, 25
n_tile = (4, 2, 6)

n_sample = 100

epochs = 100

batch_size = 16

version = 1

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
def main():
    t_start = time.time()
    period_list = np.logspace(-2, 3, n_f)
    folder_data = f'data_{version}'   # version
    folder_mask = f'mask_{version}'
    basic_name = f'CNN2_v{version}'
    edi_folder = 'edi_files'
    save_model_name = basic_name + '.model'
    savepath = os.path.join(root_path, fold_train)
    ModData_path = os.path.join(savepath, 'ModEM_Data.dat')
    frequency = 1.0 / period_list
    nodes_north = np.array([6000.0] + [3000.0] * (n_x - 2) + [6000.0])
    nodes_east = np.array([6000.0] + [3000.0] * (n_y - 2) + [6000.0])
    nodes_z = np.array([20.0, 40.0, 60.0, 80.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1000.0,
         1500.0, 1500.0, 2000.0, 2000.0, 2500.0, 2500.0, 3000.0, 3000.0, 3500.0, 3500.0])
    stations = [(i, j, 0) for i in np.linspace(-0.3, 0.3, 7) for j in np.linspace(-0.2, 0.2, 7)]
    input_size = (len(stations) * n_tile[0], n_f * n_tile[1], 6 * 2 * n_tile[2], 1)
    output_size = (n_x, n_y, n_z, 1)
    nodes_int = np.cumsum(nodes_z)
    savepath_result = os.path.join(root_path, 'results')  # save results

# load data for training,
    if True:  # True  False
        with open(os.path.join(train_fold,f'training_datasets_{version}.pkl'),'rb') as f1:
            train_df,train_data_all,train_mask_all = pickle.load(f1)
        f1.close()
        d_shape = train_data_all.shape
        scaler0 = load('std_scaler.bin');
        train_data_N = scaler0.transform(train_data_all.reshape(d_shape[0], -1)).reshape(d_shape);
        x_train0, x_valid0, y_train, y_valid = un.shuffle(train_data_N, np.expand_dims(train_mask_all,-1), test_size=5)
        x_valid0 = np.expand_dims(np.tile(x_valid0[:,:,:,:], n_tile),-1)
        valid_gen = un.gen_chunk(x_valid0,y_valid,batch_size=batch_size)

    model3d = un.build_3D_model(input_size, output_size)
    c = optimizers.Adam(lr=0.01)
    model3d.compile(optimizer=c, loss=un.rmse, metrics=[un.rmse])
    weight_path = f'weights_{version}.best.hdf5'

    model3d.load_weights(weight_path)
    idx1 = 10
    for i in range(idx1, idx1+10):
        xv_out, yv_out = next(valid_gen)
        pre_x = model3d.predict(xv_out)
        print('rmse: ', i, un.rmse(yv_out, pre_x));
        fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(23, 7))
        ax1.plot(yv_out.reshape(-1), 'b');
        ax1.plot(pre_x.reshape(-1), 'r--');
        ax1.legend(['real','pred'])
        ax2.imshow(yv_out[0, :, :, 13], interpolation='spline16', vmin=0, vmax=3, cmap='jet_r')
        ax2.set_title('actual')
        ax3.imshow(pre_x[0, :, :, 13], interpolation='spline16', vmin=0, vmax=3, cmap='jet_r')
        ax3.set_title('prediction')
        plt.show()

if __name__ == '__main__':
    main()
