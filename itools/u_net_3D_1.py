# functions for model generation and training...
#
import os
import numpy as np
#import pandas as pd
import scipy.interpolate
import six
import cv2
import random
import time
# from random import randint
import pyKriging
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pyvista as pv

plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")
import mtpy.core.edi as mtedi
import scipy.ndimage as scn

from mpl_toolkits.mplot3d import Axes3D
from numpy.random import randint
from mtpy.core import mt as mt
from mtpy.core import z as mtz
from mtpy.utils.calculator import get_period_list
# from mtpy.modeling.modem.exception import ModEMError, DataError
# from sklearn.model_selection import train_test_split
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan
from matplotlib import cm, colors, ticker

from datetime import datetime
from skimage.transform import resize
from scipy.interpolate import interp1d, interp2d, interpn, splrep, Rbf, griddata, RegularGridInterpolator, LinearNDInterpolator, InterpolatedUnivariateSpline
from scipy.spatial import cKDTree,Delaunay
from scipy.signal import convolve
from pykrige.uk3d import UniversalKriging3D
#from pykrige.ok3d import OrdinaryKriging3D
import keras
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add,Lambda,Flatten,Dense,Reshape
from keras import backend as K
from keras.regularizers import l2
from keras import optimizers
#from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv1D, UpSampling1D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate,add
#from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D
from keras.losses import binary_crossentropy
from keras.layers import ConvLSTM2D, Bidirectional, BatchNormalization, Conv3D, Cropping3D, ZeroPadding3D, Activation, Input
from keras.layers import MaxPool3D, UpSampling3D,DepthwiseConv2D, Conv3DTranspose, concatenate
from itertools import permutations, product
import tensorflow as tf
CHANNEL_AXIS = 2

#-----------------------------------------------------------------------------------------------------------------------
def res_generator_3D_inv(n_x, n_y, n_z, rho):
    res_i = rho*np.ones((n_x,n_y,n_z))
    return 10.0**res_i   #np.exp

def res_generator_3D(RMIN, RMAX, n_x, n_y, n_z):
    case = 1
    ns_h = int(n_x/2);   ns_v = int(n_z/2)
    k_psill, k_range, k_nugget = 5.0, 2.0, 0.02
    nx_zone=n_x//ns_h;  ny_zone=n_y//ns_h;  nz_zone=n_z//ns_v;
    r_coarse = np.random.uniform(RMIN, RMAX, nx_zone*ny_zone*nz_zone)
    if case==1:
        ijk0 = list(product(range(nx_zone),range(ny_zone),range(nz_zone)))
        ijk1 = list(map(lambda val: (val[0]*ns_h+randint(ns_h)+1,val[1]*ns_h+randint(ns_h)+1,val[2]*ns_v+randint(ns_v)), ijk0))
        x_coarse = np.array([val[0] for val in ijk1])
        y_coarse = np.array([val[1] for val in ijk1])
        z_coarse = np.array([val[2] for val in ijk1])
        X,Y,Z = np.meshgrid(np.linspace(0, n_x-1, n_x), np.linspace(0, n_y-1, n_y), np.linspace(0, n_z-1, n_z))
        uk3d = UniversalKriging3D(x_coarse,y_coarse,z_coarse,r_coarse,variogram_model='gaussian',
                                  variogram_parameters={'psill':k_range,'range':k_range,'nugget':k_nugget})
        k3d1, ss3d = uk3d.execute('points', X.ravel(), Y.ravel(), Z.ravel())
        res_i = k3d1.reshape(X.shape)
    if case==2:
        n_run = np.random.randint(1,3)
        res_i = 3.0 * np.ones((n_x,n_y,n_z))
        for i in range(n_run):
            step_n = np.random.randint(2,6)
            path = np.unique(paths(step_n,n_x,n_y,n_z), axis=0)
            res_i[path[:,0],path[:,1],path[:,2]] = 2.0
    res_i[res_i < RMIN] = RMIN;   res_i[res_i > RMAX] = RMAX
    return 10**res_i

def paths(i, n_x, n_y, n_z):
    dims = 3
    nxyz = (n_x, n_y, n_z)
    step_set = [-1, 0, 1]
    step_shape = (1, dims)
    if i == 1:
        origin0 = np.array([np.random.randint(low=0, high=val, size=1) for val in nxyz]).T
        steps = np.random.choice(a=step_set, size=step_shape)
        path = np.concatenate([origin0, steps]).cumsum(0)
        for i8, val8 in enumerate(nxyz):
            if path[1, i8] < 0:  path[1, i8] = 0
            if path[1, i8] >= val8:  path[1, i8] = val8 - 1
        return path
    else:
        for idx, val in enumerate(paths(i - 1)):
            steps = np.random.choice(a=step_set, size=step_shape)
            path_temp = np.concatenate([np.expand_dims(val, axis=0), steps]).cumsum(0)
            for i8, val8 in enumerate(nxyz):
                if path_temp[1, i8] < 0:  path_temp[1, i8] = 0
                if path_temp[1, i8] >= val8:  path_temp[1, i8] = val8 - 1
            if idx == 0:
                path = path_temp
            else:
                path = np.concatenate([path, path_temp])
        return path

def write_model_file_Jon(savepath,model_fn,nodes_north,nodes_east,nodes_z,res_scale,res_initial_value=0):
    # credit of MTpy. ns: north(x) cells, es: east(y) cells, zs: vertical(z) cells.
    """
        will write an initial file for ModEM. (based on Mtpy)
        Note that x is assumed to be S --> N, y is assumed to be W --> E and
        z is positive downwards.  This means that index [0, 0, 0] is the
        southwest corner of the first layer.  Therefore if you build a model
        by hand the layer block will look as it should in map view.

        Also, the xgrid, ygrid and zgrid are assumed to be the relative
        distance between neighboring nodes.  This is needed because wsinv3d
        builds the  model from the bottom SW corner assuming the cell width
        from the init file.

        Key Word Arguments:
        ----------------------

            **nodes_north** : np.array(nx)
                        block dimensions (m) in the N-S direction.
                        **Note** that the code reads the grid assuming that
                        index=0 is the southern most point.

            **nodes_east** : np.array(ny)
                        block dimensions (m) in the E-W direction.
                        **Note** that the code reads in the grid assuming that
                        index=0 is the western most point.

            **nodes_z** : np.array(nz)
                        block dimensions (m) in the vertical direction.
                        This is positive downwards.

            **save_path** : string
                          Path to where the initial file will be saved
                          to savepath/model_fn_basename

            **model_fn_basename** : string
                                    basename to save file to
                                    *default* is ModEM_Model.ws
                                    file is saved at savepath/model_fn_basename

            **title** : string
                        Title that goes into the first line
                        *default* is Model File written by MTpy.modeling.modem

            **res_model** : np.array((nx,ny,nz))
                        Prior resistivity model.

                        .. note:: again that the modeling code
                        assumes that the first row it reads in is the southern
                        most row and the first column it reads in is the
                        western most column.  Similarly, the first plane it
                        reads in is the Earth's surface.

            **res_starting_value** : float
                                     starting model resistivity value,
                                     assumes a half space in Ohm-m
                                     *default* is 100 Ohm-m

            **res_scale** : [ 'loge' | 'log' | 'log10' | 'linear' ]
                            scale of resistivity.  In the ModEM code it
                            converts everything to Loge,
                            *default* is 'loge'
        """
    # get resistivity model
    res_model = res_initial_value

    # --> write file
    with open(os.path.join(savepath,model_fn), 'w') as ifid:
        ifid.write('# {0}\n'.format('# MODEL FILE WRITTEN BY MTPY.MODELING.MODEM, Jon'.upper()))
        ifid.write('{0:>5}{1:>5}{2:>5}{3:>5} {4}\n'.format(nodes_north.size, nodes_east.size, nodes_z.size, 0, res_scale.upper()))

        # write S --> N node block
        for ii, nnode in enumerate(nodes_north):
            ifid.write('{0:>12.3f}'.format(abs(nnode)))
        ifid.write('\n')

        # write W --> E node block
        for jj, enode in enumerate(nodes_east):
            ifid.write('{0:>12.3f}'.format(abs(enode)))
        ifid.write('\n')

        # write top --> bottom node block
        for kk, zz in enumerate(nodes_z):
            ifid.write('{0:>12.3f}'.format(abs(zz)))
        ifid.write('\n')

        # write the resistivity in log e format
        if res_scale.lower() == 'loge':
            write_res_model = np.log(res_model[::-1, :, :])
        elif res_scale.lower() == 'log' or res_scale.lower() == 'log10':
            write_res_model = np.log10(res_model[::-1, :, :])
        elif res_scale.lower() == 'linear':
            write_res_model = res_model[::-1, :, :]
        else:
            raise print("resistivity scale \"{}\" is not supported.".format(res_scale))

        # write out the layers from resmodel
        for zz in range(nodes_z.size):
            ifid.write('\n')
            for ee in range(nodes_east.size):
                for nn in range(nodes_north.size):
                    ifid.write('{0:>13.5E}'.format(write_res_model[nn, ee, zz]))
                ifid.write('\n')

        center_east = -nodes_east.__abs__().sum() / 2 + 0.
        center_north = -nodes_north.__abs__().sum() / 2 + 0.
        center_z = 0
        grid_center = np.array([center_north, center_east, center_z])

        ifid.write('\n{0:>16.3f}{1:>16.3f}{2:>16.3f}\n'.format(grid_center[0], grid_center[1], grid_center[2]))
        ifid.write('{0:>9.3f}\n'.format(0))
    ifid.close()

def gen_edi(savepath_edi,edi_file,frequency,stations):
    mt_obj = mt.MT(edi_file)
    new_Z_obj, new_Tipper_obj = mt_obj.interpolate(frequency)
    for idx, val in enumerate(stations):
        mt_obj.station = f'Syn_{idx}'
        mt_obj.lon = -123.5 + val[0]
        mt_obj.lat = 50.6 + val[1]
        mt_obj.write_mt_file(save_dir=savepath_edi,fn_basename=f'Syn_{idx}',file_type='edi',new_Z_obj=new_Z_obj,
                             new_Tipper_obj=new_Tipper_obj,longitude_format='LONG',latlon_format='dd')

def gen_edi_meager(savepath_edi,edi_file,frequency,stations):
    mt_obj = mt.MT(edi_file)
    new_Z_obj, new_Tipper_obj = mt_obj.interpolate(frequency)
    for idx, val in enumerate(stations):
        mt_obj.station = f'Syn_{idx}'
        mt_obj.lon = val[0]
        mt_obj.lat = val[1]
        mt_obj.write_mt_file(save_dir=savepath_edi,fn_basename=f'Syn_{idx}',file_type='edi',new_Z_obj=new_Z_obj,
                             new_Tipper_obj=new_Tipper_obj,longitude_format='LONG',latlon_format='dd')

def read_data_file_Jon(data_fn):
    """ Read ModEM data file, credit of MTpy.
   inputs:
    data_fn = full path to data file name
    center_utm = option to provide real world coordinates of the center of
                 the grid for putting the data and model back into
                 utm/grid coordinates, format [east_0, north_0, z_0]
    Fills attributes:
        * data_array
        * period_list
        * mt_dict
    """
    with open(data_fn, 'r') as dfid:
        dlines = dfid.readlines()
    dfid.close()

    header_list = []
    metadata_list = []
    data_list = []
    period_list = []
    station_list = []
    read_impedance = False
    read_tipper = False
    inv_list = []
    for dline in dlines:
        if dline.find('#') == 0:
            header_list.append(dline.strip())
        elif dline.find('>') == 0:
            # modem outputs only 7 characters for the lat and lon
            # if there is a negative they merge together, need to split them up
            dline = dline.replace('-', ' -')
            metadata_list.append(dline[1:].strip())
            if dline.lower().find('ohm') > 0:
                units = 'ohm'
            elif dline.lower().find('mv') > 0:
                units = '[mV/km]/[nT]'
            elif dline.lower().find('vertical') > 0:
                read_tipper = True
                read_impedance = False
                inv_list.append('Full_Vertical_Components')
            elif dline.lower().find('impedance') > 0:
                read_impedance = True
                read_tipper = False
                inv_list.append('Full_Impedance')
            if dline.find('exp') > 0:
                if read_impedance is True:
                    wave_sign_impedance = dline[dline.find('(') + 1]
                elif read_tipper is True:
                    wave_sign_tipper = dline[dline.find('(') + 1]
            elif len(dline[1:].strip().split()) >= 2:
                if dline.find('.') > 0:
                    value_list = [float(value) for value in
                                  dline[1:].strip().split()]

                    center_point = np.recarray(1, dtype=[('station', '|U10'),
                                                              ('lat', np.float),
                                                              ('lon', np.float),
                                                              ('elev', np.float),
                                                              ('rel_elev', np.float),
                                                              ('rel_east', np.float),
                                                              ('rel_north', np.float),
                                                              ('east', np.float),
                                                              ('north', np.float),
                                                              ('zone', 'U4')])
                    center_point.lat = value_list[0]
                    center_point.lon = value_list[1]
                else:
                    value_fre_site = [float(value) for value in dline[1:].strip().split()]
        else:
            dline_list = dline.strip().split()
            if len(dline_list) == 11:
                for ii, d_str in enumerate(dline_list):
                    if ii != 1:
                        try:
                            dline_list[ii] = float(d_str.strip())
                        except ValueError:
                            pass
                    # be sure the station name is a string
                    else:
                        dline_list[ii] = d_str.strip()
                period_list.append(dline_list[0])
                station_list.append(dline_list[1])

                data_list.append(dline_list)

    period_list = np.array(sorted(set(period_list)))
    station_list = sorted(set(station_list))

    # make a period dictionary to with key as period and value as index
    period_dict = dict([(per, ii) for ii, per in enumerate(period_list)])

    # --> need to sort the data into a useful fashion such that each station is an mt object
    data_dict = {}
    z_dummy = np.zeros((len(period_list), 2, 2), dtype='complex')
    t_dummy = np.zeros((len(period_list), 1, 2), dtype='complex')

    index_dict = {'zxx': (0, 0), 'zxy': (0, 1), 'zyx': (1, 0), 'zyy': (1, 1), 'tx': (0, 0), 'ty': (0, 1)}

    # dictionary for true false if station data (lat, lon, elev, etc)
    # has been filled already so we don't rewrite it each time
    tf_dict = {}
    for station in station_list:
        data_dict[station] = mt.MT()
        data_dict[station].Z = mtz.Z(z_array=z_dummy.copy(),
                                     z_err_array=z_dummy.copy().real,
                                     freq=1. / period_list)
        data_dict[station].Tipper = mtz.Tipper(tipper_array=t_dummy.copy(),
                                               tipper_err_array=t_dummy.copy().real,
                                               freq=1. / period_list)
        # make sure that the station data starts out with false to fill the data later
        tf_dict[station] = False

    # fill in the data for each station
    for dd in data_list:
        # get the period index from the data line
        p_index = period_dict[dd[0]]
        # get the component index from the data line
        ii, jj = index_dict[dd[7].lower()]

        # if the station data has not been filled yet, fill it
        if not tf_dict[dd[1]]:
            data_dict[dd[1]].lat = dd[2]
            data_dict[dd[1]].lon = dd[3]
            data_dict[dd[1]].grid_north = dd[4]
            data_dict[dd[1]].grid_east = dd[5]
            data_dict[dd[1]].grid_elev = dd[6]
            data_dict[dd[1]].elev = dd[6]
            data_dict[dd[1]].station = dd[1]
            tf_dict[dd[1]] = True
        # fill in the impedance tensor with appropriate values
        if dd[7].find('Z') == 0:
            z_err = dd[10]
            if wave_sign_impedance == '+':
                z_value = dd[8] + 1j * dd[9]
            elif wave_sign_impedance == '-':
                z_value = dd[8] - 1j * dd[9]
            else:
                raise DataError("Incorrect wave sign \"{}\" (impedance)".format(wave_sign_impedance))

            if units.lower() == 'ohm':
                z_value *= 796.
                z_err *= 796.
            elif units.lower() not in ("[v/m]/[t]", "[mv/km]/[nt]"):
                raise DataError("Unsupported unit \"{}\"".format(units))

            data_dict[dd[1]].Z.z[p_index, ii, jj] = z_value
            data_dict[dd[1]].Z.z_err[p_index, ii, jj] = z_err
        # fill in tipper with appropriate values
        elif dd[7].find('T') == 0:
            if wave_sign_tipper == '+':
                data_dict[dd[1]].Tipper.tipper[p_index, ii, jj] = dd[8] + 1j * dd[9]
            elif wave_sign_tipper == '-':
                data_dict[dd[1]].Tipper.tipper[p_index, ii, jj] = dd[8] - 1j * dd[9]
            else:
                raise DataError("Incorrect wave sign \"{}\" (tipper)".format(wave_sign_tipper))
            data_dict[dd[1]].Tipper.tipper_err[p_index, ii, jj] = dd[10]

    # make mt_dict an attribute for easier manipulation later
    mt_dict = data_dict

    ns = len(list(mt_dict.keys()))
    nf = len(period_list)
    data_array_zt = []
    for ii, s_key in enumerate(sorted(mt_dict.keys())):
        mt_obj = mt_dict[s_key]
        data_array_zt.append(np.concatenate((mt_obj.Z.z.reshape(-1,4),mt_obj.Tipper.tipper.reshape(-1,2)), axis=1))
    data_array_zt = np.array(data_array_zt).reshape(ns,nf,6)
    return data_array_zt

def read_model_file_Jon(model_fn):
    """
    read an initial file and return the pertinent information including
    grid positions in coordinates relative to the center point (0,0) and
    starting model.. credit of MTpy.
    Note that the way the model file is output, it seems is that the
    blocks are setup as
    ModEM:                           WS:
    ----------                      -----
    0-----> N_north                 0-------->N_east
    |                               |
    |                               |
    V                               V
    N_east                          N_north

    Arguments:
    ----------
        **model_fn** : full path to initializing file.
    Outputs:
    --------
        **nodes_north** : np.array(nx)
                    array of nodes in S --> N direction
        **nodes_east** : np.array(ny)
                    array of nodes in the W --> E direction
        **nodes_z** : np.array(nz)
                    array of nodes in vertical direction positive downwards
        **res_model** : dictionary
                    dictionary of the starting model with keys as layers
        **res_list** : list
                    list of resistivity values in the model
        **title** : string
                     title string
    """
    with open(model_fn, 'r') as ifid:
        ilines = ifid.readlines()

    # get size of dimensions, remembering that x is N-S, y is E-W, z is + down
    nsize = ilines[1].strip().split()
    n_north = int(nsize[0])
    n_east = int(nsize[1])
    n_z = int(nsize[2])
    log_yn = nsize[4]

    # get nodes
    nodes_north = np.array([np.float(nn)
                                 for nn in ilines[2].strip().split()])
    nodes_east = np.array([np.float(nn)
                                for nn in ilines[3].strip().split()])
    nodes_z = np.array([np.float(nn)
                             for nn in ilines[4].strip().split()])

    res_model = np.zeros((n_north, n_east, n_z))

    # get model
    count_z = 0
    line_index = 6
    count_e = 0
    while count_z < n_z:
        iline = ilines[line_index].strip().split()
        # blank lines spit the depth blocks, use those as a marker to
        # set the layer number and start a new block
        if len(iline) == 0:
            count_z += 1
            count_e = 0
            line_index += 1
        # 3D grid model files don't have a space at the end
        # additional condition to account for this.
        elif (len(iline) == 3) & (count_z == n_z):
            count_z += 1
            count_e = 0
            line_index += 1
            #print(iline)
        # each line in the block is a line of N-->S values for an east value
        else:
            north_line = np.array([float(nres) for nres in iline])
            res_model[:, count_e, count_z] = north_line[::-1]
            count_e += 1
            line_index += 1
    res_model = res_model
    return res_model

def gen_chunk(train_data_N,train_mask_all,batch_size=16):
    while True:
        input_batch = [];   mask_batch = []
        for _ in range(batch_size):
            s_idx = np.random.choice(range(train_data_N.shape[0]))
            input_batch += [train_data_N[s_idx]]
            mask_batch += [train_mask_all[s_idx]]
        yield np.stack(input_batch, 0), np.stack(mask_batch, 0)

def build_3D_model(input_size, output_size):
    start_neurons = 32
    dropout = 0.5
    ks = (3, 3, 3)
    kernel_i = 'he_uniform'
    a_func = 'linear'
    input_layer = Input(input_size)
    bn = BatchNormalization()(input_layer)
    cn1 = Conv3D(start_neurons, kernel_size=ks, padding='same', activation=a_func, kernel_initializer=kernel_i)(bn)
    cn2 = Conv3D(start_neurons, kernel_size=ks, padding='same', activation=a_func, kernel_initializer=kernel_i)(cn1)
    bn2 = Activation(a_func)(BatchNormalization()(cn2))
    bn2 = Dropout(dropout)(bn2)
    bn2 = Conv3D(start_neurons, kernel_size=ks, padding='same', activation=a_func, kernel_initializer=kernel_i)(bn2)
    bn2 = Conv3D(start_neurons, kernel_size=(3, 3, 3), padding='same', activation=a_func, kernel_initializer=kernel_i)(bn2)
    dn1 = MaxPool3D((2, 2, 2))(bn2)

    cn3 = Conv3D(start_neurons*2, kernel_size=ks, padding='same', activation=a_func, kernel_initializer=kernel_i)(dn1)
    bn3 = Activation(a_func)(BatchNormalization()(cn3))
    bn3 = Dropout(dropout)(bn3)
    bn3 = Conv3D(start_neurons*2, kernel_size=ks, padding='same', activation=a_func, kernel_initializer=kernel_i)(bn3)
    bn3 = Conv3D(start_neurons*2, kernel_size=(3, 3, 3), padding='same', activation=a_func, kernel_initializer=kernel_i)(bn3)
    dn2 = MaxPool3D((2, 2, 2))(bn3)

    cn4 = Conv3D(start_neurons*4, kernel_size=ks, padding='same', activation=a_func, kernel_initializer=kernel_i)(dn2)
    bn4 = Activation(a_func)(BatchNormalization()(cn4))
    bn4 = Dropout(dropout)(bn4)
    bn4 = Conv3D(start_neurons*4, kernel_size=ks, padding='same', activation=a_func, kernel_initializer=kernel_i)(bn4)
    bn4 = Conv3D(start_neurons*4, kernel_size=(3, 3, 3), padding='same', activation=a_func, kernel_initializer=kernel_i)(bn4)
    dn3 = MaxPool3D((1, 2, 2))(bn4)

    cn5 = Conv3D(start_neurons*8, kernel_size=ks, padding='same', activation=a_func, kernel_initializer=kernel_i)(dn3)
    bn5 = Activation(a_func)(BatchNormalization()(cn5))
    bn5 = Dropout(0.3)(bn5)
    bn5 = Conv3D(start_neurons*8, kernel_size=ks, padding='same', activation=a_func, kernel_initializer=kernel_i)(bn5)

    up1 = Conv3DTranspose(start_neurons*4, kernel_size=ks, strides=(1, 2, 2), padding='same')(bn5)
    cat1 = concatenate([up1, bn4])
    cat1 = Conv3D(start_neurons*4, kernel_size=ks, padding='same', activation=a_func, kernel_initializer=kernel_i)(cat1)
    cat1 = Dropout(dropout)(cat1)
    cat1 = Conv3D(start_neurons*4, kernel_size=ks, padding='same', activation=a_func, kernel_initializer=kernel_i)(cat1)

    up2 = Conv3DTranspose(start_neurons*2, kernel_size=ks, strides=(2, 2, 2), padding='same')(cat1)
    cat2 = concatenate([up2, bn3])
    cat2 = Conv3D(start_neurons*2, kernel_size=ks, padding='same', activation=a_func, kernel_initializer=kernel_i)(cat2)
    cat2 = Dropout(dropout)(cat2)
    cat2 = Conv3D(start_neurons*2, kernel_size=ks, padding='same', activation=a_func, kernel_initializer=kernel_i)(cat2)

    up3 = Conv3DTranspose(start_neurons, kernel_size=ks, strides=(2, 2, 2), padding='same')(cat2)
    cat3 = concatenate([up3, bn2])
    cat3 = Conv3D(start_neurons, kernel_size=ks, padding='same', activation=a_func, kernel_initializer=kernel_i)(cat3)
    cat3 = Dropout(dropout)(cat3)
    cat3 = Conv3D(start_neurons, kernel_size=ks, padding='same', activation=a_func, kernel_initializer=kernel_i)(cat3)
    pre_out = Conv3D(1, kernel_size=(1, 1, 1), padding='same', activation=a_func)(cat3)

    layer_sz = pre_out.shape
    kernel_sz0 = (layer_sz[1]-output_size[0]+1, layer_sz[2]-output_size[1]+1, layer_sz[3]-output_size[2]+1)
    out = Conv3D(1, kernel_size=kernel_sz0, padding='valid', activation=a_func)(pre_out)
    sim_model = Model(inputs=[input_layer], outputs=[out])
    return sim_model

# https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9 .revised by Jon, on 2020-08-25
def _error(y_true, y_pred):
    """ Simple error """
    return y_true - y_pred

def mse(y_true, y_pred):
    """ Mean Squared Error """
    return K.mean(K.square(_error(y_true, y_pred)))

def rmse(y_true, y_pred):
    """ Root Mean Squared Error """
    return K.sqrt(mse(y_true, y_pred))

def nrmse(y_true, y_pred):
    """ Normalized Root Mean Squared Error """
    # normalized by corelation?
    return rmse(y_true, y_pred) / (K.max(y_true) - K.min(y_true))

def rmse_m(y_true, y_pred):
    """ Normalized Root Mean Squared Error plus minimum model """
    return rmse(y_true, 1*y_pred)

def ms_focus(y_true, y_pred):
    beta = 0.05
    return K.sum(K.square(_error(y_true, y_pred))/(K.square(_error(y_true, y_pred)) + beta**2))

def shuffle(X, y, test_size=5):
    ratio = int(X.shape[0]/test_size)
    X_train = X[ratio:,:]
    X_val = X[:ratio,:]
    y_train = y[ratio:,:]
    y_val = y[:ratio,:]
    return X_train, X_val, y_train, y_val

def savgol_smooth_Jon_0(train_data0):
    from scipy.signal import savgol_filter
    iwin = 3;
    iorder = 2;
    tshape = train_data0.shape
    for i in range(tshape[2]):
        for j in range(tshape[3]):
            train_data0[0,:,i,j] = savgol_filter(train_data0[0,:,i,j], iwin, iorder)
    return train_data0

def plot_slices(X,Y,Z,U):
    fig,axs = plt.subplots(1,8,figsize=(17,3),constrained_layout=True)
    for idx in range(8):
        ax1 = axs[idx]
        ax1.pcolormesh(X[:,:,2*idx],Y[:,:,2*idx],U[:,:,2*idx],shading='auto')
        ax1.axis('equal')
        ax1.set_title(str(min(U[:,:,3*idx].reshape(-1)))+'\n'+str(max(U[:,:,3*idx].reshape(-1))))
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------