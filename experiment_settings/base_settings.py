import os
from math import sqrt
from easydict import EasyDict as edict
import platform
import inflection


def calculate_in_ch(converse_gray, detect_edge=False):
    in_ch = 1 if converse_gray else 3
    in_ch = in_ch+1 if detect_edge else in_ch
    return in_ch


# immortal params
local_os_name = 'Darwin'
data_root_path = './data' if platform.system()==local_os_name else '/data/'

# mult_dir's key is module name
mult_dir = {'r_a_m': 8,
            'r_a_m_cnn': 8}

augmentation_params = {
                       'scale':[0.5, 0.75, 1.25],
                       'ratio':[sqrt(1/2), 1, sqrt(2)],  # 1/sqrt(2)だと2ratioが2倍, 逆だと0.5倍
                       'lr_shift':[-64, -32, -16, 16, 32, 64],
                       'ud_shift':[-64, -32, -16, 16, 32, 64],
                       'rotation_angle': list(range(5,360,5))
                      }

net_dir = {net_name:{'module_name':net_name, \
                    'class_name':inflection.camelize(net_name)} \
                        for net_name, _ in mult_dir.items()}

image_normalize_types_dir = {'ZCA': {'method':'zca_whitening', 'opts':{'eps':1e-5}},
                             'LCN': {'method':'local_contrast_normalization', 'opts':None},
                             'GCN': {'method':'global_contrast_normalization', 'opts':None}
                            }

dic_name = 'sysm_pathological.dict'

trainig_params = {
        'optimizer': 'RMSpropGraves',
        'lr': 1e-5,
        'batch_size': 20,
        'epoch': 200,
        'decay_factor': 0.05,  # as lr time decay
        'decay_epoch': 50,
        'snapshot_epoch': 5,
        'report_epoch': 1,
        'weight_decay': True,
        'lasso': False,
        'clip_grad': False,
        'weight_decay': 0.0005,
        'clip_value': 5.,
        'iter_type': 'multi',
    }


def get_base_params():
    base_params = {
                    'local_os_name': local_os_name,
                    'data_root_path': data_root_path,
                    'mult_dir': mult_dir,
                    'augmentation_params': augmentation_params,
                    'net_dir': net_dir,
                    'image_normalize_types_dir': image_normalize_types_dir,
                    'trainig_params': trainig_params,
                }
    return edict(base_params)
