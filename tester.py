#!/usr/bin/python2

from copy import deepcopy
import scipy.misc as sm
import numpy as np
import math as m
import topogen.py2.diamond_square as ds
import topogen.py2.fault_line as fl
import citygen.py2.rs2 as rs


if False:
    corners = [100, 120, 140, 100]
    hmap = ds.diamond_square(10, corners, 20, 0.8)
    sm.imsave('dsq.bmp', hmap)
if False:
    hmap = fl.fast_fault((800,800), 3000, 1, 0.1, 0.01)
    sm.imsave('ff.bmp', hmap)
h_corners = [80, 100, 130, 100]
d_corners = [100, 140, 100, 100]
v_corners = [-50, 50, 50, 50]
hmap = ds.diamond_square(10, h_corners, 20, 0.8)
dmap = ds.diamond_square(10, d_corners, 4, 0.6)
# vmap = ds.diamond_square(9, v_corners, 4, 0.2)
# hmap = np.ones((500, 500))
# dmap = np.ones_like(hmap)
vmap = np.ones_like(hmap)
sm.imsave('hmap.bmp', hmap)
sm.imsave('dmap.bmp', dmap)

mc_params = {
        'N': 100,
        'candidates': 3,
        }

major_params = {
        'major_candidate': {
            'gauss': True,
            'start_variance': 40,
            'length_floor': 5,
            'length_ceiling': 10
            },
        'local': {
            'height_weight': {
                'num' : 10,
                'angle': 120,
                'length': 30
                },
            'density_weight': {
                'num': 20,
                'angle': 140,
                'length': 40
                },
            'valid_weight': {
                'num': 10,
                'angle': 10,
                'length': 10
                },
            'number_floor': 40,
            'number_ceiling': 60,
            'number_scale': 1e-9,
            'angle_floor': m.pi / 18,
            'angle_ceiling': m.pi / 4,
            'angle_scale': 1e-5,
            'length_floor': 8,
            'length_ceiling': 10,
            'length_scale': 1e-5,
            },
        'global': {
            'density_weight': 20,
            'height_weight': 40,
            },
        'match': {
            'point_TOL': 1,
            'intersect_TOL': 1,
            'merge_TOL': m.pi / 6,
            'candidate_stop': 4e2,
            },
        'redundancy_TOL': 7,
        'branch': {
            'height_weight': {
                'probability': 40,
                'angle': 60
                },
            'density_weight': {
                'probability': 50,
                'angle': 20
                },
            'probability_floor': 0,
            'probability_ceiling': 1e-1,
            'probability_scale': 1e-8,
            'angle_floor': m.pi / 3,
            'angle_ceiling': (m.pi / 2) + (m.pi / 18),
            'angle_scale': 1e-8
            }
        }

minor_params = deepcopy(major_params)
minor_params['local']['length_floor'] = 5
minor_params['local']['length_ceiling'] = 10
minor_params['local']['angle_ceiling'] = m.pi / 9
minor_params['global']['height_weight'] = 0
minor_params['global']['density_weight'] = 1
minor_params['match']['min_maj_noncross'] = True
minor_params['match']['candidate_stop'] = 1e3
minor_params['branch']['probability_floor'] = 0.9
minor_params['branch']['probability_ceiling'] = 1
minor_params['branch']['angle_floor'] = (m.pi / 2) - (m.pi / 18)
minor_params['branch']['angle_ceiling'] = (m.pi / 2) + (m.pi / 18)
minor_params['redundancy_TOL'] = 7

sys = rs.RoadSys(hmap, dmap, vmap)
sys.create_system("out", mc_params, 100, 50, major_params, minor_params)
