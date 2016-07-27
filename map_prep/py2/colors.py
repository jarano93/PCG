#!/usr/bin/python2

from copy import deepcopy
import numpy as np
import math as m

def map_topo(hmap):
    dims = hmap.shape
    print hmap.shape
    if np.amin(hmap) < 0:
        print "water"
        rgb = make_green_blue(hmap, dims)
    else:
        print "land"
        rgb = make_green(hmap, dims)
    return rgb

def make_green(hmap, shape):
    height = shape[0]
    width = shape[1]
    rgb = np.zeros((height,width,3))
    hmap_max = np.amax(hmap)
    hmap_min = np.amin(hmap)
    hmap_range = hmap_max - hmap_min
    print hmap_max, hmap_min, hmap_range
    rgb[...,0] = 200.0 * (hmap - hmap_min) / (hmap_range + 1e-10)
    rgb[...,1] = 255.0
    rgb[...,2] = 100.0 + 100 * (hmap - hmap_min) / (hmap_range + 1e-10)
    return rgb


def make_green_blue(hmap, shape):
    height = shape[0]
    width = shape[1]
    rgb = np.zeros((height,width,3))
    red_band = np.zeros((height, width))
    green_band = np.zeros((height, width))
    blue_band = np.zeros((height, width))
    water_min = np.amin(hmap)
    land_max = np.amax(hmap)
    water_range = 0 - water_min
    land_range = land_max
    # red_band[land] = 200.0 * (hmap) / (land_range + 1e-10)
    # green_band[hmap >= 0] = 255.0
    # blue_band[hmap >= 0] = 100.0 + 100 * (hmap) / (land_range + 1e-10)
    # red_band[hmap < 0] = 50
    # green_band[hmap < 0] = 200 * (hmap) / (water_range + 1e-10)
    # blue_band[hmap < 0] = 255
    # rgb[...,0 and hmap >= 0] = 200.0 * (hmap) / (land_range + 1e-10)
    # rgb[...,1 and hmap >= 0] = 255.0
    # rgb[...,2 and hmap >= 0] = 100.0 + 100 * (hmap) / (land_range + 1e-10)
    # rgb[...,0 and hmap < 0] = 50
    # rgb[...,1 and hmap < 0] = 200 * (hmap) / (water_range + 1e-10)
    # rgb[...,2 and hmap < 0] = 255
    for h in xrange(height - 1):
        for w in xrange(width - 1):
            if hmap[h,w] >= 0:
                rgb[h,w,0] = 200.0 * (hmap[h,w]) / (land_range + 1e-10)
                rgb[h,w,1] = 255.0
                rgb[h,w,2] = 100.0 + 100 * (hmap[h,w]) / (land_range + 1e-10)
            else:
                rgb[h,w,0] = 150 + 125 * (hmap[h,w] / (water_range + 1e-10))
                rgb[h,w,1] = 200 + 100 * (hmap[h,w]) / (water_range + 1e-10)
                rgb[h,w,2] = 255
    # rgb[...,0] = red_band
    # rgb[...,1] = green_band
    # rgb[...,2] = blue_band
    return rgb
