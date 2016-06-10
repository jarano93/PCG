#!/usr/bin/python2

import numpy.random as nr
import numpy as np
import math as m
from PIL import Image
import topogen.py2.diamond_square as ds
import citygen.py2.rs2 as rs

corners = [100, 120, 140, 100]
hmap = ds.diamond_square(9, corners, False, 5)
print hmap.astype(int)
hmap -= np.amin(hmap)
hmap /= (np.amax(hmap) + 1e-8)
hmap = np.floor(255 * hmap).astype(int)
print hmap
img = Image.fromarray(hmap, 'L')
img.save('dsq.bmp', 'BMP')
