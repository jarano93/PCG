import numpy as np
import numpy.random as nr
import random as ra
import math as m

# REDO
def fault_line(dims, N, base, shift, deviation, shift_decay=0.2, dev_decay=0.9):
    hmap = base * np.ones(dims)
    for n in xrange(N):
        reverse_flag = ra.getrandbits(1)
        temp = np.zeros(dims)
        pt = (ra.uniform(0, dims[0]), ra.uniform(0, dims[1]))
        angle = ra.uniform(0, m.pi)
        dr = m.sin(angle)
        dc = m.cos(angle)
        if angle == m.pi / 2:
            for r in xrange(dims[0]):
                for c in xrange(dims[1]):
                    if reverse_flag:
                        temp[c,r] = 1 if c < pt(1) else 0
                    else:
                        temp[c,r] = 0 if c < pt(1) else 1
        else:
            for r in xrange(dims[0]):
                for c in xrange(dims[1]):
                    if reverse_flag:
                        temp[r,c] = 1 if r < pt[0] + dr * (c - pt[1]) / dc else 0
                    else:
                        temp[r,c] = 0 if r < pt[0] + dr * (c - pt[1]) / dc else 1
        temp *= ra.gauss(shift, deviation)
        hmap += temp
        deviation = deviation * dev_decay if deviation > 1 else 1
        shift = shift * shift_decay if shift > 1 else 1
    return hmap

def get_col(row, point, angle, dims):
    col = 0
    if angle == m.pi / 2:
        col = point[1]
    else:
        col = m.tan(angle) * (row - point[0]) + point[1]
        if col < 0:
            col = 0
        elif col > dims[1]:
            col = dims[1]
    return int(np.around(col))

# it really is a lot faster
def fast_fault(dims, N, base, shift, deviation, shift_decay=0.2, dev_decay=0.9):
    hmap = base * np.ones(dims)
    for n in xrange(N):
        side_flag = ra.getrandbits(1)
        # temp = deviation * nr.randn(dims[0], dims[1]) + shift
        temp = ra.gauss(shift, deviation) * np.ones(dims)
        pt = (ra.uniform(0, dims[0]), ra.uniform(0, dims[1]))
        angle = ra.uniform(0, m.pi)
        for r in xrange(dims[0]):
            c = get_col(r, pt, angle, dims)
            if side_flag:
                hmap[r,0:c] += temp[r,0:c]
            else:
                hmap[r,c:] += temp[r,c:]
        deviation = deviation * dev_decay if deviation > 1 else 1
        shift = shift * shift_decay if shift > 1 else 1
    return hmap
