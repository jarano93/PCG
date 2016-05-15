import numpy as np
import random as r
import numpy.random as nr

def fault_line(dims, N, base, shift, deviation, shift_decay=0.2, dev_decay=0.9):
    hmap = base * np.ones((dims[0], dims[1]))
    for n in xrange(N):
        reverse_flag = r.getrandbits(1)
        temp = np.zeros
        pt = (r.uniform(0, dims[0]), r.uniform(0, dims[1]))
        dxt = (r.uniform(-1,1), r.uniform(0,1))
        if dxy == (0,0):
            dxy = (0,1)
        if dxy[1] == 0:
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
                        temp[c,r] = 1 if r < pt[0] + dxy[0] * (x - pt[1]) / dxy[1] else 0
                    else:
                        temp[c,r] = 0 if r < pt[0] + dxy[0] * (x - pt[1]) / dxy[1] else 1
        temp *= r.gauss(shift, deviation)
        hmap += temp
        deviation = deviation * dev_decay if deviation > 1 else 1
        shift = shift * shift_decay if shift > 1 else 1
    return hmap
