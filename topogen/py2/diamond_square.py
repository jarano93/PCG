#!/usr/bin/python2
import numpy as np
import numpy.random as nr

def diamond_square(dim_pow, corner_seeds, deviation=10, decay=0.6):
    len = int(np.power(2, dim_pow))
    dim = len + 1
    half = int(len / 2)

    hmap = np.zeros((dim , dim))

    # set the corners
    hmap[0,0] = corner_seeds[0] # top left
    hmap[0,-1] = corner_seeds[1] # top right
    hmap[-1,0] = corner_seeds[2] # bottom left
    hmap[-1,-1] = corner_seeds[3] # bottom right

    sq_list = [int(np.power(2, p)) for p in xrange(dim_pow-1, -1, -1)]
    for sq in sq_list:
        dsq = 2 * sq

        # calculate/set the center of the heightmap
        for r in xrange(sq, dim, dsq):
            ra, rs = r + sq, r - sq
            for c in xrange(sq, dim, dsq):
                ca, cs = c + sq, c - sq
                hmap[r,c] = hmap[rs,cs] + hmap[ra,cs] + hmap[rs,ca] + hmap[ra,ca]
                hmap[r,c] /= 4
                hmap[r,c] += nr.normal(0, deviation)

        # calculate the midpoints of edges, ignoring OOB values
        for r in xrange(sq, dim, dsq):
            ra, rs = r + sq, r - sq
            for c in xrange(sq, dim, dsq):
                ca, cs =  c + sq, c - sq

                # top midpoint
                hmap[rs,c] = hmap[r,c] + hmap[rs,ca] + hmap[rs,cs]
                if rs == 0:
                    hmap[rs,c] /= 3
                else:
                    hmap[rs,c] = (hmap[rs,c] + hmap[r-dsq,c]) / 4
                hmap[rs,c] += nr.normal(0,deviation)

                # left midpoint
                hmap[r,cs] = hmap[r,c] + hmap[rs,cs] + hmap[ra,cs]
                if cs == 0:
                    hmap[r,cs] /= 3
                else:
                    hmap[r,cs] = (hmap[r,cs] + hmap[r,c-dsq]) / 4
                hmap[r,cs] += nr.normal(0,deviation)

                # bottom midpoint
                hmap[ra,c] = hmap[r,c] + hmap[ra,cs] + hmap[ra,ca]
                if ra == len:
                    hmap[ra,c] /= 3
                else:
                    hmap[ra,c] = (hmap[ra,c] + hmap[r+dsq,c]) / 4
                hmap[ra,c] += nr.normal(0,deviation)

                # right midpoint
                hmap[r,ca] = hmap[r,c] + hmap[rs,ca] + hmap[ra,ca]
                if ca == len:
                    hmap[r,ca] /= 3
                else:
                    hmap[r,ca] = (hmap[r,ca] + hmap[r,c+dsq]) / 4
                hmap[r,ca] += nr.normal(0,deviation)
        deviation *= decay
    return hmap

def diamond_square_center(dim_pow, corner_seeds, center_seed, deviation=10, decay=0.6):
    len = int(np.power(2, dim_pow))
    dim = len + 1
    half = int(len / 2)

    hmap = np.zeros((dim , dim))

    # set the corners
    hmap[0,0] = corner_seeds[0] # top left
    hmap[0,-1] = corner_seeds[1] # top right
    hmap[-1,0] = corner_seeds[2] # bottom left
    hmap[-1,-1] = corner_seeds[3] # bottom right

    sq_list = [int(np.power(2, p)) for p in xrange(dim_pow-1, -1, -1)]

    center_flag = True
    for sq in sq_list:
        dsq = 2 * sq

        # calculate/set the center of the heightmap
        for r in xrange(sq, dim, dsq):
            ra, rs = r + sq, r - sq
            for c in xrange(sq, dim, dsq):
                ca, cs = c + sq, c - sq
                if center_flag:
                    hmap[r,c] = center_seed
                    center_flag = False
                else:
                    hmap[r,c] = hmap[rs,cs] + hmap[ra,cs] + hmap[rs,ca] + hmap[ra,ca]
                    hmap[r,c] /= 4
                    hmap[r,c] += nr.normal(0, deviation)

        # calculate the midpoints of edges, ignoring OOB values
        for r in xrange(sq, dim, dsq):
            ra, rs = r + sq, r - sq
            for c in xrange(sq, dim, dsq):
                ca, cs =  c + sq, c - sq

                # top midpoint
                hmap[rs,c] = hmap[r,c] + hmap[rs,ca] + hmap[rs,cs]
                if rs == 0:
                    hmap[rs,c] /= 3
                else:
                    hmap[rs,c] = (hmap[rs,c] + hmap[r-dsq,c]) / 4
                hmap[rs,c] += nr.normal(0,deviation)

                # left midpoint
                hmap[r,cs] = hmap[r,c] + hmap[rs,cs] + hmap[ra,cs]
                if cs == 0:
                    hmap[r,cs] /= 3
                else:
                    hmap[r,cs] = (hmap[r,cs] + hmap[r,c-dsq]) / 4
                hmap[r,cs] += nr.normal(0,deviation)

                # bottom midpoint
                hmap[ra,c] = hmap[r,c] + hmap[ra,cs] + hmap[ra,ca]
                if ra == len:
                    hmap[ra,c] /= 3
                else:
                    hmap[ra,c] = (hmap[ra,c] + hmap[r+dsq,c]) / 4
                hmap[ra,c] += nr.normal(0,deviation)

                # right midpoint
                hmap[r,ca] = hmap[r,c] + hmap[rs,ca] + hmap[ra,ca]
                if ca == len:
                    hmap[r,ca] /= 3
                else:
                    hmap[r,ca] = (hmap[r,ca] + hmap[r,c+dsq]) / 4
                hmap[r,ca] += nr.normal(0,deviation)
        deviation *= decay
    return hmap
