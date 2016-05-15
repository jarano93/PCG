#!/usr/bin/python2
import numpy as np
import numpy.random as nr

def diamond_square(dim_pow, corner_seeds, center_seed=False, deviation=10, decay=0.6):
    len = int(np.power(2, dim_pow))
    half = int(len / 2)

    hmap = np.zero(len + 1 , len + 1)

    # set the corners
    hmap[0,0] = corner_seeds[0] # top left
    hmap[-1,0] = corner_seeds[1] # bottom left
    hmap[0,-1] = corner_seeds[2] # top right
    hmap[-1,-1] = corner_seeds[3] # bottom right

    sq_list = [int(numpy.power(2, p)) for p in xrange(dim_pow-1, -1, -1)]
    for sq in sq_list:
        dsq = 2 * sq

        # calculate/set the center of the heightmap
        for r in xrange(sq, len, double):
            ra, rs = r + sq, r - sq
            for c in xrange(sq, len, double):
                ca, cs = c + sq, c - sq
                if center_seed:
                    hmap[r,c] = center_seed
                    center_seed = False
                else:
                    hmap[r,c] = hmap[rs,cs] + hmap[ra,cs] + hmap[rs,ca] + hmap[ra,ca]
                    hmap[r,c] /= 4
                    hmap[r,c] += nr.normal(0, deviation)

        # calculate the midpoints of edges, ignoring OOB values
        for r in xrange(sq, len, double):
            ra, rs = r + sq, r - sq
            for c in xrange(sq, len, double):
                ca, cs =  c + sq, c - sq

                # top midpoint
                hmap[rs,c] = hmap[r,c] + hmap[rs,ca] + hmap[rs,cs]
                if rs == 0:
                    hmap[rs,c] /= 3
                else:
                    hmap[rs,c] = (hmap[rs,c] + hmap[r-dsq,c]) / 4
                hmap[rs,c] += nr.normal(0,deviation)

                # left midpoint
                hmap[r,cs] = hmap[r,c]
                if cs == 0:
                    hmap[r,cs] /= 3
                else:
                    hmap[r,cs] = (hmap[r,cs] + hmap[r,c-dsq]) / 4
                hmap[r,cs] += nr.normal(0,deviation)

                # bottom midpoint
                hmap[ra,c] = hmap[r,c]
                if rs == length:
                    hmap[ra,c] /= 3
                else:
                    hmap[ra,c] = (hmap[ra,c] + hmap[r+dsq,c]) / 4
                hmap[rs,c] += nr.normal(0,deviation)

                # right midpoint
                hmap[r,ca] = hmap[r,c] + hmap[rs,ca] + hmap[ra,ca]
                if ca == length:
                    hmap[r,ca] /= 3
                else:
                    hmap[r,ca] = (hmap[r,ca] + hmap[r,c+dsq]) / 4
                hmap[r,ca] += nr.normal(0,deviation)

        deviation *= decay

    return hmap
