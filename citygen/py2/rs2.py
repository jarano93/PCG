#!/usr/bin/python2

import math as m
import random as r
import numpy as np
import numpy.linalg as la

def get_angle(point):
    angle = m.atan2(point[0], point[1])
    if angle < 0:
        angle += 2 * m.pi
    return angle

def min_angle_difference(a1, a2):
    angles = np.array([a1, a1 + m.pi % 2 * m.pi])
    return min(angles - a3)

class Seed:
    def __init__(self, start, angle, length):
        self.start = np.array(start)
        self.angle = angle # [0, 2pi)
        self.length = length
        self.init_end()

    def init_end(self):
        self.end = self.start + length * np.array([m.sin(angle), m.cos(angle)])

    def get_segment(self):
        return Segment(self.start, self.end, self.length, self.angle)

    def set_length(self, length):
        self.length = length
        self.init_end()

    def set_end(self, end):
        self.end = end
        dif = end - self.start
        self.length = la.norm(dif)
        self.angle = get_angle(dif)

class Segment:
    # only instantiate via Seed's get_segment
    def __init__(self, start, end, length, angle):
        self.start = np.array(start)
        self.end = np.array(end)
        self.length = length
        self.angle = angle

    def seed_nondiverg(self, seed):
        # return true or false if the angles of seed & segment allow intersection
        start = self.start - seed.start
        end = self.end - seed.end
        start_angle = get_angle(start)
        end_angle = get_angle(end)
        if end_angle < start_angle:
            temp = start_angle
            start_angle = end_angle
            end_angle = start_angle
        if start_angle <= m.pi / 8 and end_angle >= 5 * m.pi / 8:
            if end_angle <= seed.angle or seed.angle <= start_angle:
                return true
            else:
                return false
        else:
            if start_angle <= seed.angle and seed.angle <= end_angle:
                return true
            else:
                return false

    def intersect_length(self, seed):
        # returns the length a seed needs in order to intersect self
        # only call if seed_nondiverg returns true
        start = self.start - seed.start # (x0, y0)
        end = self.end - seed.end # (x1, y1)
        x0, x1 = start[0], end[0]
        y0, y1 = start[1], end[1]
        x_dif = x1 - x0
        y_dif = y1 - y0
        numer = y0 * x_dif - x0 * y_dif
        return numer / (x_dif * m.sin(seed.angle) - y_dif * m.cos(seed.angle))


# TODO early termination for majors & minors
# TODO handling when maj_possible and min_possible are both empty

class RoadSys:
    def __init__(self, hmap, dmap, vmap):
        self.hmap = hmap
        self.dmap = dmap
        self.vmap = vmap # 2D array of 1's (valid) and 0's (invalid)
        self.maj_accepted = []
        self.maj_possible = []
        self.min_accepted = []
        self.min_possible = []

    def map_filter(self, map, coords, filter_radius):
        map_shape = map.shape
        res = []
        row = coords[0]
        col = coords[1]
        for i in xrange(len(filter_radius)):
            f = filter_radius[i]
            row_min = row - f if row - f >= 0 else 0
            row_max = row + f if row + f < map_shape[0] else map_shape[0] - 1
            col_min = col - f if col - f >= 0 else 0
            col_max = col + f if col + f < map_shape[1] else map_shape[1] - 1
            num_elems = (row_max - row_min) * (col_max - col_min)
            sub = map[row_min:row_max, col_min:col_max]
            hit = np.sum(sub) / num_elems
            res.append(hit)
        return res

    def create_system(self, fName, N_mc, N_maj, N_min, maj_rules, min_rules):
        self.majors, self.minors = [], []
        self.candidate = self.major_candidate(N_mc)
        self.majors(N_maj, maj_rules)
        self.candidate = self.minor_candidate()
        self.minors(N_min, min_rules)
        self.save_svg(fName)

    def major_candidate(self, N):
        for n in xrange(N):
            # generate random candidates, pick the best valid one
            pass

    def majors(self, N, major_params):
        self.update_params = major_params # major rules can change while updating
        for n in xrange(N):
            self.candidate = self.maj_update()
            if len(self.candidate) == 0:
                break

    def maj_update(self):
        local_candidates = get_locals()
        if len(local_candidates) == 0:
            if len(self.maj_possible) == 0:
                return []
            else:
                return [self.maj_possible.pop(0)]
        best = best_global(local_candidates)
        next_candidate, branches = self.match_maj(best)
        self.trim_maj(next_candidate, branches)
        self.maj_possible.extend(branches)
        return next_candidate

    def get_locals(self):
        # get principal candidate from the seed
        if self.valid_local(self.candidate):
            locals.append(self.candidate)
        # Monte Carlo & test to generate other local candidates
        locals = []
        num_locals, angle_range, len_range = self.local_conditions(self.candidate.start)
        for i in xrange(num_locals):
            temp = self.gen_local(angle_range, len_range)
            if self.valid_local(temp):
                locals.append(temp)
        return locals

    def valid_local(self, c):
        # tbh I think I only need to check the ends
        valid = True
        coord_start, coord_end = c.start, c.end
        if not self.vmap[coord_start[0],coord_start[1]]:
            valid = False
        if not self.vmap[coord_end[0],coord_end[1]]:
            valid = False
        return valid

    def local_conditions(self, c):
        # use the update rules
        filter_radius = [2, 7, 12]
        hn_weight = self.update_params['local']['height_weight']['num']
        dn_weight = self.update_params['local']['density_weight']['num']
        vn_weight = self.update_params['local']['valid_weight']['num']
        ha_weight = self.update_params['local']['height_weight']['angle']
        da_weight = self.update_params['local']['density_weight']['angle']
        va_weight = self.update_params['local']['valid_weight']['angle']
        hl_weight = self.update_params['local']['height_weight']['length']
        dl_weight = self.update_params['local']['density_weight']['length']
        vl_weight = self.update_params['local']['valid_weight']['length']

        num_fl = self.update_params['local']['number_floor']
        num_ceil = self.update_params['local']['number_ceiling']
        angle_fl = self.update_params['local']['angle_floor'] # m.pi / 18
        angle_ceil = self.update_params['local']['angle_ceiling']
        len_fl = self.update_params['local']['length_floor']
        len_ceil = self.update_params['local']['length_ceiling']

        num_arg = 0
        angle_arg = 0
        len_arg = 0
        h_start, h_end = self.candidate_filter(self.hmap, c, filter_radius)
        d_start, d_end = self.candidate_filter(self.dmap, c, filter_radius)
        v_start, v_end = self.candidate_filter(self.vmap, c, filter_radius)
        for i in xrange(len(filter_radius)):
            num_arg += hn_weight / (abs(h_start[i] - h_end[i]) + 1e-2)
            num_arg += dn_weight * (d_start[i] + d_end[i]) / 2
            num_arg += vn_weight / (v_start[i] + v_end[i])
            angle_arg += ha_weight / (abs(h_start[i] - h_end[i]) + 1e-2)
            angle_arg += da_weight * (d_start[i] + d_end[i]) / 2
            angle_arg += va_weight / (v_start[i] + v_end[i])
            len_arg += hl_weight / (abs(h_start[i] - h_end[i]) + 1e-2)
            len_arg += dl_weight / (d_start[i] + d_end[i])
            len_arg += vl_weight / (v_start[i] + v_end[i])

        num = num_ceil - int((num_ceil - num_fl) * m.exp(-num_arg))
        angle = angle_fl + ((angle_ceil - angle_fl) / (1 + m.exp(-angle_arg)))
        len = len_fl + ((len_ceil - len_fl) / (1 + m.exp(-len_arg)))
        return num, angle, len

    def candidate_filter(self, map, c, filter_radius):
        response_start = self.map_filter(map, c.start, filter_radius)
        response_end = self.map_filter(map, c.end, filter_radius)
        return response_start, response_end

    def gen_local(self, angle_range, len_range):
        c_angle = self.candidate.angle + 2 * m.pi # mod 2pi later
        c_len = self.candidate.length
        rand_angle = r.uniform(c_angle - angle_range, c_angle + angle_range)
        rand_angle = rand_angle % (2 * m.pi)
        len_floor = c_len - len_range if c_len - len_range > 0 else 0
        rand_len = r.uniform(len_floor, c_len + len_range)
        return Seed(self.candidate.start, rand_angle, rand_len)

    def best_global(self, candidates):
        best = candidates[0]
        val = self.test_global(best)
        for c in candidates:
            c_val = self.test_global(c)
            if c_val > val:
                best = c
                val = c_val
        return best

    def test_global(self, c):
        # use the update rules
        d_weight = self.update_params['global']['density_weight']
        h_weight = self.update_params['global']['height_weight']
        # n_weight = self.update_rules['global']['nearest_weight'] DEPRECATED
        start = c.start
        end = c.end
        density = (self.dmap[start[0],start[1]] + self.dmap[end[0],end[1]]) / 2
        height = (self.hmap[start[0],start[1]] + self.hmap[end[0],end[1]]) / 2
        return d_weight * density + h_weight * height

    # DEPRECATED
    def nearest_seg_node(self, c):
        pass

    def match_maj(self, c):
        match_c = c
        c_end = c.end
        c_angle = c.angle
        c_length = c.length
        possibles = []
        next_c = []
        branch_flag = True

        # model params
        p_TOL = self.update_rules['match']['point_TOL']
        i_TOL = self.update_rules['match']['intersect_TOL']
        m_TOL = self.update_rules['match']['merge_TOL']

        for a in self.maj_accepted:
            if la.norm(c_end - a.start) < p_TOL:
                branch_flag = False
                c.set_end(a.start)
                # snap to start-point
                if min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                    next_c= [self.maj_possible.pop(0)]
                else: # intersect
                    next_c= [Seed(c_end, c_angle, c_length)]
            elif la.norm(c_end - a.end) < p_TOL:
                branch_flag = False
                c.set_end(a.end)
                #snap to end-point
                if min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                    next_c= [self.maj_possible.pop(0)]
                else: # intersect
                    next_c= [Seed(c_end, c_angle, c_length)]
            elif a.seed_nondiverg(c):
                # intersect or merge
                i_length = a.intersect_length(c)
                if c_length > i_length or abs(i_length - c_length) < i_TOL:
                    branch_flag = False
                    c.set_length(i_length)
                    if min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                        next_c= [self.maj_possible.pop(0)]
                    else: # intersect
                        next_c= [Seed(c_end, c_angle, c_length)]
                    break

        if branch_flag:
            next_c, possibles = self.branch(c)

        maj_accepted = c.get_segment()

        return next_c, possibles

    def trim_maj(self, c, branches):
        for p in self.maj_possible:
            if self.redundant(c, p):
                self.maj_possible.remove(p)
            for b in branches:
                if self.redundant(b, m):
                    branches.remove(b)
        return branches

    def trim_min(self, c, branches):
        for p in self.min_possibles:
            if self.redundant(c, p):
                self.min_possible.remove(p)
            for b in branches:
                if self.redundant(b, p):
                    self.branches.remove(b)
        return branches

    def redundant(self, possible0, possible1):
        TOL = self.update_rules['redundandcy_TOL']
        start0, end0 = possible0.start, possible0.end
        start1, end1 = possible1.start, possible1.end
        if la.norm(start0 - start1) < TOL and la.norm(end0 - end1) < TOL:
            return True
        elif la.norm(start0 - end1) < TOL and la.norm(end0 - start1) < TOL:
            return True
        else:
            return False

    def branch(self, c):
        # just copy paste this shit in the other one fam
        # r.random for [0,1)
        branches = []
        candidate = [Seed(c.end, c.angle, c.length)]
        branch_prob, branch_angle = self.branch_conditions(c)
        angle_list1 = np.arange(m.pi, branch_angle)[1:]
        angle_list2 = 2 * m.pi - branch_angle_list1
        branch_angle_list = np.concatenate(angle_list1, angle_list2)
        branch_angle_list = (branch_angle_list + c.angle) % (2 * m.pi)
        for angle in branch_angle_list:
            if r.random() < branch_prob:
                branches.append(Seed(c.end, angle, c.length))
        return candidate, branches

    # TODO
    def branch_conditions(self, c):
        # USE THE UPDATE RULES
        # return branch_prob & angle

        hp_weight = self.update_params['branch']['height_weight']['probability']
        dp_weight = self.update_params['branch']['density_weight']['probability']
        ha_weight = self.update_params['branch']['height_weight']['angle']
        da_weight = self.update_params['branch']['density_weight']['angle']

        prob_fl = self.update_params['branch']['probability_floor']
        prob_ceil = self.update_params['branch']['probability_ceiling']
        angle_fl = self.update_params['branch']['angle_floor'] # m.pi/9
        angle_ceil = self.update_params['branch']['angle_ceiling'] # 3 * m.pi / 4

        start = c.start
        end = c.end

        h_dif = abs(self.hmap[c.end[0],c.end[1]] - self.hmap[c.start[0], c.start[1]])
        density = self.dmap[c.end[0], c.end[1]]

        prob_arg = hp_weight * h_dif + dp_weight * density
        prob = prob_fl + ((prob_ceil - prob_fl) / (1 + m.exp(-prob_arg)))
        angle_arg = ha_weight * h_dif + da_weight * density
        angle = angle_fl + ((angle_ceil - angle_fl) / (1 + m.exp(-angle_arg)))
        return prob, angle

    def minor_candidate(self):
        self.candidate = self.min_possible.pop(0)
        self.maj_possible = []
        # on second thought don't do the following
        # pop_val = r.randrange(len(self.min_possible))
        # self.canididate = self.min_possible.pop(pop_val)

    def minors(self, N, minor_rules):
        self.update_rules = minor_rules # minor rules can change while updating
        for n in xrange(N):
            self.candidate = self.min_update()
            if len(self.candidate) == 0:
                break

    def min_update(self):
        local_candidates = get_locals()
        if len(local_candidates) == 0:
            if len(self.min_possible) == 0:
                return []
            else:
                return [self.min_possible.pop(0)]
        best_candidate = best_global(local_candidates)
        match_best, next_candidate, branches = self.match_all(best)
        self.trim_min(next_candidate, branches)
        self.min_possible.extend(next_possibles)
        return next_candidate

    def match_min(self, c):
        match_c = c
        c_end = c.end
        c_angle = c.angle
        c_length = c.length
        possibles = []
        next_c = []
        branch_flag = True

        # model params
        p_TOL = self.update_rules['match']['point_TOL']
        i_TOL = self.update_rules['match']['intersect_TOL']
        m_TOL = self.update_rules['match']['merge_TOL']

        noncross = self.update_rules['match']['min_maj_noncross']

        for a in self.maj_accepted:
            if la.norm(c_end - a.start) < p_TOL:
                branch_flag = False
                c.set_end(a.start)
                # snap to start-point
                if noncross or min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                    next_c= [self.maj_possible.pop(0)]
                else: # intersect
                    next_c= [Seed(c_end, c_angle, c_length)]
            elif la.norm(c_end - a.end) < p_TOL:
                branch_flag = False
                c.set_end(a.end)
                #snap to end-point
                if noncross or min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                    next_c= [self.maj_possible.pop(0)]
                else: # intersect
                    next_c= [Seed(c_end, c_angle, c_length)]
            elif a.seed_nondiverg(c):
                # intersect or merge
                i_length = a.intersect_length(c)
                if c_length > i_length or abs(i_length - c_length) < i_TOL:
                    branch_flag = False
                    c.set_length(i_length)
                    if noncross or min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                        next_c= [self.maj_possible.pop(0)]
                    else: # intersect
                        next_c= [Seed(c_end, c_angle, c_length)]
                    break
        for a in self.min_accepted:
            if la.norm(c_end - a.start) < p_TOL:
                branch_flag = False
                c.set_end(a.start)
                # snap to start-point
                if min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                    next_c= [self.maj_possible.pop(0)]
                else: # intersect
                    next_c= [Seed(c_end, c_angle, c_length)]
            elif la.norm(c_end - a.end) < p_TOL:
                branch_flag = False
                c.set_end(a.end)
                #snap to end-point
                if min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                    next_c= [self.maj_possible.pop(0)]
                else: # intersect
                    next_c= [Seed(c_end, c_angle, c_length)]
            elif a.seed_nondiverg(c):
                # intersect or merge
                i_length = a.intersect_length(c)
                if c_length > i_length or abs(i_length - c_length) < i_TOL:
                    branch_flag = False
                    c.set_length(i_length)
                    if min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                        next_c= [self.maj_possible.pop(0)]
                    else: # intersect
                        next_c= [Seed(c_end, c_angle, c_length)]
                    break

        if branch_flag:
            next_c, possibles = self.branch(c)

        maj_accepted = c.get_segment()
        return next_c, possibles

    # TODO
    def save_svg(self, fName):
        file = open(fName, 'w')
        self.prep_svg(file)
        self.write_segments(file)
        self.end_svg(file)
        file.close()
        print "Saved svg as %s" % fName

    def prep_svg(self, file):
        shape = self.hmap.shape
        width, height = shape[1], shape[0]
        dim_string = '<svg width="%i" height="%i"\n' % (width, height)
        file.write('<?xml version="1,0"?>\n')
        file.write(dim_string)
        file.write('    xmlns="http://www,w3,org/2000/svg">\n\n'

    def write_accepted(self, file):
        for a in self.maj_accepted:
            file.write(self.segment_svg(a, black, 2))
        for a in self.min_accepted:
            file.write(self.segment_svg(a, black, 2))

    def segment_svg(self, seg, stroke, width):
        part1 = '<line x1="%f" y1="%f" ' % (seg.start[1], seg.start[0])
        part2 = 'x2="%f" y2="%f" ' % (seg.end[1], seg.end[0])
        part3 = 'stroke="%s" stroke-width="%i"/>\n' % (stroke, width)
        return part1 + part2 + part3

    def end_svg(self, file):
        file.write('\n</svg>')
