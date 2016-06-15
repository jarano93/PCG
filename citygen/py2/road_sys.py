#!/usr/bin/python2

import math as m
import random as r
import numpy as np
import numpy.linalg as la

# TODO rework with active, change stop

def get_angle(point):
    angle = m.atan2(point[0], point[1])
    if angle < 0:
        angle += 2 * m.pi
    return angle

def min_angle_difference(a1, a2):
    angles = np.array([a1, a1 + m.pi % 2 * m.pi])
    return min(angles - a2)

class Seed:
    def __init__(self, start, angle, length):
        self.start = np.array(start)
        self.angle = angle # [0, 2pi)
        self.length = length
        self.init_end()

    def init_end(self):
        self.end = self.start + (self.length * np.array([m.sin(self.angle), m.cos(self.angle)]))

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
        end = self.end - seed.start
        start_angle = get_angle(start)
        end_angle = get_angle(end)
        if end_angle < start_angle:
            temp = start_angle
            start_angle = end_angle
            end_angle = temp
        if start_angle <= m.pi / 4 and end_angle >= 5 * m.pi / 4:
            if end_angle <= seed.angle or seed.angle <= start_angle:
                return True
            else:
                return False
        else:
            if start_angle <= seed.angle and seed.angle <= end_angle:
                return True
            else:
                return False

    def intersect_length(self, seed):
        # returns the length a seed needs in order to intersect self
        # only call if seed_nondiverg returns true
        start = self.start - seed.start # (x0, y0)
        end = self.end - seed.start # (x1, y1)
        r0, r1 = start[0], end[0]
        c0, c1 = start[1], end[1]
        r_dif = r1 - r0
        c_dif = c1 - c0
        numer = r0 * c_dif - c0 * r_dif
        return numer / (c_dif * m.sin(seed.angle) - r_dif * m.cos(seed.angle))


# DONE early termination for majors & minors
# DONE handling when maj_possible and min_possible are both empty

# TODO make update rules change while updating

class RoadSys:
    def __init__(self, hmap, dmap, vmap):
        self.hmap = hmap
        self.dmap = dmap
        self.vmap = vmap # 2D array of 1's (valid) and 0's (invalid)
        self.active = []
        self.maj_accepted = []
        self.maj_possible = []
        self.min_accepted = []
        self.min_possible = []
        self.shape = self.hmap.shape

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

    def create_system(self, fName, N_mc, N_maj, N_min, maj_params, min_params):
        self.majors, self.minors = [], []
        self.update_params = maj_params # major rules can change while updating
        self.candidate = self.major_candidate(N_mc)
        self.major_system(N_maj)
        maj_fname = fName + "_maj"
        self.save_svg(maj_fname)
        self.update_rules = min_params # minor rules can change while updating
        print len(self.maj_possible), len(self.min_possible)
        self.candidate = self.minor_candidate()
        self.minor_system(N_min)
        print len(self.maj_accepted), len(self.min_accepted)
        self.save_svg(fName)

    def major_candidate(self, N):
        gauss_flag = self.update_params['major_candidate']['gauss']
        if gauss_flag:
            start_var = self.update_params['major_candidate']['start_variance']
            len_mean = self.update_params['major_candidate']['length_mean']
            len_var = self.update_params['major_candidate']['length_variance']
            start_mean_r = (self.shape[0] - 1) / 2
            start_mean_c = (self.shape[1] - 1) / 2
        else:
            len_fl= self.update_params['major_candidate']['length_floor']
            len_ceil = self.update_params['major_candidate']['length_ceiling']
        while True:
            if gauss_flag:
                best_start = [
                        r.gauss(start_mean_r, start_var),
                        r.gauss(start_mean_c, start_var)
                    ]
                best_len = r.gauss(len_mean, len_var)
            else:
                best_start = [
                        r.uniform(0, self.shape[0] - 1),
                        r.uniform(0, self.shape[1] - 1)
                    ]
                best_len = r.uniform(len_fl, len_ceil)
            best = Seed(best_start, r.uniform(0, 2 * m.pi), best_len)
            if self.valid_local(best):
                best_score = self.test_global(best)
                break
        for n in xrange(N):
            if gauss_flag:
                temp_start = [
                        r.gauss(start_mean_r, start_var),
                        r.gauss(start_mean_c, start_var)
                    ]
                temp_len = r.gauss(len_mean, len_var)
            else:
                temp_start = [
                        r.uniform(0, self.shape[0] - 1),
                        r.uniform(0, self.shape[1] - 1)
                    ]
                temp_len = r.uniform(len_fl, len_ceil)
            temp = Seed(temp_start, r.uniform(0, 2 * m.pi), temp_len)
            if not self.valid_local(temp):
                continue
            temp_score = self.test_global(temp)
            if temp_score > best_score:
                best = temp
                best_score = temp_score
        return [best]

    def major_system(self, N):
        for n in xrange(N):
            print "maj: %d" % n
            self.candidate = self.maj_update()
            # fname_tail = "_major_%i" % n
            # self.debug_svg(fname_tail)
            if len(self.candidate) == 0:
                break
        self.maj_accepted.extend(self.active)
        self.active = []

    def maj_update(self):
        # print len(self.active)
        local_candidates = self.get_locals()
        if len(local_candidates) == 0:
            if len(self.maj_possible) == 0:
                return []
            else:
                return [self.maj_possible.pop(0)]
        best = self.best_global(local_candidates)
        # print best.length, best.angle, best.end
        next_candidate, branches = self.match_maj(best)
        next_possibles = self.trim_maj(next_candidate, branches)
        self.maj_possible.extend(next_possibles)
        # print len(self.maj_possible)
        return next_candidate

    def get_locals(self):
        # get principal candidate from the seed
        locals = []
        candidate = self.candidate[0]
        if self.valid_local(candidate):
            locals.append(candidate)
        # Monte Carlo & test to generate other local candidates
        num_locals, angle_range, len_range = self.local_conditions(candidate)
        for i in xrange(num_locals):
            temp = self.gen_local(angle_range, len_range)
            if self.valid_local(temp):
                locals.append(temp)
        return locals

    def valid_local(self, c):
        # tbh I think I only need to check the ends
        coord_start, coord_end = c.start, c.end
        valid = True
        if coord_start[0] < 0 or coord_start[1] < 0:
            return False
        if coord_end[0] < 0 or coord_end[1] < 0:
            return False
        if coord_start[0] >= self.shape[0] or coord_start[1] >= self.shape[1]:
            return False
        if coord_end[0] >= self.shape[0]  or coord_end[1] >= self.shape[1]:
            return False
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
        num_scale = self.update_params['local']['number_scale']
        angle_fl = self.update_params['local']['angle_floor'] # m.pi / 18
        angle_ceil = self.update_params['local']['angle_ceiling']
        angle_scale = self.update_params['local']['angle_scale']
        len_fl = self.update_params['local']['length_floor']
        len_ceil = self.update_params['local']['length_ceiling']
        len_scale = self.update_params['local']['length_scale']
        """
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

            try:
                num = num_ceil - int((num_ceil - num_fl) * m.exp(-num_scale * num_arg))
            except OverflowError:
                num = (num_fl + num_ceil) / 2
            try:
                angle = angle_fl + ((angle_ceil - angle_fl) / (1 + m.exp(-angle_scale * angle_arg)))
            except OverflowError:
                angle = (angle_fl + angle_ceil) / 2
            length = len_fl + ((len_ceil - len_fl) / (1 + m.exp(-len_scale * len_arg)))
        """
        num = int(r.uniform(num_fl, num_ceil))
        angle = r.uniform(angle_fl, angle_ceil)
        length = r.uniform(len_fl, len_ceil)
        return num, angle, [len_fl, len_ceil]

    def candidate_filter(self, map, c, filter_radius):
        response_start = self.map_filter(map, c.start, filter_radius)
        response_end = self.map_filter(map, c.end, filter_radius)
        return response_start, response_end

    def gen_local(self, angle_range, len_range):
        candidate = self.candidate[0]
        c_angle = candidate.angle + 2 * m.pi # mod 2pi later
        c_len = candidate.length
        rand_angle = r.uniform(c_angle - angle_range, c_angle + angle_range)
        rand_angle = rand_angle % (2 * m.pi)
        # len_floor = c_len - len_range if c_len - len_range > 0 else 0
        rand_len = r.uniform(len_range[0], len_range[1])
        return Seed(candidate.start, rand_angle, rand_len)

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
        start = c.start
        end = c.end
        density = (self.dmap[start[0],start[1]] + self.dmap[end[0],end[1]]) / 2
        height = (self.hmap[start[0],start[1]] + self.hmap[end[0],end[1]]) / 2
        return d_weight * density + h_weight * height

    def match_maj(self, c):
        match_c = c
        c_end = c.end
        c_angle = c.angle
        c_length = c.length
        possibles = []
        next_c = []
        branch_flag = True

        # model params
        p_TOL = self.update_params['match']['point_TOL']
        i_TOL = self.update_params['match']['intersect_TOL']
        m_TOL = self.update_params['match']['merge_TOL']
        act_stop = self.update_params['match']['active_stop']

        for a in self.active:
            if la.norm(c_end - a.start) < p_TOL:
                branch_flag = False
                c.set_end(a.start)
                # snap to start-point
                if min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                    pass
                else: # intersect
                    # next_c= [Seed(a.end, c_angle, c_length)]
                    pass
            elif la.norm(c_end - a.end) < p_TOL:
                branch_flag = False
                c.set_end(a.end)
                #snap to end-point
                if min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                    pass
                else: # intersect
                    # next_c= [Seed(a.end, c_angle, c_length)]
                    pass
            elif a.seed_nondiverg(c):
                # intersect or merge
                i_length = a.intersect_length(c)
                if c_length > i_length or abs(i_length - c_length) < i_TOL:
                    branch_flag = False
                    c.set_length(i_length)
                    if min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                        pass
                    else: # intersect
                        # next_c= [Seed(c_end, c_angle, c_length)]
                        pass
        for a in self.maj_accepted:
            if la.norm(c_end - a.start) < p_TOL:
                branch_flag = False
                c.set_end(a.start)
                # snap to start-point
                if min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                    pass
                else: # intersect
                    # next_c= [Seed(a.end, c_angle, c_length)]
                    pass
            elif la.norm(c_end - a.end) < p_TOL:
                branch_flag = False
                c.set_end(a.end)
                #snap to end-point
                if min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                    pass
                else: # intersect
                    # next_c= [Seed(a.end, c_angle, c_length)]
                    pass
            elif a.seed_nondiverg(c):
                # intersect or merge
                i_length = a.intersect_length(c)
                if c_length > i_length or abs(i_length - c_length) < i_TOL:
                    branch_flag = False
                    c.set_length(i_length)
                    if min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                        pass
                    else: # intersect
                        # next_c= [Seed(c_end, c_angle, c_length)]
                        pass

        self.active.append(c.get_segment())

        if branch_flag:
            next_c, possibles, min_possibles = self.branch(c)
            # if r.random() < float(m.log(len(self.active) + 1e-8)) / act_stop:
                # next_c = []
            self.min_possible.extend(min_possibles)
        if len(next_c) == 0:
            try:
                next_c = [self.maj_possible.pop(0)]
            except IndexError:
                next_c = []
            self.maj_accepted.extend(self.active)
            self.active = []


        return next_c, possibles

    def trim_maj(self, candidate, branches):
        for p in self.maj_possible:
            for c in candidate:
                if self.redundant(c, p):
                    self.maj_possible.remove(p)
            for b in branches:
                if self.redundant(b, p):
                    branches.remove(b)
        return branches

    def trim_min(self, candidate, branches):
        for p in self.min_possible:
            for c in candidate:
                if self.redundant(c, p):
                    self.min_possible.remove(p)
            for b in branches:
                if self.redundant(b, p):
                    branches.remove(b)
        return branches

    def redundant(self, possible0, possible1):
        TOL = self.update_params['redundancy_TOL']
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
        min_branches = []
        candidate = [Seed(c.end, c.angle, c.length)]
        branch_prob, branch_angle = self.branch_conditions(c)
        angle_list1 = np.arange(0, m.pi, branch_angle)[1:]
        angle_list2 = 2 * m.pi - angle_list1
        branch_angle_list = np.concatenate((angle_list1, angle_list2))
        branch_angle_list = (branch_angle_list + c.angle) % (2 * m.pi)
        for angle in branch_angle_list:
            if r.random() < branch_prob:
                branches.append(Seed(c.end, angle, c.length))
            else:
                min_branches.append(Seed(c.end, angle, c.length))
        return candidate, branches, min_branches

    def branch_conditions(self, c):
        # USE THE UPDATE RULES
        # return branch_prob & angle
        hp_weight = self.update_params['branch']['height_weight']['probability']
        dp_weight = self.update_params['branch']['density_weight']['probability']
        ha_weight = self.update_params['branch']['height_weight']['angle']
        da_weight = self.update_params['branch']['density_weight']['angle']

        prob_fl = self.update_params['branch']['probability_floor']
        prob_ceil = self.update_params['branch']['probability_ceiling']
        prob_scale = self.update_params['branch']['probability_scale']
        angle_fl = self.update_params['branch']['angle_floor'] # m.pi/9
        angle_ceil = self.update_params['branch']['angle_ceiling'] # 3 * m.pi / 4
        angle_scale = self.update_params['branch']['angle_scale']

        start = c.start
        end = c.end

        h_dif = abs(self.hmap[c.end[0],c.end[1]] - self.hmap[c.start[0], c.start[1]])
        density = self.dmap[c.end[0], c.end[1]]

        # prob_arg = hp_weight * h_dif + dp_weight * density
        # prob = prob_fl + ((prob_ceil - prob_fl) / (1 + m.exp(-prob_scale * prob_arg)))
        # angle_arg = ha_weight * h_dif + da_weight * density
        # angle = angle_fl + ((angle_ceil - angle_fl) / (1 + m.exp(-angle_scale * angle_arg)))
        prob = r.uniform(prob_fl, prob_ceil)
        angle = r.uniform(angle_fl, angle_ceil)
        return prob, angle

    def minor_candidate(self):
        self.min_possible.extend(self.maj_possible)
        self.maj_possible = []
        return [self.min_possible.pop(0)]

    def minor_system(self, N):
        for n in xrange(N):
            print "min: %d" % n
            self.candidate = self.min_update()
            if len(self.candidate) == 0:
                break
        self.min_accepted.extend(self.active)
        self.active = []

    def min_update(self):
        local_candidates = self.get_locals()
        if len(local_candidates) == 0:
            if len(self.min_possible) == 0:
                return []
            else:
                return [self.min_possible.pop(0)]
        best_candidate = self.best_global(local_candidates)
        # print best_candidate.length
        next_candidate, branches = self.match_min(best_candidate)
        next_possibles = self.trim_min(next_candidate, branches)
        self.min_possible.extend(next_possibles)
        # print len(self.min_possible)
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
        p_TOL = self.update_params['match']['point_TOL']
        i_TOL = self.update_params['match']['intersect_TOL']
        m_TOL = self.update_params['match']['merge_TOL']
        act_stop = self.update_params['match']['active_stop']

        noncross = self.update_params['match']['min_maj_noncross']

        for a in self.active:
            if la.norm(c_end - a.start) < p_TOL:
                branch_flag = False
                c.set_end(a.start)
                # snap to start-point
                if noncross or min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                    pass
                else: # intersect
                    next_c= [Seed(c_end, c_angle, c_length)]
            elif la.norm(c_end - a.end) < p_TOL:
                branch_flag = False
                c.set_end(a.end)
                #snap to end-point
                if noncross or min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                    pass
                else: # intersect
                    next_c= [Seed(c_end, c_angle, c_length)]
            elif a.seed_nondiverg(c):
                # intersect or merge
                i_length = a.intersect_length(c)
                if c_length > i_length or abs(i_length - c_length) < i_TOL:
                    branch_flag = False
                    c.set_length(i_length)
                    if noncross or min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                        pass
                    else: # intersect
                        next_c= [Seed(c_end, c_angle, c_length)]
        for a in self.maj_accepted:
            if la.norm(c_end - a.start) < p_TOL:
                branch_flag = False
                c.set_end(a.start)
                # snap to start-point
                if noncross or min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                    pass
                else: # intersect
                    next_c= [Seed(c_end, c_angle, c_length)]
            elif la.norm(c_end - a.end) < p_TOL:
                branch_flag = False
                c.set_end(a.end)
                #snap to end-point
                if noncross or min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                    pass
                else: # intersect
                    next_c= [Seed(c_end, c_angle, c_length)]
            elif a.seed_nondiverg(c):
                # intersect or merge
                i_length = a.intersect_length(c)
                if c_length > i_length or abs(i_length - c_length) < i_TOL:
                    branch_flag = False
                    c.set_length(i_length)
                    if noncross or min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                        pass
                    else: # intersect
                        next_c= [Seed(c_end, c_angle, c_length)]
        for a in self.min_accepted:
            if la.norm(c_end - a.start) < p_TOL:
                branch_flag = False
                c.set_end(a.start)
                # snap to start-point
                if min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                    pass
                else: # intersect
                    next_c= [Seed(a.end, c_angle, c_length)]
            elif la.norm(c_end - a.end) < p_TOL:
                branch_flag = False
                c.set_end(a.end)
                #snap to end-point
                if min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                    pass
                else: # intersect
                    next_c= [Seed(a.end, c_angle, c_length)]
            elif a.seed_nondiverg(c):
                # intersect or merge
                i_length = a.intersect_length(c)
                if c_length > i_length or abs(i_length - c_length) < i_TOL:
                    branch_flag = False
                    c.set_length(i_length)
                    if min_angle_difference(a.angle, c_angle) < m_TOL: # merge
                        pass
                    else: # intersect
                        next_c= [Seed(c.end, c_angle, c_length)]

        self.active.append(c.get_segment())

        if branch_flag:
            next_c, possibles, _ = self.branch(c)
            if r.random() < float(m.log(len(self.active) + 1e-8)) / act_stop:
                next_c = []
        if len(next_c) == 0:
            try:
                next_c = [self.min_possible.pop(0)]
            except IndexError:
                next_c = []
            self.min_accepted.extend(self.active)
            self.active = []

        return next_c, possibles

    def debug_svg(self, name_tail):
        file_name = "debug_svg/debug" + name_tail + ".svg"
        file = open(file_name, 'w')
        self.prep_svg(file)
        self.write_segments(file)
        self.end_svg(file)
        file.close()

    def save_svg(self, fName):
        file_name = fName + ".svg"
        file = open(file_name, 'w')
        self.prep_svg(file)
        self.write_segments(file)
        self.end_svg(file)
        file.close()
        # print "Saved svg as %s" % fName

    def prep_svg(self, file):
        shape = self.hmap.shape
        width, height = shape[1], shape[0]
        dim_string = '<svg width="%i" height="%i"\n' % (width, height)
        file.write('<?xml version="1.0"?>\n')
        file.write(dim_string)
        file.write('    xmlns="http://www.w3.org/2000/svg">\n\n')

    def write_segments(self, file):
        for a in self.maj_accepted:
            file.write(self.segment_svg(a, 'red', 1))
        for a in self.min_accepted:
            file.write(self.segment_svg(a, 'black', 1))
        for a in self.active:
            file.write(self.segment_svg(a, 'green', 1))

    def segment_svg(self, seg, stroke, width):
        part1 = '<line x1="%d" y1="%d" ' % (int(seg.start[1]), int(seg.start[0]))
        part2 = 'x2="%d" y2="%d" ' % (int(seg.end[1]), int(seg.end[0]))
        part3 = 'stroke="%s" stroke-width="%i"/>\n' % (stroke, width)
        return part1 + part2 + part3

    def end_svg(self, file):
        file.write('\n</svg>')
