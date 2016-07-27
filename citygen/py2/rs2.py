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
        denom =  c_dif * m.sin(seed.angle) - r_dif * m.cos(seed.angle)
        if denom < 1e-10:
            length = -1
        else:
            length = numer / denom
        return length


class RoadSys:
    def __init__(self, hmap, dmap, vmap):
        self.hmap = hmap
        self.dmap = dmap
        self.vmap = vmap # 2D array of 1's (valid) and 0's (invalid)
        self.candidates = []
        self.maj_accepted = []
        self.min_accepted = []
        self.min_possible = []
        self.update_params = {}
        self.shape = self.hmap.shape

    def create_system(self, fName, mc_params, N_maj, N_min, maj_params, min_params):
        self.majors, self.minors = [], []
        self.update_params = maj_params # major rules can change while updating
        self.candidates = self.major_candidates(mc_params)
        self.major_system(N_maj)
        maj_fName = fName + "_maj"
        self.debug_svg(maj_fName)
        self.save_svg(maj_fName)
        self.update_params = min_params # minor rules can change while updating
        self.candidates = self.minor_candidates()
        self.minor_system(N_min)
        # self.debug_svg(fName)
        self.save_svg(fName)

    def create_system_img(self, fName, imName, mc_params, N_maj, N_min, maj_params, min_params):
        self.majors, self.minors = [], []
        self.update_params = maj_params # major rules can change while updating
        self.candidates = self.major_candidates(mc_params)
        self.major_system(N_maj)
        # maj_fName = fName + "_maj"
        self.debug_svg(maj_fName)
        self.save_svg(maj_fName)
        self.update_params = min_params # minor rules can change while updating
        self.candidates = self.minor_candidates()
        self.minor_system(N_min)
        # self.debug_svg(fName)
        self.save_svg_img(fName, imName)

    def major_candidates(self, mc_params):
        print "major candidates"
        candidates = {}
        N = mc_params['N']
        num_candidates = mc_params['candidates']
        gauss_flag = self.update_params['major_candidate']['gauss']
        len_fl= self.update_params['major_candidate']['length_floor']
        len_ceil = self.update_params['major_candidate']['length_ceiling']
        if gauss_flag:
            start_var = self.update_params['major_candidate']['start_variance']
            start_mean_r = (self.shape[0] - 1) / 2
            start_mean_c = (self.shape[1] - 1) / 2
        while True:
            if gauss_flag:
                best_start = [
                        r.gauss(start_mean_r, start_var),
                        r.gauss(start_mean_c, start_var)
                    ]
            else:
                best_start = [
                        r.uniform(0, self.shape[0] - 1),
                        r.uniform(0, self.shape[1] - 1)
                    ]
            best_len = r.uniform(len_fl, len_ceil)
            c = Seed(best_start, r.uniform(0, 2 * m.pi), best_len)
            if self.valid_local(c):
                score = self.test_global(c)
                candidates[score] = c
                if len(candidates) == num_candidates:
                    break
        min_score = min(candidates)
        for n in xrange(N):
            print "can: %d" % n
            if gauss_flag:
                temp_start = [
                        r.gauss(start_mean_r, start_var),
                        r.gauss(start_mean_c, start_var)
                    ]
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
            if temp_score > min_score:
                candidates[temp_score] = temp
                del candidates[min(candidates)]
                min_score = min(candidates)
        return candidates.values()

    def major_system(self, N):
        for n in xrange(N):
            print "maj: %d" % n
            self.candidates = self.maj_update()
            if len(self.candidates) == 0:
                break

    def maj_update(self):
        next_candidates = []
        for c in self.candidates:
            local_candidates = self.get_locals(c)
            if len(local_candidates) == 0:
                continue
            best = self.best_global(local_candidates)
            possibles = self.match_maj(best)
            next_candidates.extend(possibles)
        return self.trim(next_candidates)

    def get_locals(self, c):
        locals = []
        # get principal candidate from the seed if valid
        if self.valid_local(c):
            locals.append(c)
        # Monte Carlo & test to generate other local candidates
        num_locals, angle_range, len_range = self.local_conditions(c)
        for i in xrange(num_locals):
            temp = self.gen_local(c, angle_range, len_range)
            if self.valid_local(temp):
                locals.append(temp)
        return locals

    def valid_local(self, c):
        # tbh I think I only need to check the ends
        coord_start, coord_end = c.start, c.end
        if coord_start[0] < 0 or coord_start[1] < 0:
            return False
        elif coord_end[0] < 0 or coord_end[1] < 0:
            return False
        elif coord_start[0] >= self.shape[0] or coord_start[1] >= self.shape[1]:
            return False
        elif coord_end[0] >= self.shape[0]  or coord_end[1] >= self.shape[1]:
            return False
        elif self.vmap[int(coord_start[0]), int(coord_start[1])] <= 0:
            return False
        elif self.vmap[int(coord_end[0]), int(coord_end[1])] <= 0:
            return False
        return True

    def local_conditions(self, c):
        # use the update rules
        """
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
        num_fl = self.update_params['local']['number_floor']
        num_ceil = self.update_params['local']['number_ceiling']
        angle_fl = self.update_params['local']['angle_floor'] # m.pi / 18
        angle_ceil = self.update_params['local']['angle_ceiling']
        len_fl = self.update_params['local']['length_floor']
        len_ceil = self.update_params['local']['length_ceiling']

        num = int(r.uniform(num_fl, num_ceil))
        angle = r.uniform(angle_fl, angle_ceil)
        # length = r.uniform(len_fl, len_ceil)
        return num, angle, [len_fl, len_ceil]

    def gen_local(self, c, angle_ceil, len_range):
        c_start = c.start
        c_angle = c.angle + 2 * m.pi # mod 2pi later
        c_len = c.length
        rand_angle = r.uniform(c_angle - angle_ceil, c_angle + angle_ceil)
        rand_angle = rand_angle % (2 * m.pi)
        # len_floor = c_len - len_range if c_len - len_range > 0 else 0
        rand_len = r.uniform(len_range[0], len_range[1])
        return Seed(c_start, rand_angle, rand_len)

    def best_global(self, local_candidates):
        best = local_candidates[0]
        val = self.test_global(best)
        for c in local_candidates:
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
        density = (self.dmap[int(start[0]),int(start[1])] + self.dmap[int(end[0]),int(end[1])]) / 2
        height = (self.hmap[int(start[0]),int(start[1])] + self.hmap[int(end[0]),int(end[1])]) / 2

        score = d_weight * density + h_weight * height
        return score

    def match_maj(self, c):
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
        c_stop = self.update_params['match']['candidate_stop']

        for a in self.maj_accepted:
            if la.norm(c_end - a.start) < p_TOL:
                branch_flag = False
                c.set_end(a.start)
                # snap to start-point
                if min_angle_difference(a.angle, c_angle) > m_TOL:
                    next_c= [Seed(c.end, c.angle, c_length)]
                break
            elif la.norm(c_end - a.end) < p_TOL:
                branch_flag = False
                c.set_end(a.end)
                #snap to end-point
                if min_angle_difference(a.angle, c_angle) > m_TOL:
                    next_c= [Seed(c.end, c.angle, c_length)]
                break
            elif a.seed_nondiverg(c):
                # intersect or merge
                i_length = a.intersect_length(c)
                if c_length > i_length or abs(i_length - c_length) < i_TOL:
                    branch_flag = False
                    c.set_length(i_length)
                    if min_angle_difference(a.angle, c_angle) > m_TOL: # merge
                        next_c= [Seed(c.end, c.angle, c_length)]
                break

        self.maj_add(c)

        if branch_flag:
            next_c, branches, min_branches = self.branch(c)
            self.min_possible.extend(min_branches)
            if r.random() < float(len(self.candidates)) / c_stop:
                self.min_possible.extend(next_c)
                self.min_possible.extend(branches) # delete if unsatisfactory
                next_c = []
                branches = [] # delete if unsatisfactory
            next_c.extend(branches)

        return next_c

    def maj_add(self, c):
        accept = True
        if c.length < 1e-1:
            accept = False
        for a in self.maj_accepted:
            if self.redundant(c, a):
                accept = False
        if accept:
            self.maj_accepted.append(c.get_segment())

    def trim(self, candidates):
        for c0 in candidates:
            remove_flag = False
            for c1 in candidates:
                if c0 != c1 and self.redundant(c0, c1):
                    candidates.remove(c1)
        for c in candidates:
            for a in self.maj_accepted:
                if self.redundant(c, a):
                    candidates.remove(c)
                    remove_flag = True
                    break
            if remove_flag:
                continue
            for a in self.min_accepted:
                if self.redundant(c, a):
                    candidates.remove(c)
                    remove_flag = True
                    break
            if remove_flag:
                continue
        return candidates

    def redundant(self, c0, c1):
        TOL = self.update_params['redundancy_TOL']
        start0, end0 = c0.start, c0.end
        start1, end1 = c1.start, c1.end
        if la.norm(start0 - start1) < TOL and la.norm(end0 - end1) < TOL:
            return True
        elif la.norm(start0 - end1) < TOL and la.norm(end0 - start1) < TOL:
            return True
        else:
            return False

    def branch(self, c):
        branches = []
        min_branches = []
        candidate = [Seed(c.end, c.angle, c.length)]
        branch_prob, branch_angle = self.branch_conditions(c)
        angle_list1 = np.arange(0, m.pi, branch_angle)[1:]
        angle_list2 = 2 * m.pi - angle_list1
        branch_angle_list = np.concatenate((angle_list1, angle_list2))
        branch_angle_list = (branch_angle_list + c.angle) % (2 * m.pi)
        for angle in branch_angle_list:
            temp = Seed(c.end, angle, c.length)
            if r.random() < branch_prob:
                branches.append(temp)
            else:
                min_branches.append(temp)
        return candidate, branches, min_branches

    def branch_conditions(self, c):
        # USE THE UPDATE RULES
        # return branch_prob & angle
        # hp_weight = self.update_params['branch']['height_weight']['probability']
        # dp_weight = self.update_params['branch']['density_weight']['probability']
        # ha_weight = self.update_params['branch']['height_weight']['angle']
        # da_weight = self.update_params['branch']['density_weight']['angle']
        prob_fl = self.update_params['branch']['probability_floor']
        prob_ceil = self.update_params['branch']['probability_ceiling']
        # prob_scale = self.update_params['branch']['probability_scale']
        angle_fl = self.update_params['branch']['angle_floor'] # m.pi/9
        angle_ceil = self.update_params['branch']['angle_ceiling'] # 3 * m.pi / 4
        # angle_scale = self.update_params['branch']['angle_scale']

        # start = c.start
        # end = c.end

        # h_dif = abs(self.hmap[c.end[0],c.end[1]] - self.hmap[c.start[0], c.start[1]])
        # density = self.dmap[c.end[0], c.end[1]]

        # prob_arg = hp_weight * h_dif + dp_weight * density
        # prob = prob_fl + ((prob_ceil - prob_fl) / (1 + m.exp(-prob_scale * prob_arg)))
        # angle_arg = ha_weight * h_dif + da_weight * density
        # angle = angle_fl + ((angle_ceil - angle_fl) / (1 + m.exp(-angle_scale * angle_arg)))
        prob = r.uniform(prob_fl, prob_ceil)
        angle = r.uniform(angle_fl, angle_ceil)
        return prob, angle

    def minor_candidates(self):
        res = self.min_possible
        self.min_possible = []
        res.extend(self.candidates)
        return res

    def minor_system(self, N):
        for n in xrange(N):
            print "min: %d" % n
            self.candidates = self.min_update()
            if len(self.candidates) == 0:
                break

    def min_update(self):
        next_candidates = []
        for c in self.candidates:
            local_candidates = self.get_locals(c)
            if len(local_candidates) == 0:
                continue
            best = self.best_global(local_candidates)
            possibles = self.match_min(best)
            next_candidates.extend(possibles)
        return self.trim(next_candidates)

    def match_min(self, c):
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
        c_stop = self.update_params['match']['candidate_stop']

        noncross = self.update_params['match']['min_maj_noncross']

        for a in self.maj_accepted:
            if la.norm(c_end - a.start) < p_TOL:
                branch_flag = False
                c.set_end(a.start)
                # snap to start-point
                if not noncross and min_angle_difference(a.angle, c_angle) > m_TOL:
                    next_c= [Seed(c.end, c.angle, c_length)]
                break
            elif la.norm(c_end - a.end) < p_TOL:
                branch_flag = False
                c.set_end(a.end)
                #snap to end-point
                if not noncross and min_angle_difference(a.angle, c_angle) > m_TOL:
                    next_c= [Seed(c.end, c.angle, c_length)]
                break
            elif a.seed_nondiverg(c):
                # intersect or merge
                i_length = a.intersect_length(c)
                if not c_length > i_length and abs(i_length - c_length) < i_TOL:
                    branch_flag = False
                    c.set_length(i_length)
                    if noncross or min_angle_difference(a.angle, c_angle) > m_TOL:
                        next_c= [Seed(c.end, c.angle, c_length)]
                break
        for a in self.min_accepted:
            if la.norm(c_end - a.start) < p_TOL:
                branch_flag = False
                c.set_end(a.start)
                # snap to start-point
                if min_angle_difference(a.angle, c_angle) > m_TOL:
                    next_c= [Seed(c.end, c.angle, c_length)]
                break
            elif la.norm(c_end - a.end) < p_TOL:
                branch_flag = False
                c.set_end(a.end)
                #snap to end-point
                if min_angle_difference(a.angle, c_angle) > m_TOL:
                    next_c= [Seed(c.end, c.angle, c_length)]
                break
            elif a.seed_nondiverg(c):
                # intersect or merge
                i_length = a.intersect_length(c)
                if c_length > i_length or abs(i_length - c_length) < i_TOL:
                    branch_flag = False
                    c.set_length(i_length)
                    if min_angle_difference(a.angle, c_angle) > m_TOL:
                        next_c= [Seed(c.end, c_angle, c_length)]
                break

        self.min_add(c)

        if branch_flag:
            next_c, branches, _ = self.branch(c)
            if r.random() < len(self.candidates) / c_stop:
                next_c = []
                branches = [] # Delete if unsatisfactory
            next_c.extend(branches)

        return next_c

    def min_add(self, c):
        accept = True
        if c.length < 1e-1:
            accept = False
        for a in self.maj_accepted:
            if self.redundant(c, a):
                accept = False
        for a in self.min_accepted:
            if self.redundant(c, a):
                accept = False
        if accept:
            self.min_accepted.append(c.get_segment())

    def debug_svg(self, fName):
        file_name = fName + "_debug.svg"
        file = open(file_name, 'w')
        self.prep_svg(file)
        self.write_debug(file)
        self.end_svg(file)
        file.close()

    def write_debug(self, file):
        for p in self.min_possible:
            file.write(self.segment_svg(p, 'blue', 1))
        for c in self.candidates:
            file.write(self.segment_svg(c, 'green', 1))
        for a in self.min_accepted:
            file.write(self.segment_svg(a, 'black', 1))
        for a in self.maj_accepted:
            file.write(self.segment_svg(a, 'red', 1))

    def save_svg(self, fName):
        file_name = fName + ".svg"
        print "saving as: %s" % file_name
        file = open(file_name, 'w')
        self.prep_svg(file)
        self.write_accepted(file)
        self.end_svg(file)
        file.close()
        # print "Saved svg as %s" % fName

    def save_svg_img(self, fName, imName):
        file_name = fName + ".svg"
        print "saving as %s" % file_name
        file = open(file_name, 'w')
        self.prep_svg_img(file, imName)
        self.write_accepted(file)
        self.end_svg(file)
        file.close()

    def prep_svg(self, file):
        shape = self.hmap.shape
        width, height = shape[1], shape[0]
        dim_string = '<svg width="%i" height="%i"\n' % (width, height)
        file.write('<?xml version="1.0"?>\n')
        file.write(dim_string)
        file.write('    xmlns="http://www.w3.org/2000/svg">\n\n')

    def prep_svg_img(self, file, imName):
        shape = self.hmap.shape
        width, height = shape[1], shape[0]
        dim_string = '<svg width="%i" height="%i"\n' % (width, height)
        file.write('<?xml version="1.0"?>\n')
        file.write(dim_string)
        file.write('    xmlns="http://www.w3.org/2000/svg"')
        file.write(' xmlns:xlink= "http://www.w3.org/1999/xlink">\n\n')
        file.write('<image xlink:href="%s" height="%dpx" width="%dpx"/>\n\n' % (imName, width, height))

    def write_accepted(self, file):
        for a in self.min_accepted:
            file.write(self.segment_svg(a, 'pink', 1))
        for a in self.maj_accepted:
            file.write(self.segment_svg(a, 'red', 1))

    def segment_svg(self, seg, stroke, width):
        part1 = '<line x1="%f" y1="%f" ' % (seg.start[1], seg.start[0])
        part2 = 'x2="%f" y2="%f" ' % (seg.end[1], seg.end[0])
        part3 = 'stroke="%s" stroke-width="%i"/>\n' % (stroke, width)
        return part1 + part2 + part3

    def end_svg(self, file):
        file.write('\n</svg>')
