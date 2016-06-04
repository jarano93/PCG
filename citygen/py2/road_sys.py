#!/usr/bin/python2

import math as m
import random as r
import numpy as np
import numpy.linalg as la

def get_angle(point):
    angle = m.atan2(point[1], point[0])
    if angle < 0:
        angle += 2 * m.pi
    return angle

class Seed:
    def __init__(self, start, angle, length):
        self.start = np.array(start)
        self.angle = angle # [0, 2pi)
        self.length = length
        self.init_end()

    def init_end(self):
        self.end = self.start + length * np.array([m.cos(angle), m.sin(angle)])

    def get_segment(self):
        return Segment(self.start, self.end, self.length, self.angle)

    def set_length(self, length):
        self.length = length
        self.init_end()

    def set_end(self, end):
        self.end = end
        dif = end - self.start
        self.length = la.norm(dif)
        self.angle = m.tan2(dif[1], dif[0])

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
        self.vmap = vmap
        self.maj_accepted = []
        self.maj_possible = []
        self.min_accepted = []
        self.min_possible = []

    def create_system(self, N_mc, N_maj, N_min, maj_rules, min_rules):
        self.majors, self.minors = [], []
        self.major_candidate(N_mc)
        self.majors(N_maj, maj_rules)
        self.minor_candidate()
        self.minors(N_min, min_rules)
        self.save_svg()

    def major_candidate(self, N):
        for n in xrange(N):
            pass

    def majors(self, N, major_rules):
        self.update_rules = major_rules # major rules can change while updating
        for n in xrange(N):
            self.candidate = self.update(True)

    def update(self, major_flag):
        local_candidates = get_locals()
        if len(local_candidates) == 0:
            # for now, assume there is always a possible
            # fuck that, make it so that there IS always a possible
            if major_flag:
                return self.maj_possible.pop(0)
            else:
                return self.min_possible.pop(0)
        else:
            best_candidate = best_global(local_candidates)
            next_candidate, next_possibles = self.match_accepted(best_candidate, major_flag)
            self.trim_possibles(next_possibles)
            if major_flag:
                self.maj_possible.extend(next_possibles)
            else:
                self.min_possible.extend(next_possibles)
            return next_candidate


    def get_locals(self):
        # get principal candidate from the seed
        if self.valid_local(self.candidate):
            locals.append(self.candidate)
        # Monte Carlo & test to generate other local candidates
        locals = []
        num_locals, angle_range, len_range = self.gen_conditions(self.candidate.start)
        for i in xrange(num_locals):
            temp = self.gen_local(angle_range, len_range)
            if self.valid_local(temp):
                locals.append(temp)
        return locals

    def best_global(self, candidates):
        best = candidates[0]
        val = self.test_global(best)
        for c in candidates:
            c_val = self.test_global(c)
            if c_val > val:
                best = c
                val = c_val
        return best

    def test_global(self, candidate):
        # use the update rules
        pass


    def match_accepted(self, candidate, major):
        # TURN BEST CANDIDATE INTO ACCEPTED SEGMENT HERE
        # use update rules here also
        c_angle = candidate.angle
        c_length = candidate.length
        c_end = candidate.end
        i_break = self.update_rules['intersect_break']
        i_TOL = self.update_rules['intersect_TOL']
        m_TOL = self.update_rules['merge_TOL']
        e_TOL = self.update_rules['endpoint_TOL']
        possibles = []
        stop_flag = False
        no_intersect = True
        # if candidate would intersect an accepted segment generate a crossing
        for a in self.maj_accepted:
            maj_angle = [a.angle, a.angle + m.pi % 2 * m.pi]
            if la.norm(c_end - a.start) < e_TOL:
                no_intersect = False
                candidate.set_end(a.start)
                if not major and i_break:
                    stop_flag = True
                else:
                    if abs(maj_angle[0] - c_angle) < m_TOL or abs(maj_angle[1] - c_angle) < m_TOL:
                        stop_flag = True
                    else:
                        next_candidate = Seed(c_end, c_angle, c_length)
            elif la.norm(c_end - a.end) < e_TOL:
                no_intersect = False
                candidate.set_end(a.end)
                if not major and i_break:
                    stop_flag = True
                else:
                    if abs(maj_angle[0] - c_angle) < m_TOL or abs(maj_angle[1] - c_angle) < m_TOL:
                        stop_flag = True
                    else:
                        next_candidate = Seed(c_end, c_angle, c_length)
            elif a.seed_nondiverg(candidate):
                i_length = a.intersect_length(candidate)
                if c_length > i_length or abs(i_legnth - c_length) < i_TOL:
                    no_intersect = False
                    candidate.set_length(i_length)
                    if not major and i_break:
                        stop_flag = True
                    else:
                        # (if angles are within a certain TOL merge candidate with existing segment)
                        if abs(maj_angle[0] - c_angle) < m_TOL or abs(maj_angle[1] - c_angle) < m_TOL:
                            stop_flag = True
                        else:
                            next_candidate = Seed(c_end, c_angle, c_length)
        if not major:
            for a in self.min_accepted:
                min_angle = [a.angle, a.angle + m.pi % 2 * m.pi]
                if la.norm(c_end - a.start) < e_TOL:
                    no_intersect = False
                    candidate.set_end(a.start)
                    if abs(maj_angle[0] - c_angle) < m_TOL or abs(maj_angle[1] - c_angle) < m_TOL:
                        stop_flag = True
                    else:
                        next_candidate = Seed(c_end, c_angle, c_length)
                elif la.norm(c_end - a.end) < e_TOL:
                    no_intersect = False
                    candidate.set_end(a.end)
                    if abs(maj_angle[0] - c_angle) < m_TOL or abs(maj_angle[1] - c_angle) < m_TOL:
                        stop_flag = True
                    else:
                        next_candidate = Seed(c_end, c_angle, c_length)
                elif a.seed_nondiverg(candidate):
                    i_length = a.intersect_length(candidate)
                    if c_length > i_length or abs(i_legnth - c_length) < i_TOL:
                        no_intersect = False
                        candidate.set_length(i_length)
                        if abs(maj_angle[0] - c_angle) < m_TOL or abs(maj_angle[1] - c_angle) < m_TOL:
                            stop_flag = True
                        else:
                            next_candidate = Seed(c_end, c_angle, c_length)
        # if candidate is a little shy of intersecting an accepted segment extend & generate a crossing
        # if candidate is a little shy of an existing crossing extend to crossing

        if no_intersect:
            # BRANCHES!
            next_candidate, possibles = self.branch(candidate)

        if major:
            maj_accepted = candidate.get_segmet()
        else:
            min_accepted = candidate.get_segment()
        # if there is no next candidate from current segment series use 0th possible
        if stop_flag:
            if major:
                next_candidate = self.maj_possible.pop(0)
            else:
                next_candidate = self.min_possible.pop(0)

        return next_candidate, possibles

    def trim_possibles(self, possibles):
        for m in self.maj_possibles:
            for p in possibles:
                if self.redundant(p, m):
                    possibles.remove(p)
        for m in self.min_possibles:
            for p in possibles:
                if self.redundant(p, m):
                    possibles.remove(p)
        return possibles

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

    def minor_candidate(self):
        self.candidate = self.min_possible.pop(0)
        self.maj_possible = []
        # on second thought don't do the following
        # pop_val = r.randrange(len(self.min_possible))
        # self.canididate = self.min_possible.pop(pop_val)

    def minors(self, N, minor_rules):
        self.update_rules = minor_rules # minor rules can change while updating
        for n in xrange(N):
            self.candidate = self.update(False)

    def save_svg(self):
        pass
