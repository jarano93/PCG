#!/usr/bin/py2
import numpy as np
import numpy.linalg as la

PI2 = np.pi * 2

class Segment:
    def __init__(self, start, end):
        self.start = np.array(start)
        self.end = np.array(end)
        self.angle =  np.arctan((end[1] - start[1]) / (end[0] - start[0] ))
        # arctan returns -pi/2 to pi/2 WANT 0 to 2pi
        self.length = la.norm(self.end - self.start)

class Seed:
    def __init__(self, start, angle, length):
        self.start = start
        self.angle = angle # in radians
        self.length = length
        self.end = start + length * np.array([ np.cos(angle), np.sin(angle) ])

    def seg(self):
        return Segment((start, end), False)


class SSLSys:
    def __init__(self):
        self.accepted = []
        self.possible = []

    def seed_sys(self, seed):
        self.seed = seed

    def update(self):
        # use local & global fitness evals, choose best set as candidate
        local_candidates = get_locals()
        # if no local candidates were acceptable stop that specific path
        if len(local_candidates) == 0:
            return self.possible.pop(0)
        best_candidate = best_global(local_candidates)
        next_candidate, next_possibles = self.match_accepeted(best_candidate)
        self.possible.extend(next_possibles)
        return next_candidate

    def grow(self, N):
        for n in xrange(N):
            self.candidate = self.update()

    def get_locals(self):
        locals = []
        num_locals, angle_range, len_range = self.query_local()
        # get principal candidate from the seed
        if self.valid_local(self.candidate):
            locals.append(self.cadidate)
        # Monte Carlo & test to generate other local candidates
        for i in xrange(num_locals):
            temp = self.gen_local(angle_range, len_range) # uniform distribution
            if self.valid_local(temp): # test if generated seed is local
                locals.append(temp)
        return locals

    def best_global(self, locals):
        scores = {}
        for l in locals:
            scores[self.test_global(l)] = l
        return scores[max(scores)]

    def match_accepted(self, best):
        # if candidate would intersect an accepted segment generate a crossing
        # (reduce length to intersection, new candidate at intersection, bisect existing segment)
        # (if angles are within a certain TOL merge candidate with existing segment)
        # if candidate is a little shy of intersect an accepted segment extend & generate a crossing
        # if candidate is a little shy of an existing crossing extend to crossing
        pass
