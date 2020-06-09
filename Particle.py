import copy

import numpy as np


class Particle():
    def __init__(self, min_x, max_x, n=2):
        self.positions = np.random.uniform(low=min_x, high=max_x, size=n)
        self.p_best_local = self.positions
        self.p_best_global_value = np.math.inf
        self.speed = 0
        self.boundary = [min_x, max_x]
        self.errors_move = []

    def move(self):
        val = copy.deepcopy(self.positions)
        val += self.speed
        if (val < self.boundary[1]).all() and (val > self.boundary[0]).all():
            self.positions = val
        else:
            self.errors_move.append([self.positions, self.speed])
        if (abs(self.speed) > self.boundary[1]).all():
            self.speed = 0
        # for _, x in enumerate(val):
        #     # if self.boundary[1] > x > self.boundary[0]:
        #     if self.boundary[1] > x > self.boundary[0]:
        #         self.positions = val
        #
        #     else:
        #         self.errors_move.append([self.positions, self.speed])
        #     if x < 0.0:
        #         print(x)
