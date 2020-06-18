import csv
import time
import matplotlib.pyplot as plt
from math import sqrt

from Particle import Particle, np


class Swarm():
    def __init__(self, min_x, max_x, n, population, table, W=0.6, c1=2, c2=2, generations=100,
                 flag_choice=1, plt_name='default.txt', out_name='coef.txt'):
        # self.particles = np.full(population, Particle(n, max_x, min_x))
        self.particles = []
        [self.particles.append(Particle(n=n, min_x=min_x, max_x=max_x)) for i in range(0, population)]
        self.best_global_value = np.math.inf
        self.g_best_global = np.array([0, 0])
        self.T = table  # Аrray of NASA data for train/ check prog
        self.const = {
            'generations': generations,
            'population': population,
            'n': n,
            'W': W,
            'c1': c1,
            'c2': c2,
            'max_x': max_x,
            'min_x': min_x,
            'starting_value': self.particles,
            'strategy': flag_choice,
            'plot_name': plt_name,
            'coef_output_file': out_name
        }
        self.best_values_plot = []

    def __str__(self):
        ans = 'For ' + str(self.const['n']) + ' parametrs\n'
        for i in range(0, self.const['n']):
            ans += 'x' + str(i) + '= ' + str(self.g_best_global[i]) + ' '
        ans += '\nThe Best Function value ' + str(self.best_global_value)
        return ans

    def __repr__(self):
        ans = ''
        for i in range(0, self.const['n']):
            ans += 'x' + str(i) + '= ' + str(self.g_best_global[i]) + ' '
        # print(ans)
        return ans

    def plot_MD_of_gen(self):
        _x = np.arange(0, self.const['generations'] + 1)
        _y = np.around(self.best_values_plot, 5)
        fig, ax = plt.subplots()
        # fig = plt.figure()
        # plt.plot(_x, _y)
        title = "MD= " + str(self.best_global_value)
        ax.set_title(title)
        ax.plot(_x, _y)
        # plt.title = title
        plt.savefig(self.const['plot_name'])
        plt.close(fig)

    def fitness(self, list_of_x: np.ndarray):
        if self.const['strategy'] == 1:
            return self.fitness_MD(list_of_x)
        if self.const['strategy'] == 2:
            return self.fitness_MMRE(list_of_x)
        if self.const['strategy'] == 3:
            return self.fitness_RMS(list_of_x)
        if self.const['strategy'] == 4:
            return self.fitness_ED(list_of_x)


    def fitness_MD(self, list_of_x: np.ndarray):
        value = 0.0
        for el in self.T:
            # value += abs(list_of_x[0] * el[0] ** list_of_x[1] - el[1])
            value += abs(el[1] - list_of_x[0] * el[0] ** list_of_x[1])
        return value

    def fitness_MMRE(self, list_of_x: np.ndarray):
        value = 0.0
        for el in self.T:
            # value +=(abs(list_of_x[0]*el[0]**list_of_x[1]))
            value += (abs(el[1] - list_of_x[0] * el[0] ** list_of_x[1])) / el[1]
        value = (1 / self.T.shape[0]) * value
        return value

    def fitness_RMS(self, list_of_x: np.ndarray):
        value = 0.0
        for el in self.T:
            value += (el[1] - list_of_x[0] * el[0] ** list_of_x[1]) ** 2

        value = sqrt((1 / self.T.shape[0]) * value)
        return value

    def fitness_ED(self, list_of_x: np.ndarray):
        value = 0.0
        for el in self.T:
            value += (el[1] - list_of_x[0] * el[0] ** list_of_x[1]) ** 2

        value = sqrt(value)
        return value

    def find_best_particle(self):
        for particle in self.particles:
            new_boid = self.fitness(particle.positions)
            if new_boid < self.best_global_value:
                self.best_global_value = new_boid
                self.g_best_global = particle.positions

    def update_particles(self):
        for particle in self.particles:
            particle.p_best_global_value = self.best_global_value

    def move_particles(self):
        for particle in self.particles:
            if self.const['strategy'] == 1:
                new_speed = (self.const['W'] * particle.speed) + (self.const['c1'] * np.random.random()) * (
                        particle.p_best_local - particle.positions) + (self.const['c2'] * np.random.random()) * (
                                    self.g_best_global - particle.positions)
                particle.speed = new_speed
                particle.move()
            elif self.const['strategy'] == 2:
                X = (self.const['c1'] * np.random.random() * particle.p_best_local + self.const[
                    'c2'] * np.random.random() * self.g_best_global) / (
                            self.const['c1'] * np.random.random() + self.const['c2'] * np.random.random())
                new_speed = X * (particle.speed + self.const['c1'] * np.random.random() * (
                        particle.p_best_local - particle.positions) + self.const['c2'] * np.random.random() * (
                                         particle.p_best_local - particle.positions))
                particle.speed = new_speed
                particle.move()

    def start(self):
        start_time = time.time()
        for i in range(0, self.const['generations'] + 1):
            self.find_best_particle()  # Find gloval maximum
            self.update_particles()  # update  Particles of global max
            self.move_particles()  # move in the demision
            self.best_values_plot.append(self.best_global_value)
            if i % 100 == 0 or i + 1 == self.const['generations']:
                print('Generation ' + str(i) + ' Best global value ' + str(
                    self.best_global_value), ' required time --%s seconds--' % (time.time() - start_time))
            # if i % 500 == 0:
            #     for part in self.particles:
            #         print("a= {} ,b= {} , speed= {}".format(part.positions[0], part.positions[1], part.speed))
        with open(self.const['coef_output_file'], "a") as f:
            f.write(",".join(map(str, self.g_best_global)) + '\n')
        for part in self.particles:
            print("a= {} ,b= {} , speed= {}".format(part.positions[0], part.positions[1], part.speed))
        #  np.savetxt('coef.csv', self.g_best_global, delimiter=',')
        self.plot_MD_of_gen()
