import csv
import time
import matplotlib.pyplot as plt

from Particle import Particle, np


class Swarm():
    def __init__(self, min_x, max_x, n, population, table, W=0.6, c1=2, c2=2, generations=100, flag_test=2):
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
            'starting_value': self.particles
        }
        self.best_values_plot = []

    def __plot_3d(self):
        x = np.linspace(self.const['min_x'], self.const['max_x'], num=100)
        y = np.linspace(self.const['min_x'], self.const['max_x'], num=100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((100, 100))
        for i in range(0, X.shape[0]):
            for j in range(0, Y.shape[0]):
                Z[i][j] = self.fitness([X[i][j], Y[i][j]])
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        xdata = [self.g_best_global[0]]
        ydata = [self.g_best_global[1]]
        zdata = [self.best_global_value]
        for i in range(0, len(self.const['starting_value'])):
            xdata.append(self.const['starting_value'][i].positions[0])
            ydata.append(self.const['starting_value'][i].positions[1])
            zdata.append(self.fitness([xdata[-1], ydata[-1]]))
        ax.scatter3D(xdata, ydata, zdata, marker='o')

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
        plt.plot(_x, _y)

    def fitness(self, list_of_x: np.ndarray):
        # value = [x ** 2 for x in list_of_x]
        value = 0.0
        for i in range(0, len(list_of_x)):
            value += list_of_x[i] ** 2
        return value

    def fitness_MD(self, list_of_x: np.ndarray):
        value = 0.0
        for el in self.T:
            value += abs(list_of_x[0] * el[0] ** list_of_x[1] - el[1])
        return value

    def find_best_particle(self):
        for particle in self.particles:
            new_boid = self.fitness_MD(particle.positions)
            if new_boid < self.best_global_value:
                self.best_global_value = new_boid
                self.g_best_global = particle.positions

    def update_particles(self):
        for particle in self.particles:
            particle.p_best_global_value = self.best_global_value

    def move_particles(self):
        for particle in self.particles:
            new_speed = (self.const['W'] * particle.speed) + (self.const['c1'] * np.random.random()) * (
                    particle.p_best_local - particle.positions) + (self.const['c2'] * np.random.random()) * (
                                self.g_best_global - particle.positions)
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
            if i % 500 == 0:
                for part in self.particles:
                    print("a= {} ,b= {} , speed= {}".format(part.positions[0], part.positions[1], part.speed))

        # with open("coef.csv", "wb") as file:
        #     writer = csv.writer(file)
        #     writer.writerow(self.g_best_global)
        np.savetxt('coef.csv', self.g_best_global, delimiter=',')
        self.plot_MD_of_gen()
        plt.show()
