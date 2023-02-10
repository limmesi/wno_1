import matplotlib.pyplot as plt
import numpy as np
from math import pi
from numpy import sin, cos
import random
from matplotlib.animation import FuncAnimation
from copy import copy


class Tiger:
    def __init__(self, x_center='rand', y_center='rand'):
        self.vector = None
        self.borders = []
        if x_center == 'rand' and y_center == 'rand':
            self.x = random.randint(-50, 50)
            self.y = random.randint(-50, 50)
        else:
            self.x = x_center
            self.y = y_center

    def add_vector(self, x_end, y_end):
        self.vector = [(self.x, x_end), (self.y, y_end)]

    def add_borders(self, border_x, border_y):
        self.borders.append(Tiger(border_x, border_y))


def JarvisAlgorithm(points):
    left_index = 0
    for i in range(1, len(points)):
        if points[i].x < points[left_index].x:
            left_index = i
        elif points[i].x == points[left_index].x:
            if points[i].y > points[left_index].y:
                left_index = i
    output = []
    p = left_index
    q = 0
    first_cyc = True

    while p != left_index or first_cyc:
        first_cyc = False
        output.append(p)
        q = (p + 1) % len(points)
        for i in range(len(points)):
            if (points[p].y - points[i].y) * (points[q].x - points[p].x) - (points[p].x - points[i].x) * (
                    points[q].y - points[p].y) < 0:
                q = i
        p = q

    return output


def flatten_list(_2d_list):
    flat_list = []
    for element in _2d_list:
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


tigers = [Tiger() for _ in range(0, 10)]
for tiger in tigers:
    long_digonal = random.randint(10, 30)
    short_digonal = random.randint(10, 20)
    aplha = random.uniform(0, 2 * pi)

    x = tiger.x + long_digonal * (2 / 3) * sin(aplha)
    y = tiger.y + long_digonal * (2 / 3) * cos(aplha)
    tiger.add_vector(x, y)
    tiger.add_borders(x, y)

    x = tiger.x + short_digonal * (1 / 2) * cos(aplha)
    y = tiger.y - short_digonal * (1 / 2) * sin(aplha)
    tiger.add_borders(x, y)

    x = tiger.x - long_digonal * (1 / 3) * sin(aplha)
    y = tiger.y - long_digonal * (1 / 3) * cos(aplha)
    tiger.add_borders(x, y)

    x = tiger.x - short_digonal * (1 / 2) * cos(aplha)
    y = tiger.y + short_digonal * (1 / 2) * sin(aplha)
    tiger.add_borders(x, y)


fig, ax = plt.subplots()
temp_cages = [plt.plot([], [], color='green', animated=True) for i in range(len(tigers)*4)]
cages = [[copy(temp_cages[j][0]) for j in range(4)] for _ in range(len(tigers)*4)]

temp_vectors = [plt.plot([], [], color='red', animated=True) for _ in range(len(tigers))]
vectors = [copy(temp_vectors[i][0]) for i in range(len(tigers))]

temp_centers = [plt.plot([], [], 'bo', animated=True) for i in range(len(tigers))]
centers = [copy(temp_centers[i][0]) for i in range(len(tigers))]

temp_borders = [plt.plot([], [], color='red', linewidth='0.5', animated=True) for i in range(len(tigers)*4)]
borders = [copy(temp_borders[i][0]) for i in range(len(tigers)*4)]


def init():
    ax.set_xlim(-120, 120)
    ax.set_ylim(-120, 120)
    ax.grid()
    return *vectors, *centers, *borders


def update(t):
    for tiger in tigers:
        speed = random.uniform(0.0001, 0.01)
        dx = speed * (tiger.borders[0].x - tiger.x)
        dy = speed * (tiger.borders[0].y - tiger.y)
        tiger.x += dx
        tiger.y += dy
        for point in tiger.borders:
            point.x += dx
            point.y += dy
        tiger.vector = [(tiger.x, tiger.borders[0].x), (tiger.y, tiger.borders[0].y)]

    all_points = []
    for i in range(len(tigers)):
        all_points.append(tigers[i])
        all_points = all_points + tigers[i].borders

    chain = JarvisAlgorithm(all_points)

    for i, tiger in enumerate(tigers):
        centers[i].set_data(tiger.x, tiger.y)
        vectors[i].set_data(*tiger.vector)
        cages[i][0].set_data((tiger.borders[0].x, tiger.borders[1].x), (tiger.borders[0].y, tiger.borders[1].y))
        cages[i][1].set_data((tiger.borders[2].x, tiger.borders[1].x), (tiger.borders[2].y, tiger.borders[1].y))
        cages[i][2].set_data((tiger.borders[3].x, tiger.borders[2].x), (tiger.borders[3].y, tiger.borders[2].y))
        cages[i][3].set_data((tiger.borders[0].x, tiger.borders[3].x), (tiger.borders[0].y, tiger.borders[3].y))
    flatten_cages = flatten_list(cages)

    for i in range(len(chain)):
        start_x = all_points[chain[i]].x
        start_y = all_points[chain[i]].y
        if i < (len(chain) - 1):
            end_x = all_points[chain[i + 1]].x
            end_y = all_points[chain[i + 1]].y
            borders[i].set_data((start_x, end_x), (start_y, end_y))
        else:
            borders[i].set_data((start_x, all_points[chain[0]].x), (start_y, all_points[chain[0]].y))

    for i in range(len(chain), len(borders)):
        borders[i].set_data((0, 0), (0, 0))

    return *flatten_cages, *vectors, *centers, *borders


ani = FuncAnimation(fig,
                    update,
                    frames=np.linspace(0, 1000, 300_000),
                    init_func=init,
                    interval=1,
                    repeat=False,
                    blit=True)
plt.show()
