import matplotlib.pyplot as plt
import numpy as np
from math import pi
from numpy import sin, cos, sqrt, power
from matplotlib.animation import FuncAnimation


def startup():
    set_input_cord = input('Czy chcesz podać wspolrzedne czworokata? (t/n) \n')
    if set_input_cord == 't':
        print('Podaj wspolrzedne:\n')
        in_x2 = float(input('x2 = '))
        in_x3 = float(input('x3 = '))
        in_y3 = float(input('y3 = '))
        in_x4 = float(input('x4 = '))
        in_y4 = float(input('y4 = '))
    else:
        in_x2 = 3
        in_x3, in_y3 = 3, 4
        in_x4, in_y4 = 1, 4

    set_input_ground = input('Czy chcesz podać prostoa po ktorej ma sie toczyc czworokat? (t/n)\n')
    if set_input_ground == 't':
        gnd_fnc = input("Podaj wzor, jako argument uzyj i: ")
    else:
        gnd_fnc = 0

    return gnd_fnc, in_x2, in_x3, in_y3, in_x4, in_y4


fig, ax = plt.subplots()
bok1, =    plt.plot([], [], color='orange', animated=True)
bok2, =    plt.plot([], [], color='blue', animated=True)
bok3, =    plt.plot([], [], color='green', animated=True)
bok4, =    plt.plot([], [], color='purple', animated=True)
vertex1, = plt.plot([], [], color='red', animated=True)
vertex2, = plt.plot([], [], color='green', animated=True)
vertex3, = plt.plot([], [], color='orange', animated=True)
vertex4, = plt.plot([], [], color='purple', animated=True)
ground, = plt.plot([], [], color='black', animated=True)
plt.grid()
xdata1, ydata1 = [], []
xdata2, ydata2 = [], []
xdata3, ydata3 = [], []
xdata4, ydata4 = [], []

#variables in dev mode
input_x1, input_y1 = 0, 0
input_x2, input_y2 = 1, 0
input_x3, input_y3 = 2, 3
input_x4, input_y4 = 1, 3
ground_fnc = '0'

# ground_fnc, input_x2, input_x3, input_y3, input_x4, input_y4 = startup()

a = sqrt(power(input_x2-input_x1, 2) + power(input_y2-input_y1, 2))
b = sqrt(power(input_x3-input_x2, 2) + power(input_y3-input_y2, 2))
c = sqrt(power(input_x4-input_x3, 2) + power(input_y4-input_y3, 2))
d = sqrt(power(input_x4-input_x1, 2) + power(input_y4-input_y1, 2))
e = sqrt(power(input_x3-input_x1, 2) + power(input_y3-input_y1, 2))
f = sqrt(power(input_x4-input_x2, 2) + power(input_y4-input_y2, 2))

alpha = np.arccos((a * a + b * b - e * e) / (2 * a * b))
beta = np.arccos((b * b + c * c - f * f) / (2 * b * c))
gamma = np.arccos((c * c + d * d - e * e) / (2 * c * d))
delta = np.arccos((d * d + a * a - f * f) / (2 * d * a))

#global variables
kat = 0
first_cyc = True
second_cyc = False
third_cyc = False
fourth_cyc = False
x1_off = a
x2_off = a
x3_off = a
x4_off = a
y1_off = 0
y2_off = 0
y3_off = 0
y4_off = 0


def init():
    ax.set_xlim(0, 20)
    ax.set_ylim(-10, 10)
    data = [eval(ground_fnc) for i in np.linspace(-500, 500, 300000)]
    ground.set_data(np.linspace(-500, 500, 300000), data)
    return vertex1, bok1, bok2, bok3, bok4, ground


def update(t):
    global first_cyc, second_cyc, third_cyc, fourth_cyc
    global kat
    global x1_off, y1_off
    global x2_off, y2_off
    global x3_off, y3_off
    global x4_off, y4_off

    if first_cyc:
        # **************** I PART **********************
        print('PART I')
        x1 = a*sin(t - kat - pi/2) + x2_off # zamiast zmiennych globalnych moge urzyc xdata[] i ydata[]
        y1 = a*cos(t - kat - pi/2) + y2_off
        # **********
        x2 = x2_off
        y2 = y2_off
        # **********
        x3 = b*sin(t - kat - pi/2 + alpha) + x2_off
        y3 = b*cos(t - kat - pi/2 + alpha) + y2_off
        # **********
        x4 = f*sin(t - kat - pi/2 + np.arccos((a*a + f*f - d*d) / (2*a*f))) + x2_off
        y4 = f*cos(t - kat - pi/2 + np.arccos((a*a + f*f - d*d) / (2*a*f))) + y2_off
        # **********
        i = x3
        if round(y3, 1) == round(eval(ground_fnc), 1):
            x3_off = x3
            y3_off = y3
            first_cyc = False
            second_cyc = True
            x = x3 - x2
            y = y3 - y2
            if y < 0:
                kat = t - np.arccos((x*x + b*b - y*y) / (2*x*b))
            else:
                kat = t + np.arccos((x * x + b * b - y * y) / (2 * x * b))
            return ground, vertex1, bok1, bok2, bok3, bok4
    elif second_cyc:
        # **************** II PART **********************
        print('PART II')
        x1 = e*sin(t - pi/2 - kat + np.arccos((b*b + e*e - a*a) / (2*b*e))) + x3_off
        y1 = e*cos(t - pi/2 - kat + np.arccos((b*b + e*e - a*a) / (2*b*e))) + y3_off
        # **********
        x2 = b*sin(t - pi/2 - kat) + x3_off
        y2 = b*cos(t - pi/2 - kat) + y3_off
        # **********
        x3 = x3_off 
        y3 = y3_off
        # **********
        x4 = c*sin(t - pi/2 - kat + beta) + x3_off
        y4 = c*cos(t - pi/2 - kat + beta) + y3_off
        # **********
        i = x4
        if round(y4, 1) == round(eval(ground_fnc), 1):
            x4_off = x4
            y4_off = y4
            second_cyc = False
            third_cyc = True
            x = x4 - x3
            y = y4 - y3
            if y < 0:
                kat = t - np.arccos((x*x + c*c - y*y) / (2*x*c))
            else:
                kat = t + np.arccos((x * x + c * c - y * y) / (2 * x * c))
            return ground, vertex1, bok1, bok2, bok3, bok4
    elif third_cyc:
        # **************** III PART **********************
        print('PART III')
        x1 = d*sin(t - pi/2 - kat + gamma) + x4_off
        y1 = d*cos(t - pi/2 - kat + gamma) + y4_off
        # **********
        x2 = f*sin(t - pi/2 - kat + np.arccos((c*c + f*f - b*b) / (2*c*f))) + x4_off
        y2 = f*cos(t - pi/2 - kat + np.arccos((c*c + f*f - b*b) / (2*c*f))) + y4_off
        # **********
        x3 = c*sin(t - pi/2 - kat) + x4_off
        y3 = c*cos(t - pi/2 - kat) + y4_off
        # **********
        x4 = x4_off 
        y4 = y4_off
        # **********
        i = x1
        if round(y1, 1) == round(eval(ground_fnc), 1):
            x1_off = x1
            y1_off = y1
            third_cyc = False
            fourth_cyc = True
            x = x1 - x4
            y = y1 - y4
            if y < 0:
                kat = t - np.arccos((x * x + d * d - y * y) / (2 * x * d))
            else:
                kat = t + np.arccos((x * x + d * d - y * y) / (2 * x * d))
            return ground, vertex1, bok1, bok2, bok3, bok4
    elif fourth_cyc:
        # **************** IV PART **********************
        print('PART IV')
        x1 = x1_off 
        y1 = y1_off
        # **********
        x2 = a*sin(t - pi/2 - kat + delta) + x1_off
        y2 = a*cos(t - pi/2 - kat + delta) + y1_off
        # **********
        x3 = e*sin(t - pi/2 - kat + np.arccos((d*d + e*e - c*c) / (2*d*e))) + x1_off
        y3 = e*cos(t - pi/2 - kat + np.arccos((d*d + e*e - c*c) / (2*d*e))) + y1_off
        # **********
        x4 = d*sin(t - pi/2 - kat) + x1_off
        y4 = d*cos(t - pi/2 - kat) + y1_off
        # **********
        i = x2
        if round(y2, 1) == round(eval(ground_fnc), 1):
            x2_off = x2
            y2_off = y2
            fourth_cyc = False
            first_cyc = True
            x = x2 - x1
            y = y2 - y1
            if y < 0:
                kat = t - np.arccos((x * x + a * a - y * y) / (2 * x * a))
            else:
                kat = t + np.arccos((x * x + a * a - y * y) / (2 * x * a))
            return ground, vertex1, bok1, bok2, bok3, bok4
    else:
        return ground, vertex1, bok1, bok2, bok3, bok4

    v1 = [[x1, x2], [y1, y2]]
    v2 = [[x2, x3], [y2, y3]]
    v3 = [[x3, x4], [y3, y4]]
    v4 = [[x4, x1], [y4, y1]]

    xdata1.append(x1)
    ydata1.append(y1)
    vertex1.set_data(xdata1, ydata1)
    # xdata2.append(x2)
    # ydata2.append(y2)
    # vertex2.set_data(xdata2, ydata2)
    # xdata3.append(x3)
    # ydata3.append(y3)
    # vertex3.set_data(xdata3, ydata3)
    # xdata4.append(x4)
    # ydata4.append(y4)
    # vertex4.set_data(xdata4, ydata4)
    bok1.set_data(v1)
    bok2.set_data(v2)
    bok3.set_data(v3)
    bok4.set_data(v4)
    return ground, vertex1, bok1, bok2, bok3, bok4
    # return ground, vertex1, vertex2, vertex3, vertex4


ani = FuncAnimation(fig,
                    update,
                    frames=np.linspace(0, 1000, 300000),
                    init_func=init,
                    interval=1,
                    repeat=False,
                    blit=True)
plt.show()
