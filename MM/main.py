import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import degree


def linerFunc(x, y):
    n = 6
    ar = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    for i in range(n):
        ar[0, 0] += x[i] ** 2
        ar[0, 1] += x[i]
        ar[0, 2] += x[i] * y[i]
        ar[1, 0] += x[i]
        ar[1, 1] = n
        ar[1, 2] += y[i]

    delObs = ar[0, 0] * ar[1, 1] - ar[0, 1] * ar[1, 0]
    del1Chast = ar[0, 2] * ar[1, 1] - ar[1, 2] * ar[1, 0]
    del2Chast = ar[0, 0] * ar[1, 2] - ar[1, 0] * ar[0, 2]
    a = del1Chast / delObs
    b = del2Chast / delObs

    x_line = np.linspace(min(x),max(x),100)
    y_line = a * x_line + b

    newY= [(a*x[i]+b-y[i])**2 for i in range(n)]

    return x_line,y_line,f'Линейная: {a:.2f}x + {b:.2f}'

def degreeFunc(x,y):
    n=6
    newX = [np.log(x[i]) for i in range(n)]
    newY = [np.log(y[i]) for i in range(n)]

    ar = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    for i in range(n):
        ar[0, 0] += newX[i] ** 2
        ar[0, 1] += newX[i]
        ar[0, 2] += newX[i] * newY[i]
        ar[1, 0] += newX[i]
        ar[1, 1] = n
        ar[1, 2] += newY[i]

    delObs = ar[0, 0] * ar[1, 1] - ar[0, 1] * ar[1, 0]
    del1Chast = ar[0, 2] * ar[1, 1] - ar[1, 2] * ar[1, 0]
    del2Chast = ar[0, 0] * ar[1, 2] - ar[1, 0] * ar[0, 2]
    a = del1Chast / delObs
    b = del2Chast / delObs

    trueB = np.exp(b)

    x_line = np.linspace(min(x),max(x),100)
    y_line = trueB*np.power(x_line,a)


    return x_line,y_line, f'Степенная: {trueB:.2f} * x^{a:.2f}'

def expFunc(x,y):
    n=6
    newY=[np.log(y[i]) for i in range(n)]

    ar = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    for i in range(n):
        ar[0, 0] += x[i] ** 2
        ar[0, 1] += x[i]
        ar[0, 2] += x[i] * newY[i]
        ar[1, 0] += x[i]
        ar[1, 1] = n
        ar[1, 2] += newY[i]

    delObs = ar[0, 0] * ar[1, 1] - ar[0, 1] * ar[1, 0]
    del1Chast = ar[0, 2] * ar[1, 1] - ar[1, 2] * ar[1, 0]
    del2Chast = ar[0, 0] * ar[1, 2] - ar[1, 0] * ar[0, 2]
    a = del1Chast / delObs
    b = del2Chast / delObs

    trueB = np.exp(b)

    x_line = np.linspace(min(x),max(x),100)
    y_line = trueB*np.exp(x_line*a)

    return x_line,y_line,f'Показательная: y= {trueB:.2f} * e^{a:.2f}x'

def quadrFunx(x,y):
    n=6
    ar = np.array([[0.0,0.0,0.0,0.0],
                   [0.0,0.0,0.0,0.0],
                   [0.0,0.0,0.0,0.0]])
    for i in range(n):
        ar[0,0] += x[i]**4
        ar[0,1] += x[i]**3
        ar[0,2] += x[i]**2
        ar[0,3] += x[i]**2*y[i]
        ar[1,0] += x[i]**3
        ar[1,1] += x[i]**2
        ar[1,2] += x[i]
        ar[1,3] += x[i]*y[i]
        ar[2,0] += x[i]**2
        ar[2,1] += x[i]
        ar[2,2] = n
        ar[2,3] +=y[i]
    delObs = (ar[0,0]*ar[1,1]*ar[2,2]+ar[0,1]*ar[1,2]*ar[2,0]+ar[0,2]*ar[1,0]*ar[2,1] -
              ar[0,2]*ar[1,1]*ar[2,0] - ar[0,0]*ar[1,2]*ar[2,1] - ar[0,1]*ar[1,0]*ar[2,2])
    del1Chast = (ar[0,3]*ar[1,1]*ar[2,2]+ar[0,1]*ar[1,2]*ar[2,3]+ar[0,2]*ar[1,3]*ar[2,1] -
              ar[0,2]*ar[1,1]*ar[2,3] - ar[0,3]*ar[1,2]*ar[2,1] - ar[0,1]*ar[1,3]*ar[2,2])
    del2Chast = (ar[0,0]*ar[1,3]*ar[2,2]+ar[0,3]*ar[1,2]*ar[2,0]+ar[0,2]*ar[1,0]*ar[2,3] -
              ar[0,2]*ar[1,3]*ar[2,0] - ar[0,0]*ar[1,2]*ar[2,3] - ar[0,3]*ar[1,0]*ar[2,2])
    del3Chast = (ar[0,0]*ar[1,1]*ar[2,3]+ar[0,1]*ar[1,3]*ar[2,0]+ar[0,3]*ar[1,0]*ar[2,1] -
              ar[0,3]*ar[1,1]*ar[2,0] - ar[0,0]*ar[1,3]*ar[2,1] - ar[0,1]*ar[1,0]*ar[2,3])

    a = del1Chast / delObs
    b = del2Chast / delObs
    c = del3Chast / delObs

    x_line = np.linspace(min(x),max(x),100)
    y_line = a * x_line**2 + b *x_line + c

    return x_line,y_line,f'Квадратичная: y = {a:.2f}x^2 + {b:.2f}x + {c:.2f}'

def plotAllGraph(x,y):
    plt.figure(figsize=(12,8))
    plt.scatter(x,y,color='black',s=50,label='Исходные точки',zorder=5)

    x_line_lin, y_line_lin, label_lin = linerFunc(x,y)
    x_line_deg, y_line_deg, label_deg = degreeFunc(x,y)
    x_line_exp, y_line_exp, label_exp = expFunc(x,y)
    x_line_quad, y_line_quad, label_quad = quadrFunx(x,y)

    plt.plot(x_line_lin,y_line_lin,color='red',linewidth=2,label=label_lin)
    plt.plot(x_line_deg, y_line_deg, color='blue', linewidth=2, label=label_deg)
    plt.plot(x_line_exp, y_line_exp, color='green', linewidth=2, label=label_exp)
    plt.plot(x_line_quad, y_line_quad, color='purple', linewidth=2, label=label_quad)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Сравнение методов аппроксимации')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

x=[1.0,3.0,5.0,7.0,9.0,11.0]
y=[2.0,10.1,22.6,37.1,54.5,73.2]

plotAllGraph(x,y)