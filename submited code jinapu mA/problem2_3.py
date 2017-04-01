###COMS 4701 Jianpu Ma


import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def cost(X,y,b):
    return sum(np.array((np.matrix(X)*np.matrix(b))-np.matrix(y).transpose())**2)/(2*np.array(y).size)

def gradient_descent(X,y,b,a):
    t = (np.matrix(X)*np.matrix(b))-np.matrix(y).transpose()
    grad = a* (((np.matrix(X).transpose())*t)/(np.array(y).size))
    re=b - grad
    return re


def main():
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]

    Alpha = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    n_iter = 100

    raw_data = np.loadtxt(inputfile, delimiter=',')
    l, w = raw_data.shape
    y = raw_data[0:l, w - 1]
    X = raw_data[0:l, 0:w - 1]

    age = scale(raw_data[:, [0]])
    weight = scale(raw_data[:, [1]])
    heights = scale(raw_data[:, [2]]).flatten()
    X[:, [0]] = age
    X[:, [1]] = weight
    Pre=pd.DataFrame(np.ones([len(X), 1]))

    X = pd.concat([Pre, pd.DataFrame(X)], axis=1)

    with open(outputfile, "w") as f:
        for a in Alpha:
            costfun = np.zeros(n_iter)
            b = np.zeros([3, 1])
            for i in range(n_iter):
                b = gradient_descent(X, y, b, a)
                costfun[i] = cost(X, y, b)
            f.write("{},{},{},{},{}\n".format(a, n_iter, b[0, 0], b[1, 0], b[2, 0]))

            plt.plot(costfun)
            plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(raw_data[:, [0]], raw_data[:, [1]], raw_data[:, [2]])
            x_ = np.arange(-3, 3)
            y_ = np.arange(-2, 4)
            x_, y_ = np.meshgrid(x_, y_)
            z_ = b[0, 0] + b[1, 0] * x_ + b[2, 0] * y_
            ax.plot_surface(x_, y_, z_, color='r')
            plt.show()

        best_eta = 0.6
        best_iter = 80
        # print("eta : {}".format(my_eta))
        costfun = np.zeros(best_iter)
        b = np.zeros([3, 1])
        for i in range(best_iter):
            b = gradient_descent(X, y, b, best_eta)
            costfun[i] = cost(X, y, b)
        f.write("{},{},{},{},{}\n".format(best_eta, best_eta, b[0, 0], b[1, 0], b[2, 0]))
        plt.plot(costfun)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(raw_data[:, [0]], raw_data[:, [1]], raw_data[:, [2]])
        x_ = np.arange(-3, 3)
        y_ = np.arange(-2, 4)
        x_, y_ = np.meshgrid(x_, y_)
        z_ = b[0, 0] + b[1, 0] * x_ + b[2, 0] * y_

        ax.plot_surface(x_, y_, z_, color='r')
        plt.show()

    """ fig =plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(raw_data[:, [0]],raw_data[:, [1]],raw_data[:, [2]])

        x,y= np.meshgrid(np.arange(-3,3),np.arange(-2,4))

        z= b_[0, 0] +b_[1, 0]*x +b_[2, 0]*y
        # z= 0.0002 +0.8767*x +0.0093*y
        ax.plot_surface(x,y,z)

        plt.show()"""

if __name__ == "__main__":
    main()