import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D


import matplotlib.pyplot as plt

def compute_cost(X,y,b_):
    return sum(np.array((np.matrix(X)*np.matrix(b_))-np.matrix(y).transpose())**2)/(2*np.array(y).size)

def gradient_descent(X,y,b_,alpha):

    t = (np.matrix(X)*np.matrix(b_))-np.matrix(y).transpose()
    grad = alpha* (((np.matrix(X).transpose())*t)/np.array(y).size)
    return b_ - grad


def main():
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]

    lr = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
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
    X = pd.concat([pd.DataFrame(np.ones([len(X), 1])), pd.DataFrame(X)], axis=1)

    #print (X)

    with open(outputfile, "w") as f:
        for a in lr:
            costhist = np.zeros(n_iter)
            b_ = np.zeros([3, 1])
            for i in range(n_iter):
                b_ = gradient_descent(X, y, b_, a)
                #costhist[i] = compute_cost(X, y, b_)
            f.write("{},{},{},{},{}\n".format(a, n_iter, b_[0, 0], b_[1, 0], b_[2, 0]))

            # plt.plot(costhist)
            # plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(raw_data[:, [0]], raw_data[:, [1]], raw_data[:, [2]])
            x_ = np.arange(-3, 3)
            y_ = np.arange(-2, 4)
            x_, y_ = np.meshgrid(x_, y_)
            z_ = b_[0, 0] + b_[1, 0] * x_ + b_[2, 0] * y_
            ax.plot_surface(x_, y_, z_, color='r')
            plt.show()


        my_eta = 0.6
        my_iter = 80
        # print("eta : {}".format(my_eta))
        #costhist = np.zeros(my_iter)
        b_ = np.zeros([3, 1])
        for i in range(my_iter):
            b_ = gradient_descent(X, y, b_, my_eta)
            #costhist[i] = compute_cost(X, y, b_)
        f.write("{},{},{},{},{}\n".format(my_eta, my_iter, b_[0, 0], b_[1, 0], b_[2, 0]))
        #plt.plot(costhist)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(raw_data[:, [0]], raw_data[:, [1]], raw_data[:, [2]])
        x_ = np.arange(-3, 3)
        y_ = np.arange(-2, 4)
        x_, y_ = np.meshgrid(x_, y_)
        z_ = b_[0, 0] + b_[1, 0] * x_ + b_[2, 0] * y_

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