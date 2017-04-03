###problem3


from sklearn import svm
from sklearn import cross_validation, linear_model, tree
from sklearn.model_selection import GridSearchCV,StratifiedShuffleSplit,train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
import sys
import numpy as np
import pandas as pd

import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


#inputfile=sys.argv[1]
#outputfile=sys.argv[2]


def plotscatter(A,B,label):
    plt.scatter(A,B,c=label, zorder=10, cmap=plt.cm.Paired, marker='s')
    plt.show()


def svm_linear(Xtrain,Xtest,ytrain, ytest):
    C_range = [0.1, 0.5, 1, 5, 10, 50, 100]
    param_grid_linear = dict( C=C_range)
    lineargrid = GridSearchCV(svm.SVC(kernel='linear'), param_grid=param_grid_linear, cv=5)
    lineargrid.fit(Xtrain, ytrain)
    print("Linear kernel:: The best parameters are %s with a score of %0.2f"
          % (lineargrid.best_params_, lineargrid.best_score_))


    with open('output3.csv',"w") as f:
        f.write("{},{},{}\n".format("svm_linear",  lineargrid.best_score_,lineargrid.score(Xtest, ytest)) )




def svm_polynomial(Xtrain,Xtest,ytrain, ytest):
    C_range_poly = [0.1, 1, 3]
    degree_range_poly = [4, 5, 6]
    gamma_range_poly = [0.1, 1]
    param_grid_poly = dict(gamma=gamma_range_poly, C=C_range_poly,degree =degree_range_poly)
    grid_poly = GridSearchCV(svm.SVC(kernel='poly'), param_grid=param_grid_poly, cv=5)
    grid_poly.fit(Xtrain, ytrain)
    print("Poly kernal::The best parameters are %s with a score of %0.2f"
          % (grid_poly.best_params_, grid_poly.best_score_))
    with open('output3.csv',"a") as f:
        f.write("{},{},{}\n".format("svm_polynomial",  grid_poly.best_score_,grid_poly.score(Xtest, ytest)) )



def svm_rbf(Xtrain,Xtest,ytrain, ytest):
    C_range = [0.1, 0.5, 1, 5, 10, 50, 100]
    gamma_range = [0.1, 0.5, 1, 3, 6, 10]
    param_grid = dict(gamma=gamma_range, C=C_range)
    grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, cv=5)
    grid.fit(Xtrain, ytrain)
    print("RBF kernal::The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))
    with open('output3.csv',"a") as f:
        f.write("{},{},{}\n".format("svm_rbf",  grid.best_score_, grid.score(Xtest, ytest)) )

def logistic(Xtrain,Xtest,ytrain, ytest):
    ###### Logistic Regression
    C_range_Logistic = [0.1, 0.5, 1, 5, 10, 50, 100]
    param_grid_Logistic = dict(C=C_range_Logistic)
    LR=LogisticRegression()
    grid_LR = GridSearchCV(estimator=LR, param_grid=param_grid_Logistic, cv=5)
    grid_LR.fit(Xtrain, ytrain)
    print("Logistic kernal::The best parameters are %s with a score of %0.2f"
          % (grid_LR.best_params_, grid_LR.best_score_))
    with open('output3.csv',"a") as f:
        f.write("{},{},{}\n".format("logistic",  grid_LR.best_score_,grid_LR.score(Xtest, ytest)) )


def knn(Xtrain,Xtest,ytrain, ytest):
    ##### KNN
    kNN = KNeighborsClassifier()
    param_grid_KNN = {'n_neighbors': np.arange(1, 51), 'leaf_size': np.arange(5, 65, 5)}
    grid_kNN = GridSearchCV(estimator=kNN, param_grid=param_grid_KNN, cv=5)
    grid_kNN.fit(Xtrain, ytrain)
    print("kNN kernal::The best parameters are %s with a score of %0.2f"
          % (grid_kNN.best_params_, grid_kNN.best_score_))
    with open('output3.csv',"a") as f:
        f.write("{},{},{}\n".format("KNN",  grid_kNN.best_score_,grid_kNN.score(Xtest, ytest)) )


def decision_tree(Xtrain,Xtest,ytrain, ytest):
    ####decision tree
    DT = DecisionTreeClassifier()
    param_grid_decisiontree = {'max_depth': np.arange(1, 51), 'min_samples_split': np.arange(2, 11)}
    grid_DT = GridSearchCV(estimator=DT, param_grid=param_grid_decisiontree, cv=5)
    grid_DT.fit(Xtrain, ytrain)
    print("decision tree kernal::The best parameters are %s with a score of %0.2f"
          % (grid_DT.best_params_, grid_DT.best_score_))
    with open('output3.csv',"a") as f:
        f.write("{},{},{}\n".format("decision_tree",  grid_DT.best_score_,grid_DT.score(Xtest, ytest)) )



def random_forest(Xtrain,Xtest,ytrain, ytest):
    ####Random forest
    RF = RandomForestClassifier()
    param_grid_RF = {'max_depth': np.arange(1, 51), 'min_samples_split': np.arange(2, 11)}
    grid_RF = GridSearchCV(estimator=RF, param_grid=param_grid_RF, cv=5)
    grid_RF.fit(Xtrain, ytrain)
    print("Random forest ::The best parameters are %s with a score of %0.2f"
          % (grid_RF.best_params_, grid_RF.best_score_))
    with open('output3.csv',"a") as f:
        f.write("{},{},{}\n".format("random_forest",  grid_RF.best_score_,grid_RF.score(Xtest, ytest)) )



def main():
    npread_data = np.loadtxt('input3.csv', skiprows=1, delimiter=',')

    X_2d = np.array(npread_data[:, :2])
    label = np.array(npread_data[:, [2]])
    # raw_data = pd.read_csv('input3.csv', header=None)
    # raw_data.columns = ['A', 'B', 'Label']
    A, B, label = npread_data[:, [0]], npread_data[:, [1]], npread_data[:, [2]]
    c, r = label.shape
    label = label.reshape(c, )
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        npread_data[:, [0, 1]], npread_data[:, 2], test_size=0.4,random_state=0,stratify=npread_data[:, 2])

    """X_train=scale(X_train)
    X_test =scale(X_test)
    y_train=scale(y_train)
    y_test=scale(y_test)"""



    svm_linear(X_train, X_test, y_train, y_test)

    svm_polynomial(X_train, X_test, y_train, y_test)
    svm_rbf(X_train, X_test, y_train, y_test)
    logistic(X_train, X_test, y_train, y_test)
    knn(X_train, X_test, y_train, y_test)
    decision_tree(X_train, X_test, y_train, y_test)
    random_forest(X_train, X_test, y_train, y_test)






if __name__ == '__main__':
    main()