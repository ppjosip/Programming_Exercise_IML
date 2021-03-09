import numpy as np
import pandas as pd
import matplotlib as plt
import csv


#ToDo:
# read in data --> read csv. with a built in fct, a matrix with all X values and y values
# for loop which computes ten fold cross validation
# in the loop --> compute the W_hat (the estimated w) with the normal equation and the different lambdas
#

# Read in the Data using pandas
data = pd.read_csv('train.csv')

#Initialize Data
y_data = data['y']
x_data = data.drop(columns=['y'])
solution = np.zeros((5,1))

X = x_data.to_numpy()
Y = y_data.to_numpy()

RMSE = np.zeros((10,5))
Lambda = [0.1, 1,10,100,200]
for i in range(10):
    drop = range(i*15,(i+1)*15);

    X_Validation = X[drop,:]
    X_Train = np.delete(X,drop,0)
    Y_Validation = Y[drop]
    Y_Train = np.delete(Y, drop, 0)

    X_Train_transp = np.transpose(X_Train)
    Xt_X = np.matmul(X_Train_transp,X_Train)
    I = np.eye(len(Xt_X))

    for j in range(5):
        lam = Lambda[j]
        X_inv = np.linalg.inv(Xt_X + lam*I)
        w_lambda = np.matmul(np.matmul(X_inv,X_Train_transp),Y_Train)
        y_hat = np.matmul(X_Validation,w_lambda)
        y_y_hat_sq = np.square(Y_Validation-y_hat)
        RMSE[i,j] = np.sqrt((1/15)*np.sum(y_y_hat_sq))

RMSE_AVG = np.mean(RMSE, axis=0)

test = pd.DataFrame(RMSE_AVG, columns=['RMSE'])
test.to_csv("./RMSEoutput.csv", sep=str(','), index=None, header=None)