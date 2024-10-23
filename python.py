import numpy as np
import math

x_train = [[0,0], [0,1], [0,1], [1,1]]
y_train = [0, 1, 1, 1]

W = np.random.uniform(-6/math.sqrt(2+1), 6/math.sqrt(2+1), (1,2))
W = np.asarray(W.ravel())
b=0

lr = 0.01
epochs = 100000
epoch = 1

def activation(x):
    return 1./(1 + np.exp(-x))

while epoch < epochs :
    loss = 0
    for i,x in enumerate(x_train):
        #print("our index : " + str(i) +" our element is x = " , x )

        #forward propagation
        x = np.asarray(x)
        z = np.dot(W,x) + b
        y_hat = activation(z)
        error = y_train[i] - y_hat
        loss += 1/2 * (error)**2

        #backward propagation (update the weights and the bias)
        #w1 = w1 - lr * error * y_hat * (1-y_hat) * x[i][0]
        #w2 = w2 - lr * error * y_hat * (1-y_hat) * x[i][1]

        W = W + lr * error * y_hat * (1-y_hat) * np.asarray(x)
        b = b + lr * error * y_hat * (1-y_hat) * 1
    print(loss)
    epoch = epoch + 1

x_test = [[0,0], [0,1], [0,1], [1,1]]
y_test = [0, 1, 1, 1]

for i in range(len(x_test)):
    y_hat = activation(np.dot(W, x_test[i]) + b)
    print('la valeure reelle : '+ str(y_test[i]) +' et la valeur predite : ' + str(y_hat))


