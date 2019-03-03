import numpy as np
import tensorflow as tf
import json
import os
import sys
import shutil
import matplotlib.pyplot as plt

outputFile = open('el.txt','w')

def Euler(xa, xb, ya, n):
    eachPoint=[]
    h = (xb - xa) / (n)
    x = xa
    y = ya
    for i in range(n):
        y += h * f(x)
        x += h
        eachPoint.append(y)
    return eachPoint

def f(x):
    dxdt=tf.log(x)
    return dxdt


def train_network(input):

    x = input[0][0]
    y = input[0][1]
    # print(type(x))
    # x = tf.reshape(x, [1,x.shape[0]])
    # y = tf.reshape(y, [1,y.shape[0]])
    x = x.reshape(1,x.shape[0])
    y = y.reshape(1,y.shape[0])

    tf_x = tf.placeholder(tf.float32, x.shape)      # input x
    tf_y = tf.placeholder(tf.float32, y.shape)      # input y# neural network layers
    l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)      # hidden layer
    output = tf.layers.dense(l1, 1)    

    # neural network layers
    l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)          # hidden layer

    output = tf.layers.dense(l1, y.shape[1])   

    loss = tf.losses.mean_squared_error(tf_y, output)   # compute cost
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
    train_op = optimizer.minimize(loss) 

    sess = tf.Session()                                 # control training and others
    sess.run(tf.global_variables_initializer())         # initialize var in graph



    for step in range(1000):
    # train and net output
        _, l, pred = sess.run([train_op, loss, output], {tf_x: x, tf_y: y})
        next=sess.run([output],{tf_x:x})
    print(pred[0])
    print(next[0][0])

    for i in range (2,10,1):
        x = input[0][i]
        x = x.reshape(1,x.shape[0])
        next=sess.run([output],{tf_x:x})
        print(next)
    sess.close()


def main():
    xa=tf.constant([1.])
    xb=tf.constant([10.])
    n =tf.constant([10.])
    x= [4.,5,6,7,8,9]
    x_tf=tf.placeholder(dtype=tf.float32,shape=len(x))
    y=Euler(1.0,10.0,x_tf,10)
    a = np.zeros(shape=(len(x),10))

    sess=tf.Session()
    y=sess.run([y],{x_tf:x})
    sess.close()

    for i in range(0,10,1):
        print(y[0][i])
        wline = str(y[0][i]) + "\n"
        outputFile.write(wline)

    outputFile.close()



    train_network(y);
    

if __name__ == '__main__':
    main()