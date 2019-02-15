import numpy as np
import tensorflow as tf
import json
import os
import sys
import shutil

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
    

if __name__ == '__main__':
    main()