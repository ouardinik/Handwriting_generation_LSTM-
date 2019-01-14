import tensorflow as tf
import numpy as np


# takes output, and computes loss function of the gaussian mixture model


def mix_coef(output):
    # here the output is a ...
    e = output[:,0:1] #end of stroke values
    param_e = tf.sigmoid(e)
    
    pi, mu_x, mu_y, std_x, std_y, rho = tf.split(output[:,1:], axis=1, num_or_size_splits=6)
    
    # compute softmax of pi
    pi = tf.exp(pi)
    pi = tf.div(pi,tf.reduce_max(pi,1,keepdims=True))
   
    #compute std
    std_x = tf.exp(std_x)
    std_y = tf.exp(std_y)
    
    #compute rho
    rho = tf.tanh(rho)
    
    return pi, mu_x, mu_y, std_x, std_y, rho, param_e


def gmm2d(x, y, mu_x, mu_y, std_x, std_y, rho):
    # computes var
    var1 = tf.subtract(x,mu_x)
    var2 = tf.subtract(y,mu_y)
    var3 = -2*tf.multiply(rho,tf.multiply(var1,var2))
    
    prod_std = tf.multiply(std_x,std_y)
    inv_rho = 1 - tf.square(rho)
    
    var1 = tf.div(tf.square(var1),tf.square(std_x))
    var2 = tf.div(tf.square(var2),tf.square(std_y))
    var3 = tf.div(var3,prod_std)
    
    var = var1 + var2 + var3
    var = tf.exp(tf.div(-var,2*inv_rho))
    var = tf.div(var,2*np.pi*tf.multiply(prod_std,tf.sqrt(inv_rho)))
    
    return var 

def gmm_loss( e, x, y, pi, mu_x, mu_y, std_x, std_y, rho, param_e):
    var = gmm2d(x, y, mu_x, mu_y, std_x, std_y, rho)
    epsilon = 1e-20
    var = tf.multiply(pi,var)
    var = tf.reduce_sum(var,1,keepdims=True)
    var =  -tf.log(tf.maximum(var, 1e-20))
    if e == 1:
        var = tf.add(var, -tf.log(param_e))
    if e == 0:
        var = tf.add(var,-tf.log(1-param_e))            
    loss = tf.reduce_sum(var)
    return loss


    
    
    
    
    
    