# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 13:24:43 2015

@author: kyle
"""
import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl
import pyopencl.array as arr
from pyopencl.elementwise import ElementwiseKernel
import time

def corr(x,y,lag):
    """
    x and y are numpy arrays of values. They do not need to be the same length.
    lag is the maximum number of indicies x can be shifted relative to y.
    This function should have similar behavior to MATLAB's xcorr
    """
    if len(x)<len(y):
        x=np.pad(x,(0,len(y)),'constant')
    elif len(y)<len(x):
        y=np.pad(y,(0,len(x)),'constant')

    cor_p=np.empty(lag,dtype='float')
    cor_n=np.empty(lag,dtype='float')

    for l in range(lag):
        cor_p[l]=sum(x[:(len(x)-l)] * y[l:])
        cor_n[l]=sum(y[:(len(y)-l)] * x[l:])

    return np.concatenate((cor_p[::-1][:len(cor_p)-1],cor_n))

def cl_setup(a):
    #print cl.get_platforms()
    #a=int(raw_input("Enter Platform index"))
    device=cl.get_platforms()[a].get_devices()[0] #0=NV, 1=Beignet, 2=CPU
    ctx=cl.Context([device]) #Just grabs platform 0, whatever it is.
    queue=cl.CommandQueue(ctx)
    mf=cl.mem_flags
    return (ctx,queue,mf)

"""
def cl_corr(x_h,y_h,lag,ctx,queue,mf,prg):

    ilen=len(x_h) #length of input, assumes that len(x)=len(y).
    #ctx,queue,mf,prg=cl_setup()
    print(ctx)

    cor_p=np.empty(lag,dtype=np.float32)
    cor_n=np.empty(lag,dtype=np.float32)

    #Generate input memory buffers for x and y signals.
    x_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_h.astype(np.float32))
    y_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_h.astype(np.float32))

    #Generate result memory buffers for the dot products of x and y.
    res_p = cl.Buffer(ctx, mf.WRITE_ONLY, np.empty(ilen,dtype=np.float32).nbytes)
    res_n = cl.Buffer(ctx, mf.WRITE_ONLY, np.empty(ilen,dtype=np.float32).nbytes)

    for l in range(lag):
        prg.multiply(queue, x_h.shape, None, x_d, y_d, res_p, res_n, np.int32(l))
        p=cl.array.Array(queue,(ilen,),dtype=np.float32,data=res_p)
        n=cl.array.Array(queue,(ilen,),dtype=np.float32,data=res_n)
        cor_p[l]=cl.array.sum(p).get()
        cor_n[l]=cl.array.sum(n).get()
    return np.concatenate((cor_n[::-1][:len(cor_n)-1],cor_p))
"""


def corr_cl(x_h,y_h,lag,ctx,queue,mf):
    size=len(x_h)
    x_d=arr.to_device(queue,x_h)
    y_d=arr.to_device(queue,y_h)

    cor_p=np.empty(lag,dtype='float')
    cor_n=np.empty(lag,dtype='float')

    for l in range(lag):
        cor_p[l]=arr.sum(arr.dot(x_d[:(size-l)], y_d[l:])).get()
        cor_n[l]=arr.sum(arr.dot(y_d[:(size-l)], x_d[l:])).get()

    return np.concatenate((cor_p[::-1][:len(cor_p)-1],cor_n))


if __name__=='__main__':
    (ctx,queue,mf)=cl_setup(0)
    n=10000
    x=np.random.rand(n).astype(np.float32)
    y=np.random.rand(n).astype(np.float32)

    t0=time.time()
    print corr_cl(x,y,n,ctx,queue,mf)
    dt0=time.time()-t0

    t1=time.time()
    print corr(x,y,n)
    dt1=time.time()-t1

    print "CL platform is "+str(ctx)
    print "INPUT SIZE:"+str(n)
    print "CL TIME:"+str(dt0)
    print "CPU_TIME:"+str(dt1)
    print "PERFORMANCE RATIO:"+str(dt1/dt0)