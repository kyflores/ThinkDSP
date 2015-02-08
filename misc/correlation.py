# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 13:24:43 2015

@author: kyle
"""
import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl
from pyopencl.reduction import ReductionKernel
import pyopencl.array as arr
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

def cl_setup():
    print cl.get_platforms()
    a=int(raw_input("Enter Platform index"))
    device=cl.get_platforms()[a].get_devices()[0] #0=NV, 1=Beignet, 2=CPU
    ctx=cl.Context([device]) #Just grabs platform 0, whatever it is.
    queue=cl.CommandQueue(ctx)
    mf=cl.mem_flags
    kernel=file('correlation.c').read()
    prg=cl.Program(ctx,kernel)
    prg.build()
    return (ctx,queue,mf,prg)

def test_sum():
    ctx,queue,mf,prg=cl_setup()
    v_h=np.arange(10,dtype=np.float32)
    r_h=np.zeros(1,dtype=np.float32)

    v_d=cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=v_h)
    r_d=cl.Buffer(ctx,mf.WRITE_ONLY,r_h.nbytes)
    prg.sum(queue, v_h.shape, None, v_d, r_d, np.int32(10))
    print r_h
    cl.enqueue_copy(queue, r_h, r_d)
    print r_h
    print sum(v_h)


def cl_corr(x_h,y_h,lag,ctx,queue,mf,prg):
    """
    _h: host memory
    _d: device memory

    Assumes x_h and y_h are the same size...
    """

    ilen=len(x) #length of input, assumes that len(x)=len(y).
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
        p=arr.Array(queue,(ilen,),dtype=np.float32,data=res_p)
        n=arr.Array(queue,(ilen,),dtype=np.float32,data=res_n)
        cor_p[l]=arr.sum(p).get()
        cor_n[l]=arr.sum(n).get()
    return np.concatenate((cor_n[::-1][:len(cor_n)-1],cor_p))

if __name__=='__main__':
    (ctx,queue,mf,prg)=cl_setup()
    n=30000
    x=np.random.rand(n).astype(np.float32)
    y=np.random.rand(n).astype(np.float32)
    t0=time.time()
    corr(x,y,n)
    a=time.time()-t0
    t1=time.time()
    cl_corr(x,y,n,ctx,queue,mf,prg)
    b=time.time()-t1
    print("Input size is " + str(n))
    print("Single Threaded Time")
    print a
    print("OpenCL Time")
    print b
    print("Performance Ratio")
    print(str(a/b) + " X")
