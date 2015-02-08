//cl kernel file for correlation program

__kernel void multiply(__global const float *x_d,
                      __global const float *y_d, 
                      __global float *res_p,
                      __global float *res_n,
                      const unsigned int l)
{
    //int l=get_global_id(0);
    int i = get_global_id(0);

    res_p[i]=x_d[l+i]*y_d[i];
    res_n[i]=y_d[l+i]*x_d[i];
}

