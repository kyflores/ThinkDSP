#DCT things tho
import sys
sys.path.append('../code/')

import thinkdsp
import thinkplot
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt

def dct_comp(wave,s_l,thresh):
    """
    wave is a thinkdsp wave object we want to compress
    s_l is the segment length
    """
    assert type(wave)==thinkdsp.Wave
    ys_all=np.array_split(wave.ys,s_l)
    dcts=[]
    for a in range(len(ys_all)):
        dcts.append(scipy.fftpack.dct(ys_all[a].astype(np.float64)))
        if len(dcts[a])<=thresh:
            pass
        elif len(dcts[a]>thresh):
            dcts[a]=dcts[a][0:thresh]

        dcts[a]=scipy.fftpack.idct(dcts[a])

    ys_o=np.concatenate(dcts)
    return thinkdsp.Wave(ys_o,wave.framerate)



if __name__ == '__main__':
    wave=thinkdsp.read_wave('../code/100475__iluppai__saxophone-weep.wav')
    plt.plot(wave.ys)
    comp=dct_comp(wave,128,100)
    plt.plot(comp.ys)
    plt.show()
