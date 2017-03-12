import argparse
import math
import operator
import numpy as np
from math import*
import matplotlib.pyplot as plt
#import scipy.fftpack

from pythonosc import dispatcher
from pythonosc import osc_server


class eegData:
    def __init__(self):
        self.chan1 = []
        self.chan2= []
        self.chan3 = []
        self.chan4 = []
        self.n = 0
        self.nPnts = 600
        self.sr = 220


eeg1 = eegData()


def eeg_handler(unused_addr, args, ch1, ch2, ch3, ch4):

    global eeg1
    #print("EEG (uV) per channel: ", ch1, ch2, ch3, ch4)

    eeg1.chan1.append(ch1)
    eeg1.chan2.append(ch2)
    eeg1.chan3.append(ch3)
    eeg1.chan4.append(ch4)
    print(str(eeg1.n))
    #if (eeg1.n % eeg1.nPnts == 0):
    if (eeg1.n == eeg1.nPnts - 1):
        eeg1.n = eeg1.n + 1
        z = 1j  #imag number
        #return eeg1
        time = np.arange(-2,2,1/eeg1.sr,) #define domain for sinusoid
        sin10 = []
        for x in time:
            sin10.append(np.exp(z*2*np.pi*10*x)) #create complex sine wave at alpha freq
        stdv = 7 / (2*3.14159*10)         #set parameter for morlet wavelet
        gaus_window = []
        for x in time:
            gaus_window.append(exp(-np.power(x,2) / (2*np.power(stdv,2)))) #create gaussian curve
        wavelet = []
        for x in range(len(sin10) - 1):
            wavelet.append(sin10[x] * gaus_window[x])    #multiply sin and gaussian to get wavelet
        tempChan2 = np.array(eeg1.chan2)
        nData = len(tempChan2)
        nKern = len(wavelet)
        nConv = nData + nKern - 1
        half_wav = floor(len(wavelet)/2.0 + 1)

        waveletX = np.fft.fft(wavelet)  #freq domain wavelet
        if nConv - len(waveletX) > 0:
            tempZ = np.zeros(nConv - len(waveletX))
            waveletX = np.concatenate((waveletX,tempZ))
        index, waveletMaxAbs = max(enumerate(abs(waveletX)), key=operator.itemgetter(1))
        waveletMax = waveletX[index]
        waveletXn = [x / waveletMax for x in waveletX] #normalize freq domain
        dataX = np.fft.fft(tempChan2)  #freq domain data
        if nConv - len(dataX) > 0:
            tempZ = np.zeros(nConv - len(dataX))
            dataX = np.concatenate((dataX,tempZ))
        convResult1 = []
        for x in range(len(dataX)):
            convResult1.append(dataX[x] * waveletXn[x])
        alpha1 = np.fft.ifft(convResult1)
        alpha1Fin = alpha1[half_wav-1 : len(alpha1) - half_wav - 1]

        signal2 = np.convolve(eeg1.chan3, wavelet)  #comb for alpha freq in eeg chan 3
        alpha1P = np.power(abs(alpha1),2) #find power in alpha band
        dataTimes = np.arange(0, eeg1.nPnts / eeg1.sr, 1/ eeg1.sr)
        # wave1 = np.fft.fft(eeg1.chan2)
        #wave2 = np.fft.fft(eeg1.chan3)
        #freq1 = np.fft.fftfreq(len(eeg1.nPnts), 1 / eeg1.sr)
        #freq1 = freq1[:eeg1.nPnts//2]
        #wave1 = 2/eeg1.nPnts*np.abs(wave1[:eeg1.nPnts//2])

        with open('dontTestMe.txt', 'w') as f:
            f.write(" Time = ")
            f.write(str(dataTimes))
            f.write("alpha = ")
            f.write(str(alpha1Fin))
        print("Jimmy is a bean")
        print(str(eeg1.n))
        print(str(eeg1.nPnts))
        plt.plot(dataTimes, alpha1Fin)
        plt.show()
        exit()

        #print(wave1)
    else:
        eeg1.n = eeg1.n + 1
    return eeg1
    #print("X = ", s.x, ", Y = ", s.y, ", Z = ", s.z)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip",
                        default="127.0.0.1",
                        help="The ip to listen on")
    parser.add_argument("--port",
                        type=int,
                        default=5000,
                        help="The port to listen on")
    args = parser.parse_args()

    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/debug", print)
    dispatcher.map("/muse/eeg", eeg_handler, "EEG")

    server = osc_server.ThreadingOSCUDPServer(
        (args.ip, args.port), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()