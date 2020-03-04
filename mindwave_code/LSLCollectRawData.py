from __future__ import print_function

import mindwave, time
from pprint import pprint
from pylsl import StreamInfo, StreamOutlet

import socket,select

import time, datetime, sys

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import sys


class MindwaveLSLRecorder: 
    def __init__(self): 
        # Create eeg outlet [raw eeg, attention, meditation, blink]
        info_eeg = StreamInfo(name='Mindwave EEG', 
            type='EEG', channel_count=4, nominal_srate=128, 
            channel_format='float32', source_id='eeg_thread')
        self.__eeg_outlet = StreamOutlet(info_eeg)
        self.currentTimestamp = None
        self.__Fs = 128 # 128Hz 

    def __on_raw(self, headset, rawvalue):
        (eeg, attention, meditation, blink) = (headset.raw_value, headset.attention, headset.meditation, headset.blink)

        self.currentTimestamp = time.time()
        self.currentRawValue = eeg
        self.currentAttention = attention
        self.currentMeditation = meditation
        self.currentBlink = blink

    def run(self):
        print("Connecting")
        headset = mindwave.Headset('/dev/tty.MindWaveMobile-SerialPo')
        time.sleep(2)
        print("Connected!")

        try:
            while (headset.poor_signal > 5):
                print("Headset signal noisy %d. Adjust the headset and the earclip." % (headset.poor_signal))
                time.sleep(0.1)
                
            print("Writing output to LSL stream" )
            stime = time.time()
            headset.raw_value_handlers.append( self.__on_raw )
            prevTime = 0
            while True:
                if headset.poor_signal > 5 :
                    print("Headset signal noisy %d. Adjust the headset and the earclip." % (headset.poor_signal))

                if self.currentTimestamp is not None: 
                    self.__eeg_outlet.push_sample(np.array([self.currentRawValue, self.currentAttention, self.currentMeditation, self.currentBlink]))

                timeDiff = int(time.time()-stime)
                if(timeDiff != prevTime) : 
                    print("seconds elapsed: " + str(timeDiff))
                    prevTime = timeDiff
                time.sleep(1/self.__Fs)

        finally:
            
            # df = pd.DataFrame.from_dict(sampled_data)
            # #df.sort_values(by=['timestamp'])
            # df.to_csv(filename, index=False)

            # df = pd.DataFrame.from_dict(data)
            # df.to_csv(filename_listener, index=False)
            print("Closing!")
            headset.stop()

if __name__=="__main__":
    mlslr = MindwaveLSLRecorder()
    mlslr.run()