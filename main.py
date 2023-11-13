import os
from python_speech_features import mfcc
from scipy.io import wavfile

path = os.path.abspath(os.curdir) + '/data/TensorFlow-Speech-Commands/cat'
os.chdir(path)

mffc_arr = []
for filename in os.listdir():
    sampling_rate, samples = wavfile.read(filename, 'r')
    mffc_arr.append(mfcc(samples, sampling_rate))

print(mffc_arr)