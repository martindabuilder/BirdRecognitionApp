""" As this is a one man project, to make it easier on my laptop im adding a parser class
Its job will be to read and pass the data to the model training class one at a time, 
making it lighter on memory usage """

import os
import glob
import numpy as np
import tensorflow as tf

mel_lines = 150
spectrogram_width = 128