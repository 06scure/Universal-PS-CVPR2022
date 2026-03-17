import os
import sys
import numpy as np

class setup_configuration():
    def __init__(self):
        self.img_channels = 3 # RGB
        self.train_datatype = 'AdobeNPI' # Train Data Type (Fix)
        self.train_prefix= '0*' # only images with this prefix are loaded
        self.train_suffix = 'PNG' # only directories with this suffix are loaded
        self.train_maxNumberOfImages = 10 # max number of training images
        self.test_datatype = 'RealData' # Test Data Type (Fix)
        self.test_prefix = '0*' # only images with this prefix are loaded
        self.test_suffix = 'PNG' # only directories with this suffix are loaded (for DiLiGenT)
        self.test_maxNumberOfImages = 12 # max number of test images (reduce to avoid OOM)
