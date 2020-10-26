__author__ = "Nikolaos Stylianou"
__version__ = "1.0"
__email__ = "nstylia@csd.auth.gr"

"""
Test script to be used after the installation of all the packages required by this course. Running without any errors should provide
the information of the current versions of the libraries installed. If an error occurs, it means that a library failed to load and hence 
the user should attempt to fix the issue in order to prepare his system for the course. 
This document has been produced as part of the MSc Course on the subject of Machine Learning for the Aristotle University of Thessaloniki.
Last update: 24/10/2017
"""

print('System dependency checks')
# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

print('System ready')