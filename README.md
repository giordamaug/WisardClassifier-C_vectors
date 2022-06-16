# WisardClassifier
Machine learning supervised method for classification using WiSARD

> Authors: Maurizio Giordano and Massimo De Gregorio
> - Istituto di Calcolo e Reti ad Alte Prestazioni (ICAR) - Consiglio Nazionale delle Ricerche (CNR) (Italy)
> - Istituto di Scienze Applicate e Sistemi Intelligenti "Eduardo Caianiello" (ISASI) - Consiglio Nazionale delle Ricerche (CNR) (Italy)

----------------------
Description
----------------------

WisardClassifier is a machine learning classifer implemented as an exntension module of
the scikit-learn package in Python.
As a consequence, to use WisardClassifier you need the following packages installed in your
Python environment:

1) Numpy

2) Scikit-Learn

WisardClassifier is a classifier method for machine learning accepting as input most common 
data formats, such as ARFF, LibSVM, and CSV.

WisardClassifier uses  WiSARD, a weighless neural network model of computation which
learns and recognize binary inpiut patterns. For more details about this model please refer (and cite) to:

```
@article{DEGREGORIO2018338,
title = "An experimental evaluation of weightless neural networks for multi-class classification",
journal = "Applied Soft Computing",
volume = "72",
pages = "338 - 354",
year = "2018",
issn = "1568-4946",
doi = "https://doi.org/10.1016/j.asoc.2018.07.052",
url = "http://www.sciencedirect.com/science/article/pii/S156849461830440X",
author = "Massimo De Gregorio and Maurizio Giordano",
keywords = "Weightless neural network, WiSARD, Machine learning"
}
```

WiSARD implementation in C++ is included in sources in this distribution.
The file <code>wisard_wrapper.pyx</code> is a cython wrapper for the integration
WiSARD functions in Python. You can use and program WiSARD in Python 
by simply importing in your python scripts the <code>wisard_wrapper</code> module.
An example of the use of WiSARD in Python is discussed in the section "Testing WiSARD in Python".

WisardClassifier is a Python implementation of a classifier based on WiSARD capabilities. 
The classifier is designed to be compliant with the Scikit-Learn interface definition.
WisardClassifier main class, methods and facilities are implemented in the <code>wisard.py</code> 
script.

----------------------
Build and setup (Linux, Mac OSX)
----------------------

To build the code and setup in python (locally) you just need to execute:

```bash
$ python setup.py build_ext --inplace
```

This will produce the WiSARD wrapper library wit name: <code>wisard_wrapper.\<dllext\></code> 
where <code>\<dllext\></code> is the library extension (<code>so</code> for Mac and Linux, <code>dll</code> for Windows).


----------------------
Testing WiSARD in Python
----------------------

To use the WiSARD in your Python scripts you need to have
python 2.7 (or later) installed on your system, plus the following
modules:

1. Numpy (http://www.numpy.org)

2. Cython (http://cython.org) 

3. matplotlib (optional) (https://matplotlib.org)

Please refer to the documentation of these packages for installation.

Once you have set the python programming framework, you can use the file <code>test.py</code> simple
script to start using WiSARD in python.

```python
from wisard_wrapper import *
import numpy as np

def mk_tuple(discr, sample):
    map = discr.getMapping()
    n_bit = discr.getNBits()
    n_ram = discr.getNRams()
    intuple = np.zeros(n_ram, dtype = np.uint64)
    for i in range(n_ram):
        for j in range(n_bit):
            x = map[(i * n_bit) + j]
            intuple[i] += (2**(n_bit -1 - j))  * sample[x]
    return intuple


X = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 1, 1, 0, 0],
              [0, 0, 1, 0, 0, 0, 1, 0],
              [1, 0, 0, 0, 0, 0, 0, 1],
              [1, 1, 0, 1, 1, 1, 1, 1],
              [1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 1]], np.int32)

y = np.array(["A","A","B","B","A","A","B","A"])

test = np.array([0, 0, 1, 0, 0, 0, 1, 0], np.int32)

# init WiSARD (create discriminator for each class "A" and "B")
wisard = {}
wisard["A"] = PyDiscriminator(2,8)
wisard["B"] = PyDiscriminator(2,8)

# train WiSARD
for s in range(X.shape[0]):
    tuple = mk_tuple(wisard[y[s]],X[s])
    print(tuple)
    wisard[y[s]].TrainByTuple(tuple)

# print WiSARD state
print(wisard["A"].toString())
print(wisard["B"].toString())
    
# predict by WiSARD
responses = {}
test_tuple = mk_tuple(wisard["A"],test)
responses["A"] = wisard["A"].ClassifyByTuple(test_tuple)
test_tuple = mk_tuple(wisard["B"],test)
responses["B"] = wisard["B"].ClassifyByTuple(test_tuple)
print("A responds with score %.2f\n"%responses["A"])
print("B responds with score %.2f\n"%responses["B"])
```

----------------------
Testing WisardClassifier in Python (Scikit Learn)
----------------------

To use WisardClassifier in Scikit Learn Python library 
you need the following packages:

1. Scipy (https://www.scipy.org)

2. Scikit Learn (http://scikit-learn.org)

Please refer to the documentation of these packages for installation.

In this package a python script <code>wisard_classifier.py</code> is distributed to let users 
execute WisardClassifier in python 

```bash
$ python wisard_classifier.py -i <dataset>
```

Where <code>\<dataset\></code> can be any data file in ARFF, LibSVM, or CSV format.

For a complete list of script parameters you can read the command help:

```bash
python2.7 wisard_classifier.py -h
```

----------------------
Testing WisardClassifier on KEEL datasets
----------------------

You can test WisardClassifier on a large set of classification problems you 
find in the <code>keel-datasets</code> folder. Problem datasets were obtained from the
KEEL archive (http://sci2s.ugr.es/keel/datasets.php) and transformed in LibSVM format.

As an example, to test WisardClassifier on the 'wine' dataset with 32 bits and with a resolution of data discretization
of 2018 distinct values, simply run:

```bash
$ python wisard_classifier.py -b 32 -z 2018 -i KEEL_datasets/wine.libsvm
```
    
