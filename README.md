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

WisardClassifier core functions and memory management is implemented in C++.
(sources are in the <code>wislib</code> directory).

----------------------
Build and setup (Linux, Mac OSX)
----------------------

To build the code and setup in python (locally) you just need to execute:

```bash
$ python setup.py build_ext --inplace
```

This will produce the WiSARD wrapper library wit name: <code>wisard_wrapper.cython-arch.dllext</code> 
where <code><arch></code> is the target architecture and <code>dllext</code> is the library extension (<code.so</code> for Mac and Linux).


----------------------
Testing
----------------------

To use the WisardClassifier in your Python scripts you need to have
python 2.7 (or later) installed on your system, plus the following
modules:

1. Numpy (http://www.numpy.org)

2. Cython (http://cython.org) 

3. matplotlib (optional) (https://matplotlib.org)

Please refer to the documentation of these packages for installation.

Once you have set the python programming framework, you can use the file <code>test.py</code> simple
script to start using WiSARD.

```python
from wisardwrapper import *
import numpy as np

def mk_tuple(discr, sample):
    intuple = np.zeros(discr.contents.n_ram, dtype = np.uint64)
    for i in range(discr.contents.n_ram):
        for j in range(discr.contents.n_bit):
            x = discr.contents.map[(i * discr.contents.n_bit) + j]
            intuple[i] += (2**(discr.contents.n_bit -1 - j))  * sample[x]
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
wisard["A"] = make_discr(2,8,"random",0)
wisard["B"] = make_discr(2,8,"random",0)

# train WiSARD
for s in range(X.shape[0]):
    tuple = mk_tuple(wisard[y[s]],X[s])
    train_discr(wisard[y[s]],tuple)

# print WiSARD state
print_discr(wisard["A"]);
print_discr(wisard["B"]);
    
# predict by WiSARD
responses = {}
test_tuple = mk_tuple(wisard["A"],test)
responses["A"] = classify_discr(wisard["A"],test_tuple);
test_tuple = mk_tuple(wisard["B"],test)
responses["B"] = classify_discr(wisard["B"],test_tuple);
print("A responds with score %.2f\n"%responses["A"]);
print("B responds with score %.2f\n"%responses["B"]);
```


To use WisardClassifier in Scikit Learn Python library 
you need the following packages:

1. Scipy (https://www.scipy.org)

2. Scikit Learn (http://scikit-learn.org)

Please refer to the documentation of these packages for installation.

In this package a python script <code>wisard_classifier.py</code> is distributed to let users 
execute WisardClasifier in python 

```bash
$ python wisard_classifier.py -i <dataset>
```

Where <code>\<dataset\></code> can be any data file in ARFF, LIBSVM, or CSV format.
    
