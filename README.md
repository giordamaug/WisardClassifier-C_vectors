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

This will produce the WiSARD wrapper library wit name: <code>wisard_wrapper.cpython-<arch>.dllext</code> 
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

-------------------------
WiSARD in Scikit Learn
-------------------------

To use WiSARD in Scikit Learn Python library you need the following packages:

1. Scipy (https://www.scipy.org)

2. Scikit Learn (http://scikit-learn.org)

Please refer to the documentation of these packages for installation.

Hereafter we report a Python script <code>test_wis.py</code> as an example of usage of WisardClassifier within the Scikit-Learn
machine learning programming framework. For a more complete example, see file <code>test.py</code>.

```python
# import sklearn and scipy stuff
from sklearn.datasets import load_svmlight_file
from sklearn import cross_validation
import scipy.sparse as sps
from scipy.io import arff
# import wisard classifier library
from wis import WisardClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from utilities import *
import time

# (Try) import matplot for graphics
try:
    import matplotlib.pyplot as plt
    matplotfound = True
except ImportError:
    matplotfound = False
    pass

B_enabled = True
# IRIS (arff) - load datasets
data, meta = arff.loadarff(open("datasets/iris.arff", "r"))
y_train = np.array(data['class'])
X_train = np.array([list(x) for x in data[meta._attrnames[0:-1]]])
X_train = X_train.toarray() if sps.issparse(X_train) else X_train  # avoid sparse data
class_names = np.unique(y_train)
# IRIS (arff) - cross validation example
clf = WisardClassifier(n_bits=16,bleaching=B_enabled,n_tics=256,mapping='linear',debug=True,default_bleaching=3)
kf = cross_validation.LeaveOneOut(len(class_names))
predicted = cross_validation.cross_val_score(clf, X_train, y_train, cv=kf, n_jobs=1)
print("Accuracy Avg: %.2f" % predicted.mean())

# IRIS (libsvm) - load datasets
X_train, y_train = load_svmlight_file(open("datasets/iris.libsvm", "r"))
class_names = np.unique(y_train)
X_train = X_train.toarray() if sps.issparse(X_train) else X_train  # avoid sparse data
# IRIS - cross validation example (with fixed seed)
clf = WisardClassifier(n_bits=16,n_tics=1024,debug=True,bleaching=B_enabled,random_state=848484848)
kf = cross_validation.StratifiedKFold(y_train, 10)
predicted = cross_validation.cross_val_score(clf, X_train, y_train, cv=kf, n_jobs=1, verbose=0)
print("Accuracy Avg: %.2f" % predicted.mean())

# DNA (libsvm) - load datasets
X_train, y_train = load_svmlight_file(open("datasets/dna.tr", "r"))
X_train = X_train.toarray() if sps.issparse(X_train) else X_train  # avoid sparse data
class_names = np.unique(y_train)
X_test, y_test = load_svmlight_file(open("datasets/dna.t", "r"))
X_test = X_test.toarray() if sps.issparse(X_test) else X_test  # avoid sparse data

# DNA (arff) - testing example
clf = WisardClassifier(n_bits=16,n_tics=512,debug=True,bleaching=B_enabled,random_state=848484848,n_jobs=-1)
y_pred = clf.fit(X_train, y_train)
tm = time.time()
y_pred = clf.predict(X_test)
print("Time: %d"%(time.time()-tm))
predicted = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy: %.2f" % predicted)

# DNA - plot (print) confusion matrix
if matplotfound:
    plt.figure()
    plot_confusion_matrix(cm, classes=class_names,title='Confusion matrix')
    plt.show()
else:
    print_confmatrix(cm)
```
