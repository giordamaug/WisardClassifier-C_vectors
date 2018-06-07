# The WiSARD library
A C++ implementation of the WiSARD weightless neural network.
> Authors: Maurizio Giordano and Massimo De Gregorio
> - Istituto di Calcolo e Reti ad Alte Prestazioni (ICAR) - Consiglio Nazionale delle Ricerche (CNR) (Italy)
> - Istituto di Scienze Applicate e Sistemi Intelligenti "Eduardo Caianiello" (ISASI) - Consiglio Nazionale delle Ricerche (CNR) (Italy)

----------------------
Description
----------------------

WiSARD stands for "Wilkie, Stonham, Aleksander Recognition Device". 
It is a weightless neural network model to recognize binary patterns.
For a introduction to WiSARD, please read <a href="https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2009-6.pdf">A brief introduction to Weightless Neural Systems</a>

This software is an efficient C++ library implementation of WiSARD
basic components and functions. It can be used in any C++ application.

The C++ WiSARD library is distributed together with a Python interface,
called WiSARDpy, to use WiSARD in Python programming with 
fast training/classification time.

----------------------
Citation Details
----------------------
  
If you use this library, please cite the followong work

M. De Gregorio, M. Giordano.
"The WiSARD classifier"
In: Proceedings of 24th European Symposium on Artificial Neural Networks, 
Computational Intelligence and Machine Learning, ESANN 2016; Bruges; Belgium

Bibtex:

```
@CONFERENCE{DeGregorio2016,
author={De Gregorio, M. and Giordano, M.},
title={The WiSARD classifier},
journal={ESANN 2016 - 24th European Symposium on Artificial Neural Networks},
year={2016},
pages={447-452},
url={https://www.scopus.com/inward/record.uri?eid=2-s2.0-84994165233&partnerID=40&md5=c77502db0e36746bf85293361cb1f122},
document_type={Conference Paper},
source={Scopus},
}
```

----------------------
License
----------------------
  
The source code is provided without any warranty of fitness for any purpose.
You can redistribute it and/or modify it under the terms of the
GNU General Public License (GPL) as published by the Free Software Foundation,
either version 3 of the License or (at your option) any later version.
A copy of the GPL license is provided in the "GPL.txt" file.

----------------------
Compile source (Linux, Mac OSX)
----------------------

To run the code the following libraries are required:

2. CMake  2.8  (later version may also work)

3. C++ Compiler (tested only with GCC 5.x or later versions)

```
$ cmake .
$ make
```


----------------------
WiSARD in Python
----------------------

To use WiSARD in your Python scripts you need to have
python 2.7 (or later) installed on your system, plus the following
modules:

1. Numpy

2. Cython

Once you have set the python programming framework, you can use the following simple
script to start using WiSARD (that can be found in folder WiSARDpy).

```python
from wisard import *

# dataset is list of list, or 2-dimensional numpy array
X = np.array(
   [ [0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 1, 1, 1, 0, 0],
     [0, 0, 1, 0, 0, 0, 1, 0],
     [1, 0, 0, 0, 0, 0, 0, 1],
     [1, 1, 0, 1, 1, 1, 1, 1],
     [1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 1]])

# label set is a list, or a numpy array
y = np.array(['A','A','B','B','A','A','B','A',])

# create wisard object (with 2 bit resolution)
w = WiSARD(2)
# create/train the wisard object
w.fit(X, y)

# classify by wisard
result = w.predict(X)
# classify by enabling bleaching
w.setBleaching()
result_b = w.predict(X)
```

Note that the bleaching implementation of the WiSARD is included in the python wrapper 
of our WiSARD Library, not in the library itself. 
This implementation is very similar to the one implemented in the https://github.com/firmino/PyWANN 
software distribution, which. Both implementations refer to the work:

Danilo S. Carvalho, Hugo C. C. Carneiro, Felipe M. G. Franca, Priscila M. V. Lima.
"B-bleaching: Agile Overtraining Avoidance in the WiSARD Weightless Neural Classifier"
In: Proceedings of 21st European Symposium on Artificial Neural Networks (ESANN 2013) - ISBN 978-2-87419-081-0. 
Available from http://www.i6doc.com/en/livre/?GCOI=28001100131010.
