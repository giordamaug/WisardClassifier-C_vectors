#import numpy library
import numpy as np
# import sklearn and scipy stuff
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score
import scipy.sparse as sps
from scipy.io import arff
# import wisard classifier library
from wis import WisardClassifier
#import utilities for matplot
from utilities import *
import time
import argparse
import sys,os
# (Try) import matplot for graphics
try:
    import matplotlib.pyplot as plt
    matplotfound = True
except ImportError:
    matplotfound = False
    pass

B_enabled = True

parser = argparse.ArgumentParser(description='WiSARD Classifier')
parser.add_argument('-i', "--inputfile", metavar='<inputfile>', type=str, help='input file (', required=True)
parser.add_argument('-l', "--labelpos", metavar='<labelpos>', default=-1, type=int, help='label position')
parser.add_argument('-n', "--labelname", metavar='<labelname>', default="class", type=str, help='label name')
parser.add_argument('-b', "--nbits", metavar='<bitno>', type=int, help='number of bits [16]', default=16,required=False)
parser.add_argument('-f', "--cvfolds", metavar='<cvfolds>', type=int, help='number of folds in cv [10]', default=10,required=False)
parser.add_argument('-z', "--ntics", metavar='<ticno>', type=int, help='number of tics [1024]', default=1024,required=False)


def main(argv):
    # parsing command line
    args = parser.parse_args()

    # check dataset format (arff, libsvm)
    datafile = args.inputfile
    if not os.path.isfile(datafile):
        raise ValueError("Cannot open file %s" % datafile)
    if datafile.endswith('.arff'):
        data, meta = arff.loadarff(open(datafile, "r"))
        try:
            y = np.array(data[args.labelname])
            used_attrnames = [m for m in meta._attrnames if m != args.labelname]
            X = np.array([list(x) for x in data[used_attrnames]])
        except:
            raise ValueError("Cannot find label %s in dataset" % args.labelname)
    elif datafile.endswith('.libsvm'):
        X, y = load_svmlight_file(open(datafile, "r"))
        if sps.issparse(X):
            X = X.toarray()
    class_names = np.unique(y)
    print("Dataset %s X=%r y=%r" % (os.path.basename(datafile).split(".")[0],X.shape,tuple(class_names)))

    clf = WisardClassifier(n_bits=args.nbits,n_tics=args.ntics,
                           debug=True,bleaching=False,random_state=848484848, n_jobs=-1)
    y_pred = cross_val_predict(clf, X, y, cv=args.cvfolds)
    print("Accuracy: %.3f" % accuracy_score(y, y_pred))
    print("F-Measure: %.3f" % f1_score(y, y_pred, average='weighted'))

    cm = confusion_matrix(y, y_pred)
    if matplotfound:
        plt.figure()
        plot_confusion_matrix(cm, classes=class_names,title='Confusion matrix')
        plt.show()
    else:
        print_confmatrix(cm, fieldsize=5)

if __name__ == "__main__":
    main(sys.argv[1:])
    print "DONE",
