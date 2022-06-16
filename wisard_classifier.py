#import numpy library
import numpy as np
# import sklearn and scipy stuff
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score
import scipy.sparse as sps
from scipy.io import arff
# import wisard classifier library
from wisard import WisardClassifier
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
parser.add_argument('-l', "--labelpos", metavar='<labelpos>', default=-1, type=int, help='classification label position in dataset (default is last position -1)')
parser.add_argument('-n', "--labelname", metavar='<labelname>', default="class", type=str, help='classification label name in dataset (default is "class")')
parser.add_argument('-b', "--nbits", metavar='<bitno>', type=int, help='number of bits for WiSARD resolution [16]', default=16,required=False)
parser.add_argument('-f', "--cvfolds", metavar='<cvfolds>', type=int, help='number of folds in stratified cross-validation (default is 10))', default=10,required=False)
parser.add_argument('-z', "--ntics", metavar='<ticno>', type=int, help='number of tics for real discretization (default is 1024)', default=1024,required=False)
parser.add_argument('-p', "--njobs", metavar='<njobs>', type=int, help='number of cores used for parallel jobs (default is one core = 1)', default=1,required=False)
parser.add_argument('-d', "--debug", metavar='<debug>', type=bool, help='enable progress monitoring (enabled by default)', default=True,required=False)


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
            used_attrnames = [m for m in meta._attributes.keys() if m != args.labelname]
            X = np.array([list(x) for x in data[used_attrnames]])
        except:
            raise ValueError("Cannot find label %s in dataset" % args.labelname)
        print(X.dtype)
        print(y.dtype)
    elif datafile.endswith('.libsvm'):
        X, y = load_svmlight_file(datafile)
        if sps.issparse(X):
            X = X.toarray()
    elif datafile.endswith('.csv'):
        data = np.array(np.genfromtxt(datafile, delimiter=",",dtype=None))[1:]
        y = data[:,args.labelpos]
        X = np.delete(data, args.labelpos, 1)
        X = X.astype(np.float)
    elif datafile.endswith('.tsv'):
        data = np.array(np.genfromtxt(datafile, delimiter="\t",dtype=None))[1:]
        y = data[:,args.labelpos]
        X = np.delete(data, args.labelpos, 1)
        X = X.astype(np.float)
    else:
        raise Exception("wrong dataset extension")

    class_names = np.unique(y)
    print("Dataset %s X=%r y=%r" % (os.path.basename(datafile).split(".")[0],X.shape,tuple(class_names)))

    clf = WisardClassifier(n_bits=args.nbits,n_tics=args.ntics,
                           debug=args.debug,bleaching=False,random_state=848484848, n_jobs=args.njobs)
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
    print("DONE")
