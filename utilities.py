#!/opt/local/bin/python2.7
# -*- coding: utf-8 -*-

# Code source: Maurizio Giordano
#
#
# License: GPL
import numpy as np
try:
    import matplotlib
    #matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    import itertools
    matplotfound = True
except ImportError:
    matplotfound = False
    pass

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[0;32m'
    WHITEBLACK = '\033[1m\033[40;37m'
    BLUEBLACK = '\033[1m\033[40;94m'
    YELLOWBLACK = '\033[1m\033[40;93m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_confmatrix(table,fieldsize=3,decimals=3):
    nclasses = len(table)
    hfrmt = '{0: >%d}' % fieldsize
    dfrmt = '%%%dd' % fieldsize
    ffrmt = '%%%d.0%df' % (fieldsize,decimals)
    str = (' ' * fieldsize)
    for c in range(nclasses):
        str +=  ' '  + color.BOLD + hfrmt.format(c) + color.END
    print(str)
    print((' ' * fieldsize) + '┌' + ('─' * fieldsize + '┬') * (nclasses-1) + ('─' * fieldsize) + '┐')
    for k in range(nclasses):
        str = color.BOLD + hfrmt.format(k) + color.END
        for j in range(nclasses):
            if table[k][j]==0:
                str += '│' + (' '* fieldsize)
                continue
            if j==k:
                str += '│' + dfrmt % (table[k][j])
            else:
                str += '│' + color.RED + dfrmt % (table[k][j]) + color.END
        str += '│'
        print(str + '')
    print((' ' * fieldsize) + '└' + ('─' * fieldsize + '┴') * (nclasses-1) + ('─' * fieldsize) + '┘')

if matplotfound:
    def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title += " (Normalized)"
        else:
            title += " (No Normalization)"

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')