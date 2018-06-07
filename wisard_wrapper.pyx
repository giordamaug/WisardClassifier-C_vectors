import numpy as np
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort as stdsort
from libcpp.string cimport string

cdef extern from "Discriminator.h" namespace "wnn":
    cdef cppclass Discriminator:
        Discriminator(int, int, string)
        int n_rams, n_locs, n_bits, n_pixels,
        double maxmi
        string name, maptype
        vector[double] mi
        vector[int] map
        int getNBits()
        int getSize()
        int getNRams()
        double getMaxMI()
        vector[double] getMI()
        string toString(int)
        void TrainByTuple(vector[int]&) except +
        void Train(vector[double]&, vector[double]&, vector[double]&, int) except +
        void TrainNoScale(vector[double]&, int) except +
        void UnTrainByTuple(vector[int]&) except +
        double ClassifyByTuple(vector[int]&) except +
        double Classify(vector[double]&, vector[double]&, vector[double]&, int) except +
        double ClassifyNoScale(vector[double]&, int) except +
        vector[double] ResponseByTuple(vector[int]&) except +
        vector[double] Response(vector[double]&, vector[double]&, vector[double]&,  int) except +
        vector[double] ResponseNoScale(vector[double]&, int) except +

cdef class PyDiscriminator:
    cdef Discriminator *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, bits, size, maptype="random"):
        self.thisptr = new Discriminator(bits, size, maptype)
    def __dealloc__(self):
        del self.thisptr
    def getNBits(self):
        return self.thisptr.getNBits()
    def getSize(self):
        return self.thisptr.getSize()
    def getMaxMI(self):
        return self.thisptr.getMaxMI()
    def getNRams(self):
        return self.thisptr.getNRams()
    def getMI(self):
        return np.array(self.thisptr.getMI())
    def Train(self, data, ranges, offsets, tics):
        self.thisptr.Train(data, ranges, offsets, tics)
    def TrainNoScale(self, data, tics):
        self.thisptr.TrainNoScale(data, tics)
    def TrainByTuple(self, tuple):
        self.thisptr.TrainByTuple(tuple)
    def UnTrainByTuple(self, tuple):
        self.thisptr.UnTrainByTuple(tuple)
    def ClassifyByTuple(self, tuple):
        return self.thisptr.ClassifyByTuple(tuple)
    def Classify(self, data, ranges, offsets, tics):
        return self.thisptr.Classify(data, ranges, offsets, tics)
    def ClassifyNoScale(self, data, tics):
        return self.thisptr.ClassifyNoScale(data, tics)
    def ResponseByTuple(self, tuple):
        return np.array(self.thisptr.ResponseByTuple(tuple))
    def Response(self, data, ranges, offsets, tics):
        return np.array(self.thisptr.Response(data, ranges, offsets, tics))
    def ResponseNoScale(self, data, tics):
        return np.array(self.thisptr.ResponseNoScale(data, tics))
    def toString(self, int mode=0):
        return self.thisptr.toString(mode)
