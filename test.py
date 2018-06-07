
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


