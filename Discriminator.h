#include "Ram.h"
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <vector>

namespace wnn {
    class Discriminator {
    private:
        std::vector<Ram> ramarray;
        void init(int);
        void finish(int);
    public:
        unsigned long long mypowers[64] = {
            1UL, 2UL, 4UL, 8UL, 16UL, 32UL, 64UL, 128UL, 256UL, 512UL, 1024UL, 2048UL, 4096UL, 8192UL, 16384UL, 32768UL, 65536UL, 131072UL , 262144UL, 524288UL,
            1048576UL, 2097152UL, 4194304UL, 8388608UL, 16777216UL, 33554432UL, 67108864UL, 134217728UL, 268435456UL, 536870912UL, 1073741824UL, 2147483648UL,
            4294967296UL, 8589934592UL, 17179869184UL, 34359738368UL, 68719476736UL, 137438953472UL, 274877906944UL, 549755813888UL, 1099511627776UL, 2199023255552UL,
            4398046511104UL, 8796093022208UL, 17592186044416UL, 35184372088832UL, 70368744177664UL, 140737488355328UL, 281474976710656UL, 562949953421312UL, 1125899906842624UL,
            2251799813685248UL, 4503599627370496UL, 9007199254740992UL, 18014398509481984UL, 36028797018963968UL, 72057594037927936UL, 144115188075855872UL, 288230376151711744UL,
            576460752303423488UL, 1152921504606846976UL, 2305843009213693952UL, 4611686018427387904UL, 9223372036854775808UL
        };
        
        int n_rams;
        int n_locs;
        int n_bits;
        int n_pixels;
        std::string maptype = "random";
        std::vector<int> map;
        std::vector<double> mi;
        int maxmi;
        Discriminator(int n_bit, int size, std::string maptype);
        ~Discriminator();
        int getNBits();
        int getNRams();
        int getSize();
        double getMaxMI();
        std::vector<double> getMI();
        void Train(std::vector<double> data, std::vector<double> ranges, std::vector<double> offsets, int n_tics);
        void TrainNoScale(std::vector<double> data, int n_tics);
        void TrainByTuple(std::vector<int> tuple);
        void UnTrainByTuple(std::vector<int> tuple);
        double ClassifyByTuple(std::vector<int> tuple);
        double Classify(std::vector<double> data, std::vector<double> ranges, std::vector<double> offsets, int n_tics);
        double ClassifyNoScale(std::vector<double> data, int n_tics);
        std::vector<double> ResponseByTuple(std::vector<int> tuple);
        std::vector<double> Response(std::vector<double> data, std::vector<double> ranges, std::vector<double> offsets, int n_tics);
        std::vector<double> ResponseNoScale(std::vector<double> data, int n_tics);
        std::string toString(int);
    };
}
