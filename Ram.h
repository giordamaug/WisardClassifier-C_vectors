//
//  Ram.h
//  
//
//  Created by Maurizio Giordano on 25/05/18.
//

#ifndef Ram_h
#define Ram_h
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

typedef std::pair<unsigned long, double> Wentry;

namespace wnn {
    class Ram {
    protected:
        int n_locs;
    public:
        std::vector<Wentry> wentries;
        int idx;
        Ram(int Idx, int NLocs);
        ~Ram();
        double getEntry(unsigned long long key);
        void setEntry(unsigned long long key, double value);
        void incrEntry(unsigned long long key, double value);
        void decrEntry(unsigned long long key, double value);
        std::string toString();
    };
}

#endif /* Ram_h */
