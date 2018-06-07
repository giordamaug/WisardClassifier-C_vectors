//
//  Ram.cpp
//  
//
//  Created by Maurizio Giordano on 25/05/18.
//

#include "Ram.h"

namespace wnn {
    
    Ram::Ram(int Id, int NLocs) {
        idx = Id;
        n_locs = NLocs;
    }
    Ram::~Ram() {};
    
    double Ram::getEntry(unsigned long long key) {
        auto it = std::find_if( wentries.begin(), wentries.end(), [key](const Wentry& element){ return element.first == key;} );
        if (it != wentries.end())
            return it->second;
        else
            return 0.0;
    }
    
    void Ram::setEntry(unsigned long long key, double value) {
        auto it = std::find_if( wentries.begin(), wentries.end(), [key](const Wentry& element){ return element.first == key;} );
        if (it != wentries.end())
            it->second = value;
        else
            wentries.push_back(std::make_pair(key,value));
    }
    void Ram::incrEntry(unsigned long long key, double value) {
        auto it = std::find_if( wentries.begin(), wentries.end(), [key](const Wentry& element){ return element.first == key;} );
        if (it != wentries.end())
            it->second += value;
        else {
            wentries.push_back(std::make_pair(key,value));
        }

    };
    void Ram::decrEntry(unsigned long long key, double value) {
        auto it = std::find_if( wentries.begin(), wentries.end(), [key](const Wentry& element){ return element.first == key;} );
        if (it != wentries.end()) {
            it->second -= value;
            if (it->second == 0.0) wentries.erase(it);
        }
        
    };
    std::string Ram::toString() {
        std::stringstream ss;
        ss << "{";
        
        std::vector<Wentry>::iterator it;
        for (it = wentries.begin(); it != wentries.end(); it++) {
            ss << it->first << ":" << it->second << " ";
        }
        if (wentries.size() > 0) ss.seekp(-1, std::ios_base::end);
        ss << "},";
        return ss.str();
    }
}
