//
//  Discriminator.cpp
//  
//
//  Created by Maurizio Giordano on 24/05/18.
//

#include "Discriminator.h"
#include "Exception.cpp"
#include <assert.h>

namespace wnn {
    void Discriminator::init(int NBits) {
        n_bits = NBits;
        n_locs = mypowers[n_bits];
        maxmi = (double) 0;
    }
    void Discriminator::finish(int Size) {
        if (Size % n_bits == 0) n_rams = Size / n_bits;
        else n_rams = Size / n_bits + 1;
        n_pixels = Size;  // round up size
        for(int index = 0; index < n_rams; index++)
            ramarray.push_back(*new Ram(index, n_locs));
        for (int i = 0; i < n_pixels; i++) {
            map.push_back(i);
            mi.push_back(0.0);
        }
        if (maptype == "random") std::random_shuffle(map.begin(), map.end());
    }
    Discriminator::Discriminator(int NBits, int Size, std::string mapType) {
        maptype = mapType;
        init(NBits);
        finish(Size);
    }
    Discriminator::~Discriminator() {};
    
    int Discriminator::getNBits() { return n_bits; }
    int Discriminator::getNRams() { return n_rams; }
    int Discriminator::getSize() { return n_pixels; }
    double Discriminator::getMaxMI() { return maxmi; }
    void Discriminator::Train(std::vector<double> data, std::vector<double> ranges, std::vector<double> offsets, int n_tics) {
        unsigned long long address;
        int x, i, index, value;
        
        if (data.size() != ranges.size() and data.size() != offsets.size()) throw wnn::WisardException( "wrong data/offsets/ranges size" );
        if (n_pixels % data.size() != 0) throw wnn::WisardException( "wrong retina partition" );
        std::vector<Ram>::iterator it;
        for (it = ramarray.begin(); it != ramarray.end(); it++) {
            // compute neuron simulus
            address=(unsigned long long)0;
            // decompose record data values into wisard input
            for (i=0;i<n_bits;i++) {
                x = map[(((it->idx * n_bits) + i) % n_pixels)];
                index = x/n_tics;
                value = (int) ((data[index] - offsets[index]) * n_tics / ranges[index]);
                if ((x % n_tics) < value) {
                    address |= mypowers[n_bits -1 - i];
                }
            }
            it->incrEntry(address, 1.0);
        }
    }
    void Discriminator::TrainNoScale(std::vector<double> data, int n_tics) {
        unsigned long long address;
        int x, i, index, value;
        
        if (n_pixels % data.size() != 0) throw wnn::WisardException( "wrong retina partition" );
        std::vector<Ram>::iterator it;
        for (it = ramarray.begin(); it != ramarray.end(); it++) {
            // compute neuron simulus
            address=(unsigned long long)0;
            // decompose record data values into wisard input
            for (i=0;i<n_bits;i++) {
                x = map[(((it->idx * n_bits) + i) % n_pixels)];
                index = x/n_tics;
                value = (int) (data[index] * n_tics);
                if ((x % n_tics) < value) {
                    address |= mypowers[n_bits -1 - i];
                }
            }
            it->incrEntry(address, 1.0);
        }
    }

    void Discriminator::TrainByTuple(std::vector<int> tuple) {
        if (tuple.size() != (unsigned long)n_rams) throw wnn::WisardException( "wrong tuple size" );
        std::vector<Ram>::iterator it;
        for (it = ramarray.begin(); it != ramarray.end(); it++) {
            it->incrEntry((unsigned long long)tuple.at(it->idx), 1.0);
        }
    }
    void Discriminator::UnTrainByTuple(std::vector<int> tuple) {
        if (tuple.size() != (unsigned long)n_rams) throw wnn::WisardException( "wrong tuple size" );
        std::vector<Ram>::iterator it;
        for (it = ramarray.begin(); it != ramarray.end(); it++) {
            it->decrEntry((unsigned long long)tuple.at(it->idx), 1.0);
        }
    }
    double Discriminator::ClassifyByTuple(std::vector<int> tuple) {
        if (tuple.size() != (unsigned long)n_rams) throw wnn::WisardException( "wrong tuple size" );
        std::vector<Ram>::iterator it;
        int sum=0;
        for (it = ramarray.begin(); it != ramarray.end(); it++)
            if (it->getEntry((unsigned long long)tuple.at(it->idx)) > 0.0) sum++;
        return sum/(double)n_rams;
    }
    double Discriminator::Classify(std::vector<double> data, std::vector<double> ranges, std::vector<double> offsets, int n_tics) {
        unsigned long long address;
        int x, i, index, value;
        
        if (data.size() != ranges.size() and data.size() != offsets.size()) throw wnn::WisardException( "wrong data/offsets/ranges size" );
        if (n_pixels % data.size() != 0) throw wnn::WisardException( "wrong retina partition" );
        std::vector<Ram>::iterator it;
        int sum=0;
        for (it = ramarray.begin(); it != ramarray.end(); it++) {
            // compute neuron simulus
            address=(unsigned long long)0;
            // decompose record data values into wisard input
            for (i=0;i<n_bits;i++) {
                x = map[(((it->idx * n_bits) + i) % n_pixels)];
                index = x/n_tics;
                value = (int) ((data[index] - offsets[index]) * n_tics / ranges[index]);
                if ((x % n_tics) < value) {
                    address |= mypowers[n_bits -1 - i];
                }
            }
            if (it->getEntry(address) > 0.0) sum++;
        }
        return sum/(double)n_rams;
    }
    double Discriminator::ClassifyNoScale(std::vector<double> data, int n_tics) {
        unsigned long long address;
        int x, i, index, value;
        
        if (n_pixels % data.size() != 0) throw wnn::WisardException( "wrong retina partition" );
        std::vector<Ram>::iterator it;
        int sum=0;
        for (it = ramarray.begin(); it != ramarray.end(); it++) {
            // compute neuron simulus
            address=(unsigned long long)0;
            // decompose record data values into wisard input
            for (i=0;i<n_bits;i++) {
                x = map[(((it->idx * n_bits) + i) % n_pixels)];
                index = x/n_tics;
                value = (int) (data[index] * n_tics);
                if ((x % n_tics) < value) {
                    address |= mypowers[n_bits -1 - i];
                }
            }
            if (it->getEntry(address) > 0.0) sum++;
        }
        return sum/(double)n_rams;
    }
    std::vector<double> Discriminator::ResponseByTuple(std::vector<int> tuple) {
        std::vector<double> response;
        if (tuple.size() != (unsigned long)n_rams) throw wnn::WisardException( "wrong tuple size" );
        std::vector<Ram>::iterator it;
        for (it = ramarray.begin(); it != ramarray.end(); it++)
            response.push_back(it->getEntry((unsigned long long)tuple.at(it->idx)));
        return response;
    }
    std::vector<double> Discriminator::Response(std::vector<double> data, std::vector<double> ranges, std::vector<double> offsets, int n_tics) {
        unsigned long long address;
        int x, i, index, value;
        std::vector<double> response;

        if (data.size() != ranges.size() and data.size() != offsets.size()) throw wnn::WisardException( "wrong data/offsets/ranges size" );
        if (n_pixels % data.size() != 0) throw wnn::WisardException( "wrong retina partition" );
        std::vector<Ram>::iterator it;
        for (it = ramarray.begin(); it != ramarray.end(); it++) {
            // compute neuron simulus
            address=(unsigned long long)0;
            // decompose record data values into wisard input
            for (i=0;i<n_bits;i++) {
                x = map[(((it->idx * n_bits) + i) % n_pixels)];
                index = x/n_tics;
                value = (int) ((data[index] - offsets[index]) * n_tics / ranges[index]);
                if ((x % n_tics) < value) {
                    address |= mypowers[n_bits -1 - i];
                }
            }
            response.push_back(it->getEntry(address));
        }
        return response;

    }
    std::vector<double> Discriminator::ResponseNoScale(std::vector<double> data, int n_tics) {
        unsigned long long address;
        int x, i, index, value;
        std::vector<double> response;

        if (n_pixels % data.size() != 0) throw wnn::WisardException( "wrong retina partition" );
        std::vector<Ram>::iterator it;
        for (it = ramarray.begin(); it != ramarray.end(); it++) {
            // compute neuron simulus
            address=(unsigned long long)0;
            // decompose record data values into wisard input
            for (i=0;i<n_bits;i++) {
                x = map[(((it->idx * n_bits) + i) % n_pixels)];
                index = x/n_tics;
                value = (int) (data[index] * n_tics);
                if ((x % n_tics) < value) {
                    address |= mypowers[n_bits -1 - i];
                }
            }
            response.push_back(it->getEntry(address));
        }
        return response;
    }

    std::vector<double> Discriminator::getMI() {
        
        std::vector<Ram>::iterator it;
        double value;
        int offset;
        for (int i=0; i< n_pixels; i++) mi[i] = 0.0;
        for (it = ramarray.begin(), offset=0; it != ramarray.end(); it++, offset+=n_bits) {
            std::vector<Wentry>::iterator wit;
            for (wit = it->wentries.begin(); wit != it->wentries.end(); wit++) {
                for (int b = 0; b < n_bits; b++) {
                    if (((wit->first)>>(unsigned long long)(n_bits - 1 - b) & 1) > 0) {
                        value = mi[map[(offset + b) % n_pixels]] += wit->second;
                        if (maxmi < value) maxmi = value;
                    }
                }
            }
        }
        return mi;
    }
    std::vector<int> Discriminator::getMapping() { return map;
    }
    std::string Discriminator::toString(int mode) {
        std::stringstream ss;
        
        ss << "{ bits: " << n_bits << "\n  nram: " << n_rams << "\n  loc: " << n_locs << "\n  size: " << n_pixels;
        ss << "\n  map: [";
        bool disable = false;
        int step = mode;
        for (int i=0; i < n_pixels; i++) {
            if (disable && mode>0) {
                if (i + step  >= n_pixels) disable = false;
            } else
                if (!disable && mode>0) {
                    if (i + step   >= n_pixels) {
                        ss << map.at(i) << " ";
                    } else if (i >= step)  {
                        ss << "... ";
                        disable = true;
                    } else {
                        ss << map.at(i) << " ";
                    }
                } else
                    ss << map.at(i) << " ";
        }
        ss.seekp(-1, std::ios_base::end);
        ss << "]";
        ss << "\n  rams: [";
        disable = false;
        std::vector<Ram>::iterator it;
        int cnt = 0;
        for (it = ramarray.begin();it != ramarray.end(); it++, cnt++) {
            if (disable && mode>0) {
                if (cnt + step  >= n_rams) disable = false;
            } else
                if (!disable && mode>0) {
                    if (cnt + step >= n_rams) {
                        ss << it->toString();
                    } else if (cnt >= step)  {
                        ss << " ... ";
                        disable = true;
                    } else {
                        ss << it->toString();
                    }
                } else
                    ss << it->toString();
        }
        if (ramarray.size() > 0) ss.seekp(-1, std::ios_base::end);
        ss << "]" << std::endl;
        ss << "}" << std::endl;
        return ss.str();
    }

}
