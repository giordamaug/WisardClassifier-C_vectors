//
//  Exception.cpp
//  
//
//  Created by Maurizio Giordano on 31/05/18.
//

#include "Exception.hpp"

using namespace std;

namespace wnn {
    class WisardException : public std::exception {
    public:
        WisardException(std::string const &message) : msg_(message) { }
        virtual char const *what() const noexcept { return msg_.c_str(); }
        
    private:
        std::string msg_;
    };
}
