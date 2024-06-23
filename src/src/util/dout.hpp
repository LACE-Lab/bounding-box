#ifndef DOUT_HPP
#define DOUT_HPP

#include <iostream>

#ifdef DEBUG
#define DOUT cerr
#else
#define DOUT 0 and cerr
#endif

#endif
