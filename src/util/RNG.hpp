#ifndef RNG_HPP
#define RNG_HPP

#include <random>

class RNG {
  public:
   RNG(unsigned seed) : rng_(seed) {}
   double randomFloat() {return unitDist_(rng_);}
   unsigned randomInt() {return rng_();}
   double gaussian(double mean, double stddev);
  private:
   std::mt19937 rng_;
   std::uniform_real_distribution<double> unitDist_;
};

#endif
