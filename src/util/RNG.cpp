#include "RNG.hpp"
#include <random>

double RNG::gaussian(double mean, double stddev) {
   std::normal_distribution dist(mean, stddev);
   return dist(rng_);
}
