#ifndef RL_TYPES
#define RL_TYPES

#include <vector>
#include <tuple>

using act_t = std::size_t;

using rlfloat_t = float;
using State = std::vector<rlfloat_t>;

struct Bound {
   rlfloat_t lower;
   rlfloat_t upper;
};
using StateBound = std::vector<Bound>;

struct Normal {
   rlfloat_t mean;
   rlfloat_t var;
};
using StateNormal = std::vector<Normal>;

struct PredUnc {
   rlfloat_t pred;
   rlfloat_t uncertainty;
};
using StatePredUnc = std::vector<PredUnc>;

using Population = std::vector<rlfloat_t>;
using StatePop = std::vector<State>;

#endif
