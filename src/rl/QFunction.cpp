#include "QFunction.hpp"
#include "dout.hpp"

#include <algorithm>

using namespace std;

SumQ::SumQ(const vector<QFunction*>& qFuncs, act_t numActions) :
   qFuncs_{qFuncs},
   numActions_(numActions) {   
}

SumQ::~SumQ() {
   for (auto q : qFuncs_) {
      delete q;
   }
}

float SumQ::getQ(const State& state, act_t action) const {
   float qVal = 0;
   for (auto q : qFuncs_) {
      qVal += q->getQ(state, action);
   }
   return qVal;
}

void SumQ::getAllActQs(const State& state, vector<float>& qVals) const {
   qVals.clear();
   qVals.resize(numActions_, 0);
   vector<float> qvs;
   for (auto q : qFuncs_) {
      q->getAllActQs(state, qvs);
      for (act_t a = 0; a < numActions_; ++a) {
	 qVals[a] += qvs[a];
      }
   }
}

Bound SumQ::getQBound(const StateBound& stateBound, act_t action) const {
   Bound qBound{0, 0};
   for (auto q : qFuncs_) {
      Bound qr = q->getQBound(stateBound, action);
      qBound.lower += qr.lower;
      qBound.upper += qr.upper;
   }
   return qBound;
}

void SumQ::getAllActQBounds(const StateBound& state, vector<Bound>& qBounds) const {
   qBounds.clear();
   qBounds.resize(numActions_, {0, 0});
   vector<Bound> qrs;
   for (auto q : qFuncs_) {
      q->getAllActQBounds(state, qrs);
      for (act_t a = 0; a < numActions_; ++a) {
	 qBounds[a].lower += qrs[a].lower;
	 qBounds[a].upper += qrs[a].upper;
      }
   }
}

void SumQ::updateQ(const State& state, act_t action, float change) {
   for (auto q : qFuncs_) {
      q->updateQ(state, action, change);
   }
}

float SumQ::getStepSizeNormalizer() const {
   float norm = 0;
   for (auto q : qFuncs_) {
      norm += q->getStepSizeNormalizer();
   }
   return norm;
}
