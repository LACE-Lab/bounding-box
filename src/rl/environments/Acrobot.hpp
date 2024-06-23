#ifndef ACROBOT
#define ACROBOT

#include "PredictionModel.hpp"
#include "RNG.hpp"
#include <cmath>

class Acrobot : public PredictionModel {
  public:
   Acrobot(bool distractor, RNG& initRNG);
   virtual ~Acrobot() = default;
   
   virtual void getStatePrediction(const State& premise, act_t action, State& predictedState) const;
   virtual void getStateBounds(const State& premise, act_t action, State& predictedState, StateBound& predictedBounds) const;
   virtual void getStatePredSample(const State& premise, act_t action, State& predictions) const;

   virtual rlfloat_t getRewardPrediction(const State& premise, act_t action) const;
   virtual rlfloat_t getRewardBounds(const State& premise, act_t action, Bound& rewardBound) const;
   virtual rlfloat_t getRewardPredSample(const State& premise, act_t action) const;

   virtual rlfloat_t getTermPrediction(const State& premise, act_t action) const;
   virtual rlfloat_t getTermBounds(const State& premise, act_t action, Bound& termBound) const;
   virtual bool getTermPredSample(const State& premise, act_t action) const;

  protected:
   virtual std::vector<rlfloat_t> dsdt(const State& state) const;
   virtual std::vector<rlfloat_t> scalarMult(const std::vector<rlfloat_t>& vec, rlfloat_t sca) const;
   virtual std::vector<rlfloat_t> scalarAdd(const std::vector<rlfloat_t>& vec, rlfloat_t sca) const;
   virtual std::vector<rlfloat_t> vectorAdd(const std::vector<rlfloat_t>& vec1, const std::vector<rlfloat_t>& vec2) const;
   virtual rlfloat_t wrap(rlfloat_t val, const rlfloat_t min, const rlfloat_t max) const;

   bool distractor_;
   mutable RNG rng_;
};

#endif
