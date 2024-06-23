#ifndef GO_RIGHT
#define GO_RIGHT

#include "PredictionModel.hpp"
#include "RNG.hpp"
#include "Params.hpp"

class GoRight : public PredictionModel {
  public:
   GoRight(const Params& params);
   virtual ~GoRight() = default;
   
   virtual void getStatePrediction(const State& premise, act_t action, State& predictions) const;
   virtual rlfloat_t getRewardPrediction(const State& premise, act_t action) const;
   virtual rlfloat_t getTermPrediction(const State& premise, act_t action) const;
   
  protected:
   std::size_t numInd_;
   std::size_t length_;
   rlfloat_t prizeMult_;
   std::size_t maxStat_;
   rlfloat_t statScale_;
};

class GoRightUncertain : public BBIPredictionModel {
  public:
   GoRightUncertain(RNG& rng, const Params& params);
   virtual ~GoRightUncertain() = default;

   using BBIPredictionModel::getStateBounds;
   virtual void getStateBounds(const StateBound& premise, const std::vector<act_t>& action, StateBound& predictedBounds) const;

   using PredictionModel::getStatePrediction;
   virtual void getStatePrediction(const State& premise, act_t action, State& predictedState) const;
   using PredictionModel::getStateBounds;
   virtual void getStateBounds(const State& premise, act_t action, State& predictedState, StateBound& predictedBounds) const;
   
   virtual void getStateDistribution(const State& premise, act_t action, StateNormal& stateDist) const;
   virtual void getStatePredSample(const State& premise, act_t action, State& predictions) const;

   using BBIPredictionModel::getRewardPrediction;
   virtual void getRewardBounds(const StateBound& premise, const std::vector<act_t>& action, Bound& rewardBound) const;

   using PredictionModel::getRewardPrediction;
   virtual rlfloat_t getRewardPrediction(const State& premise, act_t action) const;

   using BBIPredictionModel::getTermPrediction;
   virtual void getTermBounds(const StateBound& premise, const std::vector<act_t>& action, Bound& termBound) const;

   using PredictionModel::getTermPrediction;
   virtual rlfloat_t getTermPrediction(const State& premise, act_t action) const;

  protected:
   mutable RNG rng_;

   std::size_t numInd_;
   std::size_t length_;
   rlfloat_t prizeMult_;
   std::size_t maxStat_;
   rlfloat_t statScale_;

   mutable StateBound premiseBound_;
};

#endif
