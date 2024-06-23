#ifndef PREDICTION_MODEL
#define PREDICTION_MODEL

#include "Trajectory.hpp"
#include "RLTypes.hpp"

#include <iostream>
#include <vector>
#include <tuple>

class PredictionModel {
  public:
   virtual ~PredictionModel() = default;

   virtual void getStatePrediction(const State& premise, act_t action, State& predictedState) const = 0;
   virtual void getStateBounds(const State& premise, act_t action, State& predictedState, StateBound& predictedBounds) const;
   virtual void getStateDistribution(const State& premise, act_t action, StateNormal& stateDist) const;
   virtual void getStatePredSample(const State& premise, act_t action, State& sample) const;
   
   virtual rlfloat_t getRewardPrediction(const State& premise, act_t action) const = 0;
   virtual rlfloat_t getRewardBounds(const State& premise, act_t action, Bound& rewardBound) const;
   virtual void getRewardDistribution(const State& premise, act_t action, Normal& rewardDist) const;
   virtual rlfloat_t getRewardPredSample(const State& premise, act_t action) const;
   
   virtual rlfloat_t getTermPrediction(const State& premise, act_t action) const = 0;
   virtual rlfloat_t getTermBounds(const State& premise, act_t action, Bound& termBound) const;
   virtual void getTermDistribution(const State& premise, act_t action, Normal& termDist) const;
   virtual bool getTermPredSample(const State& premise, act_t action) const;
};

class LearnedModel : virtual public PredictionModel {
  public:
   virtual void addExample(const Trajectory& traj, std::size_t t) = 0;
   
   virtual void removeTrajectory(const Trajectory* traj) = 0;
   
   virtual void updatePredictions() = 0;
};

class BBIPredictionModel : virtual public PredictionModel {
  public:
   using PredictionModel::getStateBounds;
   virtual void getStateBounds(const StateBound& premise, const std::vector<act_t>& action, StateBound& predictedBounds) const = 0;
   virtual void getStateBounds(const StateBound& premise, act_t action, StateBound& predictedBounds) const {getStateBounds(premise, std::vector<act_t>(1, action), predictedBounds);}   
   
   using PredictionModel::getRewardBounds;
   virtual void getRewardBounds(const StateBound& premise, const std::vector<act_t>& action, Bound& rewardBound) const = 0;
   virtual void getRewardBounds(const StateBound& premise, act_t action, Bound& rewardBound) const {getRewardBounds(premise, std::vector<act_t>(1, action), rewardBound);}
   
   using PredictionModel::getTermBounds;
   virtual void getTermBounds(const StateBound& premise, const std::vector<act_t>& action, Bound& termBound) const = 0;
   virtual void getTermBounds(const StateBound& premise, act_t action, Bound& termBound) const {getTermBounds(premise, std::vector<act_t>(1, action), termBound);}
};

#endif
