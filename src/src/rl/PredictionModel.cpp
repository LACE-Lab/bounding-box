#include "PredictionModel.hpp"

using namespace std;

// By default presumes determinism

void PredictionModel::getStateBounds(const State& premise, act_t action, State& predictedState, StateBound& predictedBounds) const {
   getStatePrediction(premise, action, predictedState);
   predictedBounds.clear();
   for (auto d : predictedState) {
      predictedBounds.push_back({d, d});
   }
}

void PredictionModel::getStateDistribution(const State& premise, act_t action, StateNormal& stateDist) const {
   State pred;
   getStatePrediction(premise, action, pred);
   stateDist.clear();
   for (auto d : pred) {
      stateDist.push_back({d, 0});
   }
}

void PredictionModel::getStatePredSample(const State& premise, act_t action, State& sample) const {
   getStatePrediction(premise, action, sample);
}

rlfloat_t PredictionModel::getRewardBounds(const State& premise, act_t action, Bound& rewardBound) const {
   rlfloat_t pred = getRewardPrediction(premise, action);
   rewardBound = {pred, pred};
   return pred;
}

void PredictionModel::getRewardDistribution(const State& premise, act_t action, Normal& rewardDist) const {
   rlfloat_t pred = getRewardPrediction(premise, action);
   rewardDist = {pred, 0};
}

rlfloat_t PredictionModel::getRewardPredSample(const State& premise, act_t action) const {
   return getRewardPrediction(premise, action);
}

rlfloat_t PredictionModel::getTermBounds(const State& premise, act_t action, Bound& termBound) const {
   rlfloat_t pred = getTermPrediction(premise, action);
   termBound = {pred, pred};
   return pred;
}

void PredictionModel::getTermDistribution(const State& premise, act_t action, Normal& termDist) const {
   rlfloat_t pred = getTermPrediction(premise, action);
   termDist = {pred, 0};
}

bool PredictionModel::getTermPredSample(const State& premise, act_t action) const {
   return getTermPrediction(premise, action) > 0.5;
}
