#ifndef INC_DT_MODEL
#define INC_DT_MODEL

#include "FastIncModelTree.hpp"
#include "PredictionModel.hpp"
#include "Params.hpp"
#include "RNG.hpp"

class IncDTModel : public LearnedModel, public BBIPredictionModel {
  public:
   IncDTModel(size_t inDim, act_t numActions, RNG& rng, const Params& params);
   virtual ~IncDTModel();

   virtual void addExample(const Trajectory& traj, size_t t);
   
   virtual void removeTrajectory(const Trajectory*) {};
   
   virtual void updatePredictions();

   using PredictionModel::getStatePrediction;
   virtual void getStatePrediction(const State& premise, act_t action, State& predictedState) const;
   using PredictionModel::getStateBounds;
   virtual void getStateBounds(const State& premise, act_t action, State& preictedState, StateBound& predictedBounds) const;

   using BBIPredictionModel::getStateBounds;
   virtual void getStateBounds(const StateBound& premise, const std::vector<act_t>& action, StateBound& predictionBounds) const;

   using PredictionModel::getRewardPrediction;
   virtual rlfloat_t getRewardPrediction(const State& premise, act_t action) const;
   using PredictionModel::getRewardBounds;
   virtual rlfloat_t getRewardBounds(const State& premise, act_t action, Bound& rewardBound) const;

   using BBIPredictionModel::getRewardBounds;
   virtual void getRewardBounds(const StateBound& premise, const std::vector<act_t>& action, Bound& rewardBound) const;

   using PredictionModel::getTermPrediction;
   virtual rlfloat_t getTermPrediction(const State& premise, act_t action) const;
   using PredictionModel::getTermBounds;
   virtual rlfloat_t getTermBounds(const State& premise, act_t action, Bound& termBound) const;

   using BBIPredictionModel::getTermBounds;
   virtual void getTermBounds(const StateBound& premise, const std::vector<act_t>& action, Bound& termBound) const;

   virtual void getStateDistribution(const State& premise, act_t action, StateNormal& stateDist) const;
   virtual void getRewardDistribution(const State& premise, act_t action, Normal& rewardDist) const;
   virtual void getTermDistribution(const State& premise, act_t action, Normal& termDist) const;
   
   virtual void getStatePredSample(const State& premise, act_t action, State& sample) const;
   virtual rlfloat_t getRewardPredSample(const State& premise, act_t action) const;
   virtual bool getTermPredSample(const State& premise, act_t action) const;

  private:
   bool predictChange_;
   std::vector<FastIncModelTree*> stateModels_;
   FastIncModelTree* rwdModel_;
   FastIncModelTree* termModel_;
   std::vector<FastIncModelTree*> models_;      
   RNG rng_;
};

#endif
