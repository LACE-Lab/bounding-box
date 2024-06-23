#ifndef NNMODEL
#define NNMODEL

#include "PredictionModel.hpp"
#include "RNG.hpp"
#include "Params.hpp"
#include "Example.hpp"

#include <torch/torch.h>
#include <vector>
#include <tuple>

class NNModel : public LearnedModel, public BBIPredictionModel {
  public:
   enum TrainingType {mse, bound, iqn, gaussian};

   NNModel(size_t inDim, size_t targetDim, act_t numActions, const std::vector<Bound>& dimBounds, RNG& rng, TrainingType trainType, const Params& params);
   virtual ~NNModel();

   virtual void addExample(const Trajectory& traj, size_t t);
   virtual void removeTrajectory(const Trajectory* traj);
   
   virtual void updatePredictions();

   using BBIPredictionModel::getStateBounds;
   virtual void getStateBounds(const StateBound& premise, const std::vector<act_t>& action, StateBound& predictedBounds) const;

   using BBIPredictionModel::getRewardBounds;
   virtual void getRewardBounds(const StateBound& premise, const std::vector<act_t>& action, Bound& rewardBound) const;

   using BBIPredictionModel::getTermBounds;
   virtual void getTermBounds(const StateBound& premise, const std::vector<act_t>& action, Bound& termBound) const;

   using PredictionModel::getStatePrediction;
   virtual void getStatePrediction(const State& premise, act_t action, State& predictedState) const;
   using PredictionModel::getStateBounds;
   virtual void getStateBounds(const State& premise, act_t action, State& predictedState, StateBound& predictedBounds) const;

   using PredictionModel::getRewardPrediction;
   virtual rlfloat_t getRewardPrediction(const State& premise, act_t action) const;
   using PredictionModel::getRewardBounds;
   virtual rlfloat_t getRewardBounds(const State& premise, act_t action, Bound& rewardBound) const;

   using PredictionModel::getTermPrediction;
   virtual rlfloat_t getTermPrediction(const State& premise, act_t action) const;
   using PredictionModel::getTermBounds;
   virtual rlfloat_t getTermBounds(const State& premise, act_t action, Bound& termBound) const;   

   virtual void getStateDistribution(const State& premise, act_t action, StateNormal& stateDist) const;
   virtual void getRewardDistribution(const State& premise, act_t action, Normal& rewardDist) const;
   virtual void getTermDistribution(const State& premise, act_t action, Normal& termDist) const;   

   virtual void getStatePredSample(const State& premise, act_t action, State& sample) const;
   virtual rlfloat_t getRewardPredSample(const State& premise, act_t action) const;
   virtual bool getTermPredSample(const State& premise, act_t action) const;

  protected:
   virtual void prepareInputVector(const State& premise, act_t action, std::vector<double>& inVec) const;
   virtual torch::Tensor prepareInput(const State& premise, act_t action) const;
   virtual torch::Tensor prepareInput(const StateBound& premise, const std::vector<act_t>& action) const;

   virtual void prepareTargetVector(const State& result, rlfloat_t reward, rlfloat_t term, std::vector<double>& targetVec) const;
   virtual torch::Tensor prepareTarget(const State& result, rlfloat_t reward, rlfloat_t term) const;
   
   struct BBIModule : torch::nn::Module {
      virtual torch::Tensor forward(torch::Tensor x) = 0;
      // Assumes 3xn, with min, mean, max as the rows
      virtual torch::Tensor bbiForward(torch::Tensor bound) = 0;
      virtual void updateBookkeeping() {}
   };
      
   struct Net : BBIModule {
      Net(size_t inDim, const std::vector<size_t>& layerDims, size_t targetDim);
      torch::Tensor forward(torch::Tensor x);
      torch::Tensor bbiForward(torch::Tensor bound);
      void updateBookkeeping();

      torch::nn::Sequential layers_;
   };   

   struct BBILinear : BBIModule {
      BBILinear(torch::nn::Linear realModule);
      torch::Tensor forward(torch::Tensor x);      
      torch::Tensor bbiForward(torch::Tensor bound);
      void updateBookkeeping();

      torch::nn::Linear realModule_;
      long long inDim_;
      long long targetDim_;      
      torch::Tensor posWeights_;
   };

   struct BBIReLU : BBIModule {
      BBIReLU(torch::nn::ReLU realModule) : realModule_{register_module("realModule_", realModule)}{}
      torch::Tensor forward(torch::Tensor x) {return realModule_->forward(x);}
      torch::Tensor bbiForward(torch::Tensor bound) {return realModule_->forward(bound);}

      torch::nn::ReLU realModule_;
   };
   
   size_t inDim_;
   act_t numActions_;
   size_t targetDim_;

   std::vector<std::shared_ptr<Net> > nets_;
   std::vector<std::shared_ptr<torch::optim::Adam> > optimizers_;   

   std::vector<std::vector<Example*> > allExamples_;
   size_t batchSize_;

   mutable RNG rng_;

   TrainingType trainType_;
   size_t netInDim_;

   std::vector<rlfloat_t> inputScale_;
   std::vector<rlfloat_t> inputShift_;

   double varianceSmoothing_;
   bool predictChange_;
};

#endif
