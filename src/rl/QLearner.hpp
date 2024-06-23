#ifndef QLEARNER
#define QLEARNER

#include "Trajectory.hpp"
#include "QFunction.hpp"
#include "PredictionModel.hpp"
#include "Params.hpp"
#include "RNG.hpp"
#include "RLTypes.hpp"

#include <unordered_map>
#include <vector>
#include <iostream>
#include <fstream>

class QLearner
{
  public:
   struct Measurements {
      std::vector<State> states;
      std::vector<rlfloat_t> rewards;
      std::vector<rlfloat_t> terms;
      
      std::vector<rlfloat_t> targets;
      std::vector<rlfloat_t> uncertainties;
      std::vector<rlfloat_t> weights;

      std::vector<std::vector<rlfloat_t> > stateError;
      std::vector<rlfloat_t> rwdError;
      std::vector<rlfloat_t> termError;
      std::vector<rlfloat_t> targetError;
      std::vector<rlfloat_t> uncertaintyError;
   };

   QLearner(QFunction* qFunc, act_t numActions, RNG& rng, const Params& params);
   virtual ~QLearner();

   virtual void qUpdate(const Trajectory& traj, std::size_t t);   

   virtual void mveUpdate(const Trajectory& traj,
			  std::size_t t,
			  PredictionModel* model,
			  PredictionModel* env,
			  Measurements& measurements);   
   virtual void mveUpdate(const Trajectory& traj,
			  std::size_t t,
			  PredictionModel* model,
			  Measurements& measurements) {
      mveUpdate(traj, t, model, nullptr, measurements);
   }

   virtual void oneStepUncertaintySMVEUpdate(const Trajectory& traj,
					     std::size_t t,
					     PredictionModel* model,
					     PredictionModel* env,
					     BBIPredictionModel* uncertainEnv,
					     Measurements& measurements);   
   virtual void oneStepUncertaintySMVEUpdate(const Trajectory& traj,
					     std::size_t t,
					     PredictionModel* model,
					     Measurements& measurements) {
      oneStepUncertaintySMVEUpdate(traj, t, model, nullptr, nullptr, measurements);
   }

   virtual void targetRangeSMVEUpdate(const Trajectory& traj,
				      std::size_t t,
				      BBIPredictionModel* model,
				      PredictionModel* env,
				      BBIPredictionModel* uncertainEnv,
				      Measurements& measurements);
   virtual void targetRangeSMVEUpdate(const Trajectory& traj,
				      std::size_t t,
				      BBIPredictionModel* model,
				      Measurements& measurements) {
      targetRangeSMVEUpdate(traj, t, model, nullptr, nullptr, measurements);
   }
   
   virtual void monteCarloSMVEUpdate(const Trajectory& traj,
				     std::size_t t,
				     PredictionModel* model,
				     PredictionModel* env,
				     BBIPredictionModel* uncertainEnv,
				     Measurements& measurements);
   virtual void monteCarloSMVEUpdate(const Trajectory& traj,
				     std::size_t t,
				     PredictionModel* model,
				     Measurements& measurements) {
      monteCarloSMVEUpdate(traj, t, model, nullptr, nullptr, measurements);
   }

   virtual act_t getGreedyAction(const State& s) const;
   virtual act_t getGreedyAction(const State& s, rlfloat_t& qVal) const;   

  protected:
   virtual void weightedAvgUpdate(const Trajectory &traj,
				  const std::size_t t,
				  const std::vector<rlfloat_t>& targets,
				  const std::vector<rlfloat_t>& weights);
   
   std::tuple<act_t, rlfloat_t> greedy(const State& state) const;
   Bound greedy(const StateBound& stateBound, std::vector<act_t>& greedyActs) const;

   void expectationRollout(const Trajectory& traj,
			   std::size_t t,
			   std::size_t horizon,
			   PredictionModel* model,
			   Measurements& measurements);
   
   void oneStepUncRollout(const Trajectory& traj,
			  std::size_t t,
			  std::size_t horizon,
			  PredictionModel* model,
			  Measurements& measurements);
   
   void bbiRollout(const Trajectory& traj,
		   std::size_t t,
		   std::size_t horizon,
		   BBIPredictionModel* model,
		   std::vector<Bound>& targetBounds);
   void getTargetRanges(const std::vector<Bound>& targetBounds,
			rlfloat_t predictedQ,
			const std::vector<rlfloat_t>& targets,
			std::vector<rlfloat_t>& uncertainties);

   void monteCarloRollout(const Trajectory& traj,
			  std::size_t t,
			  std::size_t horizon,
			  PredictionModel* model,
			  std::vector<Population>& targetPops,
			  Measurements& measurements);
   void getMCTargetRanges(const std::vector<Population>& targetPop,
			  rlfloat_t predictedQ,
			  const std::vector<rlfloat_t> targets,
			  std::vector<rlfloat_t>& uncertainties);
   void getMCTargetVariances(const std::vector<Population>& targetPop,
			     const std::vector<rlfloat_t> targets,
			     std::vector<rlfloat_t>& uncertainties);

   void measurePredictionError(const Trajectory& traj,
			       std::size_t t,
			       PredictionModel* env,
			       Measurements& measurements);
   void measureBBIError(const Trajectory& traj,
			std::size_t t,
			BBIPredictionModel* uncertainEnv,
			Measurements& measurements);
   
   void uncertaintiesToWeights(const std::vector<rlfloat_t>& uncertainties,
			       std::vector<rlfloat_t>& weights);
   
   QFunction* qFunc_;
   rlfloat_t initialStepsize_;
   act_t numActions_;
   mutable RNG rng_;
   const Params& params_;
};

#endif
