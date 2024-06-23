#include "IncDTModel.hpp"
#include "dout.hpp"

using namespace std;

IncDTModel::IncDTModel(size_t inDim,
		 act_t numActions,
		 RNG& rng,
		 const Params& params) :
   predictChange_(params.getInt("predict_change")),
   rng_{rng.randomInt()} {
   for (size_t i = 0; i < inDim; ++i) {
      stateModels_.push_back(new FastIncModelTree{inDim, numActions, rng, params});
      models_.push_back(stateModels_.back());
   }
         
   rwdModel_ = new FastIncModelTree{inDim, numActions, rng, params};
   models_.push_back(rwdModel_);

   termModel_ = new FastIncModelTree{inDim, numActions, rng, params};
   models_.push_back(termModel_);
}

IncDTModel::~IncDTModel(){
   for (auto m : models_) {
      delete m;
   }
}

void IncDTModel::addExample(const Trajectory& traj, size_t t) {
   for (size_t i = 0; i < stateModels_.size(); ++i) {
      if (predictChange_) {	 
	 stateModels_[i]->addExample(new PropertyChangeExample(traj, t, i));
      } else {
	 stateModels_[i]->addExample(new PropertyExample(traj, t, i));
      }
   }
   rwdModel_->addExample(new RewardExample(traj, t));
   termModel_->addExample(new TerminalExample(traj, t));
}

void IncDTModel::updatePredictions() {
   size_t i = 0;
   for (auto m : models_) {
      DOUT << "Model Update " << i << endl;
      ++i;
      m->split();
   }
}

void IncDTModel::getStatePrediction(const State& premise, act_t action, State& predictedState) const {
   predictedState.clear();
   for (size_t i = 0; i < stateModels_.size(); ++i) {
      State pred;
      stateModels_[i]->getPrediction(premise, action, pred);
      if (predictChange_) {
	 pred[0] += premise[i];
      }
      predictedState.push_back(pred[0]);
   }
}

void IncDTModel::getStateBounds(const State& premise, act_t action, State& predictedState, StateBound& predictedBounds) const {
   predictedState.clear();
   predictedBounds.clear();
   vector<rlfloat_t> pred(1); 
   vector<Bound> bound(1);
   for (size_t i = 0; i < stateModels_.size(); ++i) {
      stateModels_[i]->getPrediction(premise, action, pred);
      predictedState.push_back(pred[0]);
      stateModels_[i]->getPredBounds(premise, action, bound);
      predictedBounds.push_back(bound[0]);
      if (predictChange_) {
	 predictedState.back() += premise[i];
	 predictedBounds.back().lower += premise[i];
	 predictedBounds.back().upper += premise[i];
      }
   }
}

void IncDTModel::getStateBounds(const StateBound& premise, const vector<act_t>& action, StateBound& stateBounds) const {
   stateBounds.clear();
   vector<Bound> pred(1);
   for (size_t i = 0; i < stateModels_.size(); ++i) {
      DOUT << "Model " << i << " Pred" << endl;
      stateModels_[i]->getPredBounds(premise, action, pred);
      stateBounds.push_back(pred[0]);
      if (predictChange_) {
	 stateBounds.back().lower += premise[i].lower;
	 stateBounds.back().upper += premise[i].upper;
      }
   }
}

rlfloat_t IncDTModel::getRewardPrediction(const State& premise, act_t action) const {
   State pred;
   rwdModel_->getPrediction(premise, action, pred);
   return pred[0];
}

rlfloat_t IncDTModel::getRewardBounds(const State& premise, act_t action, Bound& rewardBound) const {
   vector<Bound> bounds;
   rwdModel_->getPredBounds(premise, action, bounds);
   rewardBound = bounds[0];
   vector<rlfloat_t> pred;
   rwdModel_->getPrediction(premise, action, pred);
   return pred[0];
}

void IncDTModel::getRewardBounds(const StateBound& premise, const vector<act_t>& action, Bound& rewardBound) const {
   vector<Bound> bounds;
   rwdModel_->getPredBounds(premise, action, bounds);
   rewardBound = bounds[0];
}

rlfloat_t IncDTModel::getTermPrediction(const State& premise, act_t action) const {
   State pred;
   termModel_->getPrediction(premise, action, pred);
   return pred[0];
}

rlfloat_t IncDTModel::getTermBounds(const State& premise, act_t action, Bound& termBound) const {
   StateBound bounds;
   termModel_->getPredBounds(premise, action, bounds);
   termBound = bounds[0];
   vector<rlfloat_t> pred;
   termModel_->getPrediction(premise, action, pred);
   return pred[0];
}

void IncDTModel::getTermBounds(const StateBound& premise, const vector<act_t>& action, Bound& termBound) const {
   vector<Bound> bounds;
   termModel_->getPredBounds(premise, action, bounds);
   termBound = bounds[0];
}

void IncDTModel::getStateDistribution(const State& premise, act_t action, StateNormal& stateDist) const {
   stateDist.clear();
   vector<Normal> dist;
   for (size_t i = 0; i < stateModels_.size(); ++i) {
      stateModels_[i]->getPredDist(premise, action, dist);
      stateDist.push_back(dist[0]);
      if (predictChange_) {
	 stateDist.back().mean += premise[i];
      }	 
   }
}

void IncDTModel::getRewardDistribution(const State& premise, act_t action, Normal& rewardDist) const {
   vector<Normal> dist;
   rwdModel_->getPredDist(premise, action, dist);
   rewardDist = dist[0];
}

void IncDTModel::getTermDistribution(const State& premise, act_t action, Normal& termDist) const {
   vector<Normal> dist;
   termModel_->getPredDist(premise, action, dist);
   termDist = dist[0];
}

void IncDTModel::getStatePredSample(const State& premise, act_t action, State& sample) const {
   sample.clear();
   for (size_t i = 0; i < stateModels_.size(); ++i) {
      State pred;
      stateModels_[i]->getPredSample(premise, action, pred);
      sample.push_back(pred[0]);
      if (predictChange_) {
	 sample.back() += premise[i];
      }
   }
}

rlfloat_t IncDTModel::getRewardPredSample(const State& premise, act_t action) const {
   State pred;
   rwdModel_->getPredSample(premise, action, pred);
   return pred[0];
}

bool IncDTModel::getTermPredSample(const State& premise, act_t action) const {
   State pred;
   termModel_->getPredSample(premise, action, pred);
   return pred[0] > 0.5;
}
