#include "QLearner.hpp"
#include "dout.hpp"

#include <cmath>
#include <iostream>
#include <string>

using namespace std;

QLearner::QLearner(QFunction* qFunc,
                   act_t numActions,
		   RNG& rng,
		   const Params& params) :
   qFunc_(qFunc),
   initialStepsize_(params.getFloat("step_size")),
   numActions_(numActions),
   rng_(rng.randomInt()),
   params_(params) {
}

QLearner::~QLearner() {
   delete qFunc_;
}

///////////////////////////////
// Q-learning                //
///////////////////////////////
void QLearner::qUpdate(const Trajectory &traj, size_t t)
{
   rlfloat_t discount = params_.getFloat("discount");

   act_t greedyAct = 0; //Default values if game is over
   rlfloat_t greedyQ = 0;
   if (!traj.getResultGameOver(t)) {
      tie(greedyAct, greedyQ) = greedy(traj.getResultState(t));
   }

   rlfloat_t r = traj.getReward(t);
   rlfloat_t target = r + discount * greedyQ;

   act_t a = traj.getAction(t);
   rlfloat_t tdErr = target - qFunc_->getQ(traj.getPremiseState(t), a);
   DOUT << "Target: " << target << " tdErr: " << tdErr << endl;

   rlfloat_t stepSize = initialStepsize_/qFunc_->getStepSizeNormalizer();
   rlfloat_t change = stepSize*tdErr;
   qFunc_->updateQ(traj.getPremiseState(t), a, change);
}

act_t QLearner::getGreedyAction(const State& s) const {
   rlfloat_t dummy;
   return getGreedyAction(s, dummy);
}

act_t QLearner::getGreedyAction(const State &s, rlfloat_t& qVal) const {
   auto [a, q] = greedy(s);
   qVal = q;
   return a;
}

tuple<act_t, rlfloat_t> QLearner::greedy(const State& state) const {
   vector<rlfloat_t> qVals;
   qFunc_->getAllActQs(state, qVals);

   rlfloat_t greedyQ = -numeric_limits<rlfloat_t>::infinity();
   vector<act_t> greedyActs;
   for (act_t a = 0; a < numActions_; a++) {
      rlfloat_t q = qVals[a];
      
      DOUT << "a " << a << " q " << q << endl;
      
      if (q > greedyQ) {
         greedyActs.clear();
         greedyActs.push_back(a);
         greedyQ = q;
      } else if (fabs(q - greedyQ) < 1e-6) {
         greedyActs.push_back(a);
      }
   }

   act_t act = greedyActs[static_cast<act_t>(rng_.randomFloat()*greedyActs.size())];
   return make_tuple(act, greedyQ);
}

///////////////////////////////
// Unselective SMVE          //
///////////////////////////////
void QLearner::mveUpdate(const Trajectory& traj,
			 size_t t,
			 PredictionModel* model,
			 PredictionModel* env,
			 Measurements& measurements) {
   size_t horizon = params_.getInt("horizon");

   if (traj.getResultGameOver(t)) { // Episode has terminated so we can just update and leave.
      qUpdate(traj, t);
      return;
   }

   RNG rngCopy = rng_;

   expectationRollout(traj, t, horizon, model, measurements);

   for (size_t i = 0; i < measurements.states.size(); ++i) {
      measurements.uncertainties.push_back(0);
      measurements.weights.push_back(1);
      measurements.uncertaintyError.push_back(0);
   }
   
   DOUT << "Model Free Target: " << measurements.targets[0] << endl;
   DOUT << "Uncertainties: ";
   for (auto u : measurements.uncertainties) {
      DOUT << u << " ";
   }
   DOUT << endl;
   DOUT << "Raw Weights: ";
   for (auto w : measurements.weights) {
      DOUT << w << " ";
   }
   DOUT << endl;   

   if (env != nullptr) {
      RNG curRNG = rng_; 
      rng_ = rngCopy;      // Reset the RNG so random tie breaking is the same
      measurePredictionError(traj, t, env, measurements);
      rng_ = curRNG;       // Now put it back where it was after the original rollout
   } 
   
   weightedAvgUpdate(traj, t, measurements.targets, measurements.weights);
}

void QLearner::expectationRollout(const Trajectory& traj,
				  size_t t,
				  size_t horizon,
				  PredictionModel* model,
				  Measurements& measurements) {
   rlfloat_t discount = params_.getFloat("discount");

   vector<State>& states = measurements.states;
   states.clear();
   State curS = traj.getResultState(t);
   states.push_back(curS);
   
   vector<rlfloat_t>& rewards = measurements.rewards;
   rewards.clear();
   rlfloat_t cumR = traj.getReward(t);
   rewards.push_back(cumR);

   vector<rlfloat_t>& terms = measurements.terms;
   terms.clear();
   rlfloat_t term = traj.getResultGameOver(t);
   bool terminated = term > 0.5;
   terms.push_back(term);

   vector<rlfloat_t>& targets = measurements.targets;
   targets.clear();
   auto [action, q] = greedy(curS);
   targets.push_back(cumR + discount*q);
   
   rlfloat_t totalDiscount = discount;   
   for (size_t i = 1; i < horizon; ++i) {      
      DOUT << "Expectation Rollout " << i << endl;
      DOUT << "curS: ";
      for (auto d : curS) {
	 DOUT << d << " ";
      }
      DOUT << endl;
      
      State nextS = curS;      
      bool nextTerminated = true;
      rlfloat_t r = 0;
      rlfloat_t nextQ = 0;
      rlfloat_t nextTerm = 1;
      act_t nextAct = 0;

      if(!terminated) {
	 // prediction
	 model->getStatePrediction(curS, action, nextS);	 
	 r = model->getRewardPrediction(curS, action);
	 nextTerm = model->getTermPrediction(curS, action);
	 
	 DOUT << "action: " << action << endl;
	 DOUT << "nextS: ";
	 for (auto d : nextS) {
	    DOUT << d << " ";
	 }
	 DOUT << endl;
	 DOUT << "Model rwd: " << r << endl;
	 DOUT << "Model next term: " << term << endl;
	 
	 nextTerminated = term > 0.5;
	 if (!nextTerminated) {
	    for (size_t d = 0; d < nextS.size(); ++d) {
	       nextS[d] = nextS[d];
	    }
	    tie(nextAct, nextQ) = greedy(nextS);
	    
	    DOUT << "Next act: " << nextAct << " Next q: " << nextQ << endl;    
	 }
      }

      states.push_back(nextS);
      rewards.push_back(r);
      terms.push_back(nextTerm);

      cumR += totalDiscount*r;
      totalDiscount *= discount;      
      targets.push_back(cumR + totalDiscount*nextQ);
    
      // increment s
      curS = nextS;
      action = nextAct;

      terminated = terminated or nextTerminated;
   }      
}

void QLearner::weightedAvgUpdate(const Trajectory &traj,
				 const size_t t,
				 const vector<rlfloat_t>& targets,
				 const vector<rlfloat_t>& horizonWeights) {
   // calculating target from the value vector
   size_t n = targets.size();

   rlfloat_t totalW = 0;   // total weight
   rlfloat_t totalVal = 0;   // total target value (r&q)
   
   DOUT << "Targets: ";   
   for (size_t i = 0; i < n; i++) {      
      DOUT << targets[i] << " ";      
      totalVal += targets[i] * horizonWeights[i];
      totalW += horizonWeights[i];
   }   
   DOUT << endl;
   
   rlfloat_t target = totalVal / totalW;
   
   DOUT << "Weights: ";
   for (auto w : horizonWeights) {
      DOUT << w/totalW << " ";
   }
   DOUT << endl;

   DOUT << "Adjusted target: " << target << endl;
   
   // update
   const State& s = traj.getPremiseState(t);
   act_t a = traj.getAction(t);
   rlfloat_t curQ = qFunc_->getQ(s, a);
   rlfloat_t tdErr = target - curQ;
   
   DOUT << "Current Q: " << curQ << endl;
   
   rlfloat_t stepSize = initialStepsize_/qFunc_->getStepSizeNormalizer();
   rlfloat_t change = stepSize * tdErr;
   qFunc_->updateQ(s, a, change);
}

///////////////////////////////
// One-step uncertainty SMVE //
///////////////////////////////
void QLearner::oneStepUncertaintySMVEUpdate(const Trajectory& traj,
					    size_t t,
					    PredictionModel* model,
					    PredictionModel* env,
					    BBIPredictionModel* uncertainEnv,
					    Measurements& measurements) {
   size_t horizon = params_.getInt("horizon");

   if (traj.getResultGameOver(t)) { // Episode has terminated so we can just update and leave.
      qUpdate(traj, t);
      return;
   }

   RNG copyRNG = rng_;
   
   oneStepUncRollout(traj, t, horizon, model, measurements);  
   uncertaintiesToWeights(measurements.uncertainties, measurements.weights);
   
   DOUT << "Model Free Target: " << measurements.targets[0] << endl;
   DOUT << "Uncertainties: ";
   for (auto u : measurements.uncertainties) {
      DOUT << u << " ";
   }
   DOUT << endl;
   DOUT << "Raw Weights: ";
   for (auto w : measurements.weights) {
      DOUT << w << " ";
   }
   DOUT << endl;   

   if (env != nullptr) {
      RNG curRNG = rng_; 
      rng_ = copyRNG;      // Reset the RNG so random tie breaking is the same
      measurePredictionError(traj, t, env, measurements);
      rng_ = curRNG;       // Now put it back where it was after the original rollout
   }

   if (uncertainEnv != nullptr) {
      RNG curRNG = rng_; 
      rng_ = copyRNG;      // Reset the RNG so random tie breaking is the same

      Measurements uncMeasurements;
      oneStepUncRollout(traj, t, horizon, uncertainEnv, uncMeasurements);
      
      rng_ = curRNG;       // Now put it back where it was after the original rollout

      vector<rlfloat_t>& uncUncertainties = uncMeasurements.uncertainties;
      for (size_t i = 0; i < uncUncertainties.size(); ++i) {
	 if (uncUncertainties[i] == numeric_limits<rlfloat_t>::infinity() and
	     measurements.uncertainties[i] == numeric_limits<rlfloat_t>::infinity()) {
	    measurements.uncertaintyError.push_back(0);
	 } else {
	    measurements.uncertaintyError.push_back(uncUncertainties[i] - measurements.uncertainties[i]);
	 }
      }
   }

   weightedAvgUpdate(traj, t, measurements.targets, measurements.weights);
}

void QLearner::uncertaintiesToWeights(const vector<rlfloat_t>& uncertainties,
				      vector<rlfloat_t>& weights) {
   rlfloat_t temperature = params_.getFloat("temperature");
   rlfloat_t decay = params_.getFloat("decay");

   rlfloat_t totalDecay = 1;
   for (auto u : uncertainties) {
      weights.push_back(exp(-u/temperature)*totalDecay);
      totalDecay *= decay;
   }
}

void QLearner::oneStepUncRollout(const Trajectory& traj,
				 size_t t,
				 size_t horizon,
				 PredictionModel* model,
				 Measurements&  measurements) {
   rlfloat_t discount = params_.getFloat("discount");
   bool incRwd = params_.getInt("inc_rwd");
   bool incState = params_.getInt("inc_state");

   vector<State>& states = measurements.states;
   states.clear();
   states.push_back(traj.getResultState(t));
   State curS = traj.getResultState(t);

   vector<rlfloat_t>& rewards = measurements.rewards;
   rewards.clear();
   rlfloat_t cumR = traj.getReward(t);
   rewards.push_back(cumR);

   vector<rlfloat_t>& terms = measurements.terms;
   terms.clear();
   rlfloat_t term = traj.getResultGameOver(t);
   bool terminated = term > 0.5;
   terms.push_back(term);

   vector<rlfloat_t>& targets = measurements.targets;
   targets.clear();
   auto [action, q] = greedy(curS);
   targets.push_back(cumR + discount*q);
   
   vector<rlfloat_t>& uncertainties = measurements.uncertainties;
   uncertainties.clear();
   uncertainties.push_back(0);

   rlfloat_t totalDiscount = discount;   
   for (size_t i = 1; i < horizon; ++i) {      
      DOUT << "Expectation Rollout " << i << endl;
      DOUT << "curS: ";
      for (auto d : curS) {
	 DOUT << d << " ";
      }
      DOUT << endl;
      
      State nextS = curS;
      StatePredUnc nextSPred;
      PredUnc rPred{0, 0};
      PredUnc nextTermPred;
      rlfloat_t nextQ = 0;
      bool nextTerm = true;
      act_t nextAct = 0;
      
      if(!terminated) {
	 if (params_.getInt("use_variance")) { // Variance
	    StateNormal nextSDist;
	    model->getStateDistribution(curS, action, nextSDist);
	    for (auto d : nextSDist) {
	       nextSPred.push_back({d.mean, d.var});
	    }

	    Normal rDist;
	    model->getRewardDistribution(curS, action, rDist);
	    rPred = {rDist.mean, rDist.var};

	    Normal nextTermDist;
	    model->getTermDistribution(curS, action, nextTermDist);
	    nextTermPred = {nextTermDist.mean, nextTermDist.var};
	 
	    DOUT << "action: " << action << endl;
	    DOUT << "nextSDist: ";
	    for (auto d : nextSDist) {
	       DOUT << "N(" << d.mean << "," << d.var << ") ";
	    }
	    DOUT << endl;
	    DOUT << "Model rwd: N(" << rDist.mean << "," << rDist.var << ")" << endl;
	    DOUT << "Model next term: N(" << nextTermDist.mean << "," << nextTermDist.var << ")" << endl;	    
	 } else { // Bounds
	    StateBound nextSBound;	    
	    model->getStateBounds(curS, action, nextS, nextSBound);
	    nextSPred.clear();
	    for (size_t i = 0; i < nextSBound.size(); ++i) {
	       nextSPred.push_back({nextS[i], nextSBound[i].upper - nextSBound[i].lower});
	    }
	    
	    Bound rBound;
	    float r = model->getRewardBounds(curS, action, rBound);
	    rPred = {r, rBound.upper - rBound.lower};

	    Bound nextTermBound;
	    float term = model->getTermBounds(curS, action, nextTermBound);
	    nextTermPred = {term, nextTermBound.upper - nextTermBound.lower};
	 
	    DOUT << "action: " << action << endl;
	    DOUT << "nextSBound: ";
	    for (size_t i = 0; i < nextSBound.size(); ++i) {
	       DOUT << "(" << nextSBound[i].lower << "," << nextS[i] << "," << nextSBound[i].upper << ") ";
	    }
	    DOUT << endl;
	    DOUT << "Model rwd: (" << rBound.lower << "," << r << "," << rBound.upper << ")" << endl;
	    DOUT << "Model next term: (" << nextTermBound.lower << "," << term << "," << nextTermBound.upper << ")" << endl;
	 }
	 
	 nextTerm = term > 0.5;
	 if (!nextTerm) {
	    for (size_t d = 0; d < nextSPred.size(); ++d) {
	       nextS[d] = nextSPred[d].pred;
	    }
	    tie(nextAct, nextQ) = greedy(nextS);
	    
	    DOUT << "Next act: " << nextAct << " Next q: " << nextQ << endl;    
	 }
      } else {
	 for (auto x : nextS) {
	    nextSPred.push_back({x, 0});
	 }
	 nextTermPred = {1, 0};
      }

      states.push_back(nextS);
      rewards.push_back(rPred.pred);
      terms.push_back(nextTermPred.pred);

      cumR += totalDiscount*rPred.pred;
      totalDiscount *= discount;
      targets.push_back(cumR + totalDiscount*nextQ);

      rlfloat_t uncertainty = uncertainties.back();
      if (incRwd) {
	 uncertainty += rPred.uncertainty;
      }
      if (incState) {
	 for (size_t d = 0; d < nextSPred.size(); d++) {
	    uncertainty += nextSPred[d].uncertainty;
	 }
      }
      uncertainties.push_back(uncertainty);

      // increment s
      curS = nextS;
      action = nextAct;

      terminated = terminated or nextTerm;
   }   
}

///////////////////////////////
// BBI Target Range SMVE     //
///////////////////////////////
void QLearner::targetRangeSMVEUpdate(const Trajectory& traj,
				     size_t t,
				     BBIPredictionModel* model,
				     PredictionModel* env,
				     BBIPredictionModel* uncertainEnv,
				     Measurements& measurements) {
   size_t horizon = params_.getInt("horizon");

   if (traj.getResultGameOver(t)) { // Episode has terminated so we can just update and leave.
      qUpdate(traj, t);
      return;
   }

   RNG copyRNG = rng_;
   
   expectationRollout(traj, t, horizon, model, measurements);

   // Current q estimate
   rlfloat_t predictedQ = qFunc_->getQ(traj.getPremiseState(t), traj.getAction(t));      

   // Target ranges
   vector<Bound> targetBounds;
   bbiRollout(traj, t, horizon, model, targetBounds);
   getTargetRanges(targetBounds, predictedQ, measurements.targets, measurements.uncertainties);
   uncertaintiesToWeights(measurements.uncertainties, measurements.weights);
   
   DOUT << "Model Free Target: " << measurements.targets[0] << endl;
   DOUT << "Uncertainties: ";
   for (auto u : measurements.uncertainties) {
      DOUT << u << " ";
   }
   DOUT << endl;
   DOUT << "Raw Weights: ";
   for (auto w : measurements.weights) {
      DOUT << w << " ";
   }
   DOUT << endl;

   if (env != nullptr) {
      RNG curRNG = rng_; 
      rng_ = copyRNG;      // Reset the RNG so random tie breaking is the same
      measurePredictionError(traj, t, env, measurements);
      rng_ = curRNG;       // Now put it back where it was after the original rollout
   }

   if (uncertainEnv != nullptr) {
      RNG curRNG = rng_; 
      rng_ = copyRNG;      // Reset the RNG so random tie breaking is the same
      measureBBIError(traj, t, uncertainEnv, measurements);
      rng_ = curRNG;
   }

   weightedAvgUpdate(traj, t, measurements.targets, measurements.weights);   
}

void QLearner::bbiRollout(const Trajectory& traj,
			  size_t t,
			  size_t horizon,
			  BBIPredictionModel* model,
			  vector<Bound>& targetBounds) {
   rlfloat_t discount = params_.getFloat("discount");

   State curS = traj.getResultState(t);
   StateBound curSBound;
   for (auto d : curS) {
      curSBound.push_back({d, d});
   }

   rlfloat_t cumR = traj.getReward(t);
   Bound cumRwdBound{cumR, cumR};
   
   rlfloat_t term = traj.getResultGameOver(t);
   Bound terminalBound{term, term};

   vector<act_t> actSet;
   Bound qBound = greedy(curSBound, actSet);
   targetBounds.push_back({cumR + discount*qBound.lower, cumR + discount*qBound.upper});

   rlfloat_t totalDiscount = discount;
   for (size_t i = 1; i < horizon; ++i) {     
      DOUT << "BBIRollout " << i << endl;
      DOUT << "curSBound: ";
      for (auto d : curSBound) {
	 DOUT << "(" << d.lower << "," << d.upper << ") ";
      }
      DOUT << endl;
      
      Bound rBound{0, 0};
      Bound nextQBound{0, 0};
      Bound nextTermBound{1, 1};
      vector<act_t> nextActSet;
      
      StateBound nextSBound;
      
      if (terminalBound.lower <= 0.5) {
	 DOUT << "ActSet: ";
	 for (auto a : actSet) {
	    DOUT << a << " ";
	 }
	 DOUT << endl;

	 model->getStateBounds(curSBound, actSet, nextSBound);
	 DOUT << "nextSBound: ";
	 for (auto d : nextSBound) {
	    DOUT << "(" << d.lower << "," << d.upper << ") ";
	 }
	 DOUT << endl;
	 
	 model->getRewardBounds(curSBound, actSet, rBound);	 
	 if (terminalBound.upper > 0.5) {
	    rBound.lower = min(0.0f, rBound.lower);
	    rBound.upper = max(0.0f, rBound.upper);
	 }	 
	 DOUT << "Model rwd: (" << rBound.lower << "," << rBound.upper << ")" << endl;
	 	 
	 model->getTermBounds(curSBound, actSet, nextTermBound);	 
	 DOUT << "Model next term: (" << nextTermBound.lower << "," << nextTermBound.upper << ")" << endl;
	 	 
	 if (nextTermBound.lower <= 0.5) { // Possible that we haven't terminated     
	    nextQBound = greedy(nextSBound, nextActSet);
	    
	    if (max(terminalBound.upper, nextTermBound.upper) > 0.5) {
	       nextQBound.lower = min(0.0f, nextQBound.lower);
	       nextQBound.upper = max(0.0f, nextQBound.upper);
	    }	    
	    DOUT << "Next q bounds: " << "(" << nextQBound.lower << "," << nextQBound.upper << ")" << endl;
	 } else {
	    nextSBound = curSBound;	    
	    DOUT << "Definitely terminated" << endl;	    
	 }
      }

      cumRwdBound.lower += totalDiscount*rBound.lower;
      cumRwdBound.upper += totalDiscount*rBound.upper;      
      totalDiscount *= discount;      
      rlfloat_t returnMin = (cumRwdBound.lower + totalDiscount*nextQBound.lower);
      rlfloat_t returnMax = (cumRwdBound.upper + totalDiscount*nextQBound.upper);

      targetBounds.push_back({returnMin, returnMax});

      curSBound = nextSBound;
      actSet = nextActSet;
      
      terminalBound.lower = max(terminalBound.lower, nextTermBound.lower);
      terminalBound.upper = max(terminalBound.upper, nextTermBound.upper);
   }
}

Bound QLearner::greedy(const StateBound& stateBound, vector<act_t>& greedyActs) const {
   vector<Bound> qBounds;
   qFunc_->getAllActQBounds(stateBound, qBounds);
   
   Bound greedyQBound {-numeric_limits<float>::infinity(), -numeric_limits<float>::infinity()};
   greedyActs.clear();
   vector<Bound> greedyActBounds;
   
   for (act_t a = 0; a < numActions_; a++) {
      Bound qBound = qBounds[a];
      
      DOUT << "a " << a << " q (" << qBound.lower << "," << qBound.upper << ")" << endl;
      
      if (qBound.lower > greedyQBound.upper) { // Clear winner
         greedyActs.clear();
         greedyActs.push_back(a);
	 greedyActBounds.clear();
	 greedyActBounds.push_back(qBound);
         greedyQBound = qBound;
      } else if (qBound.upper >= greedyQBound.lower) { // A contender
	 // If the new action has a higher min, we can always get at least this new min
	 greedyQBound.lower = max(qBound.lower, greedyQBound.lower);
	 // If the new action has a higher max, we can possibly get up to this new max
	 greedyQBound.upper = max(qBound.upper, greedyQBound.upper);
	 
	 act_t numGreedyActs = greedyActs.size();
	 act_t i = 0;
	 while (i < numGreedyActs) {
	    if (greedyActBounds[i].upper < greedyQBound.lower) {
	       // Max is smaller than the new min. This is no longer a contender.
	       // Move the reject to the back for future removal.
	       swap(greedyActs[i], greedyActs.back());
	       swap(greedyActBounds[i], greedyActBounds[numGreedyActs-1]);
	       --numGreedyActs;
	    } else {
	       ++i;
	    }
	 }
	 // Remove the rejects
	 greedyActs.resize(numGreedyActs);	 
	 greedyActBounds.resize(numGreedyActs);
	 
	 // And add the new one!
	 greedyActs.push_back(a);
	 greedyActBounds.push_back(qBound);
      }
      // else max is smaller than min, clear loser
   }
   
   return greedyQBound;
}

void QLearner::getTargetRanges(const vector<Bound>& targetBounds,
			       rlfloat_t predictedQ,
			       const vector<rlfloat_t>& targets,
			       vector<rlfloat_t>& uncertainties) {
   bool directionalRange = params_.getInt("directional_range");
   bool rejectOverlap = params_.getInt("reject_overlap");
   rlfloat_t temperature = params_.getFloat("temperature");

   for (size_t i = 0; i < targetBounds.size(); ++i) {
      if (temperature != numeric_limits<rlfloat_t>::infinity()) {
	 const Bound& targetBound = targetBounds[i];
	 rlfloat_t returnMin = targetBound.lower;
	 rlfloat_t returnMax = targetBound.upper;
	 
	 rlfloat_t directionalEdge;
	 bool overlap;
	 if (targets[i] > predictedQ) {
	    directionalEdge = returnMin;
	    overlap = (returnMin < predictedQ);
	 } else if (targets[i] < predictedQ) {
	    directionalEdge = returnMax;
	    overlap = (returnMax > predictedQ);
	 } else { // targets[i] == predictedQ
	    directionalEdge = returnMax; // Arbitrary choice
	    overlap = (returnMin < predictedQ) or (returnMax > predictedQ);
	 }
	 
	 if (rejectOverlap and overlap) {
	    uncertainties.push_back(numeric_limits<rlfloat_t>::infinity());
	 } else if (directionalRange) {
	    uncertainties.push_back(fabs(targets[i] - directionalEdge));
	 } else {
	    uncertainties.push_back(returnMax - returnMin);
	 }
      } else {
	 uncertainties.push_back(0);
      }
   }
}

///////////////////////////////
// Monte-Carlo SMVE          //
///////////////////////////////
void QLearner::monteCarloSMVEUpdate(const Trajectory& traj,
				    size_t t,
				    PredictionModel* model,
				    PredictionModel* env,
				    BBIPredictionModel* uncertainEnv,
				    Measurements& measurements) {
   size_t horizon = params_.getInt("horizon");
   bool useVariance = params_.getInt("use_variance");

   if (traj.getResultGameOver(t)) { // Episode has terminated so we can just update and leave.
      qUpdate(traj, t);
      return;
   }

   RNG copyRNG = rng_;
   
   vector<Population> targetPops;
   monteCarloRollout(traj, t, horizon, model, targetPops, measurements);

   if (useVariance) {
      getMCTargetVariances(targetPops, measurements.targets, measurements.uncertainties);
   } else {
      // Current q estimate
      rlfloat_t predictedQ = qFunc_->getQ(traj.getPremiseState(t), traj.getAction(t));   
      getMCTargetRanges(targetPops, predictedQ, measurements.targets, measurements.uncertainties);
   }

   uncertaintiesToWeights(measurements.uncertainties, measurements.weights);
   
   DOUT << "Model Free Target: " << measurements.targets[0] << endl;
   DOUT << "Uncertainties: ";
   for (auto u : measurements.uncertainties) {
      DOUT << u << " ";
   }
   DOUT << endl;
   DOUT << "Raw Weights: ";
   for (auto w : measurements.weights) {
      DOUT << w << " ";
   }
   DOUT << endl;

   if (env != nullptr) {
      RNG curRNG = rng_; 
      rng_ = copyRNG;      // Reset the RNG so random tie breaking is the same
      measurePredictionError(traj, t, env, measurements);
      rng_ = curRNG;       // Now put it back where it was after the original rollout
   }

   if (uncertainEnv != nullptr) {
      RNG curRNG = rng_; 
      rng_ = copyRNG;      // Reset the RNG so random tie breaking is the same
      measureBBIError(traj, t, uncertainEnv, measurements);
      rng_ = curRNG;       // Now put it back where it was after the original rollout
   }

   weightedAvgUpdate(traj, t, measurements.targets, measurements.weights);   
}

void QLearner::monteCarloRollout(const Trajectory& traj,
				 size_t t,
				 size_t horizon,
				 PredictionModel* model,
				 vector<Population>& targetPops,
				 Measurements& measurements) {
   rlfloat_t discount = params_.getFloat("discount");
   size_t numSamples = params_.getInt("num_samples");

   vector<State>& states = measurements.states;
   states.clear();
   StatePop curSPop(numSamples, traj.getResultState(t));
   states.push_back(curSPop[0]);

   vector<rlfloat_t>& rewards = measurements.rewards;
   rewards.clear();
   Population cumRPop(numSamples, traj.getReward(t));
   rewards.push_back(cumRPop[0]);
   
   vector<rlfloat_t>& terms = measurements.terms;
   terms.clear();
   rlfloat_t term = traj.getResultGameOver(t);
   terms.push_back(term);
   vector<unsigned char> termPop(numSamples, term >= 0.5);

   vector<rlfloat_t>& targets = measurements.targets;
   targets.clear();
   targetPops.clear();   
   auto [a, q] = greedy(curSPop[0]);   
   targetPops.push_back(Population(numSamples, cumRPop.back() + discount*q));
   targets.push_back(targetPops.back()[0]);

   // In case of random tie-breaking between actions
   vector<act_t> actPop;
   for (size_t i = 0; i < numSamples; ++i) {
      tie(a, q) = greedy(curSPop[i]);
      actPop.push_back(a);
   }

   rlfloat_t totalDiscount = discount;   
   for (size_t i = 1; i < horizon; ++i) {
      DOUT << "Rollout " << i << endl;
      targets.push_back(0);
      targetPops.push_back(Population(numSamples, 0));
      Population rPop(numSamples, 0);
      Population tPop(numSamples, 0);
      StatePop sPop(numSamples);

      DOUT << "numSamples: " << numSamples << endl;
      for (size_t j = 0; j < numSamples; ++j) {
	 DOUT << "MC Rollout " << i << " Sample " << j << endl;
	 DOUT << "curS: ";
	 for (auto d : curSPop[j]) {
	    DOUT << d << " ";
	 }
	 DOUT << endl;

	 rlfloat_t nextQ = 0;
	 act_t nextAct = 0;

	 if (!termPop[j]) {
	    model->getStatePredSample(curSPop[j], actPop[j], sPop[j]);
	    DOUT << "Next s: ";
	    for (size_t d = 0; d < sPop[j].size(); ++d) {
	       DOUT << sPop[j][d] << " ";
	    }
	    DOUT << endl;
	    rPop[j] = model->getRewardPredSample(curSPop[j], actPop[j]);
	    tPop[j] = model->getTermPredSample(curSPop[j], actPop[j]);

	    DOUT << "Predicted r: " << rPop[j] << endl;
	    DOUT << "Predicted term: " << tPop[j] << endl;
	    
	    if (tPop[j] < 0.5) {	 
	       tie(nextAct, nextQ) = greedy(sPop[j]);
	    }
	 } else {
	    sPop[j] = curSPop[j];
	    tPop[j] = 1;
	 }

	 cumRPop[j] += totalDiscount*rPop[j];
      	 targetPops[i][j] = cumRPop[j] + totalDiscount*discount*nextQ;
	 targets.back() += targetPops[i][j]/numSamples;
      
	 DOUT << "cumR: " << cumRPop[j] << " nextQ: " << nextQ << " target: " << targetPops[i][j] << endl;

	 curSPop[j] = sPop[j];
	 actPop[j] = nextAct;
	 termPop[j] = termPop[j] or (tPop[j] >= 0.5);
      }

      State s(sPop[0].size(), 0);
      rlfloat_t r = 0;
      rlfloat_t t = 0;
      for (unsigned j = 0; j < numSamples; ++j) {
	 for (unsigned d = 0; d < s.size(); ++d) {
	    s[d] += sPop[j][d]/numSamples;
	 }
	 r += rPop[j]/numSamples;
	 t += tPop[j]/numSamples;
      }
      states.push_back(s);
      rewards.push_back(r);
      terms.push_back(t);
      
      totalDiscount *= discount;
   }   
}

void QLearner::getMCTargetRanges(const vector<Population>& targetPop,
				 rlfloat_t predictedQ,
				 const vector<rlfloat_t> targets,
				 vector<rlfloat_t>& uncertainties) {
   vector<Bound> targetBounds;
   for (size_t i = 0; i < targetPop.size(); ++i) {
      rlfloat_t maxTarget = targetPop[i][0];
      rlfloat_t minTarget = targetPop[i][0];
      for (size_t j = 1; j < targetPop[i].size(); ++j) {
	 maxTarget = max(maxTarget, targetPop[i][j]);
	 minTarget = min(minTarget, targetPop[i][j]);
      }
      targetBounds.push_back({minTarget, maxTarget});
   }
   getTargetRanges(targetBounds, predictedQ, targets, uncertainties);
}

void QLearner::getMCTargetVariances(const vector<Population>& targetPops,
				    const vector<rlfloat_t> targets,
				    vector<rlfloat_t>& uncertainties) {
   size_t numSamples = params_.getInt("num_samples");
   rlfloat_t temperature = params_.getFloat("temperature");

   for (size_t i = 0; i < targetPops.size(); ++i) {
      if (numSamples <= 0 or temperature == numeric_limits<rlfloat_t>::infinity()) {
	 uncertainties.push_back(0);
      } else {
	 rlfloat_t uncertainty = 0;
	 for (size_t j = 0; j < targetPops[i].size(); ++j) {
	    uncertainty += (targetPops[i][j] - targets[i])*(targetPops[i][j] - targets[i]);	       
	 }
	 uncertainty /= numSamples - 1;
	 uncertainties.push_back(uncertainty);
      }
   }
}

///////////////////////////////
// Error measurement         //
///////////////////////////////
void QLearner::measurePredictionError(const Trajectory& traj,
				      size_t t,
				      PredictionModel* env,
				      Measurements& measurements) {
   size_t horizon = params_.getInt("horizon");

   Measurements envMeasurements;
   expectationRollout(traj, t, horizon, env, envMeasurements);
   
   for (size_t i = 0; i < envMeasurements.states.size(); ++i) {
      measurements.stateError.push_back(vector<rlfloat_t>());
      for (size_t j = 0; j < measurements.states[i].size(); ++j) {
	 measurements.stateError.back().push_back(envMeasurements.states[i][j] -
						  measurements.states[i][j]);
      }
      measurements.rwdError.push_back(envMeasurements.rewards[i] - measurements.rewards[i]);
      measurements.termError.push_back(envMeasurements.terms[i] - measurements.terms[i]);      
      measurements.targetError.push_back(envMeasurements.targets[i] - measurements.targets[i]);
   }
}

void QLearner::measureBBIError(const Trajectory& traj,
			       size_t t,
			       BBIPredictionModel* uncertainEnv,
			       Measurements& measurements) {
   size_t horizon = params_.getInt("horizon");
   
   // Now get targets using the uncertain oracle
   Measurements uncMeasurements;
   expectationRollout(traj, t, horizon, uncertainEnv, uncMeasurements);
   
   vector<Bound> uncTargetBounds;
   bbiRollout(traj, t, horizon, uncertainEnv, uncTargetBounds);
   
   rlfloat_t predictedQ = qFunc_->getQ(traj.getPremiseState(t), traj.getAction(t));      
   vector<rlfloat_t> uncUncertainties;
   getTargetRanges(uncTargetBounds, predictedQ, uncMeasurements.targets, uncUncertainties);
   
   for (size_t i = 0; i < uncUncertainties.size(); ++i) {
      if (uncUncertainties[i] == numeric_limits<rlfloat_t>::infinity() and
	  measurements.uncertainties[i] == numeric_limits<rlfloat_t>::infinity()) {
	 measurements.uncertaintyError.push_back(0);
      } else {
	 measurements.uncertaintyError.push_back(uncUncertainties[i] - measurements.uncertainties[i]);
      }
   }
}
