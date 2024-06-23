#include "GoRight.hpp"
#include "dout.hpp"
#include <cmath>
#include <algorithm>

using namespace std;

GoRight::GoRight(const Params& params) :
   numInd_(params.getInt("gor_num_ind")),
   length_(params.getInt("gor_length")),
   prizeMult_(params.getFloat("gor_prize_mult")),
   maxStat_(2),
   statScale_(rlfloat_t(length_)/maxStat_) {
}

void GoRight::getStatePrediction(const State& premise, act_t action, State& predictedState) const {
   int direction = action*2 - 1;
   if ((premise[0] < 0.5 and action == 0) or
       (premise[0] >= length_ - 0.5 and action == 1)) {
      direction = 0;
   }
   
   predictedState.resize(3 + numInd_);
   rlfloat_t nextPos = premise[0] + direction;

   rlfloat_t stat = premise[1 + numInd_];
   rlfloat_t prevStat = premise[2 + numInd_];
   rlfloat_t nextStat;

   // e.g. for maxStat_ = 2: 0 0 1 0 2 1 1 2 2 (repeat) 
   if (stat == prevStat and stat >= statScale_*(maxStat_ - 0.5)) { 
      nextStat = stat - maxStat_*statScale_;                       // e.g. 2 2 -> 0
   } else if (stat == prevStat) {
      nextStat = stat + statScale_;                                // e.g. 0 0 -> 1, 1 1 -> 2
   } else if (stat > prevStat and stat >= statScale_*(maxStat_ - 0.5)) {
      nextStat = prevStat + statScale_;                            // e.g. 0 2 -> 1, 1 2 -> 2
   } else if (stat > prevStat) {
      nextStat = prevStat;                                         // e.g. 0 1 -> 0
   } else if (prevStat >= statScale_*(maxStat_ - 0.5)) {
      nextStat = stat;                                             // e.g. 2 1 -> 1
   } else {
      nextStat = prevStat + statScale_;                            // e.g. 1 0 -> 2
   }
      
   predictedState[1 + numInd_] = nextStat;
   predictedState[2 + numInd_] = stat;

   for (size_t i = 1; i <= numInd_; ++i) { 
      predictedState[i] = premise[i]; // Usually stays the same
   }
   if (premise[0] < length_ - 0.5 and
       nextPos >= length_ - 0.5 and
       nextStat >= statScale_*(maxStat_ - 0.5)) { // Entering prize state with biggest status
      for (size_t i = 1; i <= numInd_; ++i) { // Flip all to 1
	 predictedState[i] = 1;
      }
   } else if (premise[0] >= length_ - 0.5 and nextPos >= length_ - 0.5) { // Staying at prize
      bool any1s = false;
      bool any0s = false;
      for (size_t i = numInd_; i >= 2; --i) {
	 predictedState[i] = premise[i-1];

	 if (premise[i] > 0.5) {
	    any1s = true;
	 } else {
	    any0s = true;
	 }
      }

      if ((!any1s and predictedState[1] < 0.5) or
	  (!any0s and predictedState[1] >= 0.5)) {
	 predictedState[1] = 1;
      } else {
	 predictedState[1] = 0;
      }
   } else if (premise[0] >= length_ - 0.5 and nextPos < length_ - 0.5) {
      for (size_t i = 1; i <= numInd_; ++i) { // Leaving the prize, flip all to 0
	 predictedState[i] = 0;
      }
   }
   
   predictedState[0] = nextPos;
}

rlfloat_t GoRight::getRewardPrediction(const State& premise, act_t action) const {
   State result;
   getStatePrediction(premise, action, result);

   bool allOn = true;
   for (size_t i = 1; i <= numInd_; ++i) {
      allOn = allOn and (premise[i] > 0.5);
   }
   
   if (allOn and action == 1) {
      return prizeMult_*ceil((1 - pow(0.9, length_ + maxStat_))/pow(0.9, length_ + maxStat_ + 1));
   } else if (action == 1) {
      return -1;
   } else {
      return 0;
   }
}

rlfloat_t GoRight::getTermPrediction(const State&, act_t) const {
   return 0;
}

GoRightUncertain::GoRightUncertain(RNG& rng, const Params& params) :
   rng_(rng.randomInt()),
   numInd_(params.getInt("gor_num_ind")),
   length_(params.getInt("gor_length")),
   prizeMult_(params.getFloat("gor_prize_mult")),
   maxStat_(2),
   statScale_(rlfloat_t(length_)/maxStat_),
   premiseBound_(3 + numInd_) {
}

void GoRightUncertain::getStatePrediction(const State& premise, act_t action, State& predictedState) const {
   StateBound predictedBounds;
   getStateBounds(premise, action, predictedState, predictedBounds);
}

void GoRightUncertain::getStateBounds(const State& premise, act_t action, State& predictedState, StateBound& predictedBounds) const {
   StateBound premiseBound;
   for (auto d : premise) {
      premiseBound.push_back({d, d});
   }
   getStateBounds(premiseBound, action, predictedBounds);
   StateNormal dist;
   getStateDistribution(premise, action, dist);
   predictedState.clear();
   for (auto d : dist) {
      predictedState.push_back(d.mean);
   }
}

void GoRightUncertain::getStateDistribution(const State& premise, act_t action, StateNormal& stateDist) const {
   stateDist.clear();
   int direction = action*2 - 1;
   if ((premise[0] < 0.5 and action == 0) or
       (premise[0] >= length_ - 0.5 and action == 1)) {
      direction = 0;
   }
   
   stateDist.resize(2 + numInd_);
   rlfloat_t nextPos = premise[0] + direction;
   // Deterministic
   stateDist[0] = {nextPos, 0};

   rlfloat_t stat = premise[1 + numInd_];
   size_t flatStat = round(stat/statScale_);
   // Uniform discrete distribution, but starting at 0 :-/
   // E: (0 + 1 + 2 + ... + maxStat_)/(maxStat_ + 1) = maxStat_(maxStat_ + 1)/2(maxStat_ + 1)
   //                                                = maxStat_/2
   // V: (0^2 + 1^2 + 2^2 + ... + maxStat_^2)/(maxStat_ + 1) + maxStat_^2/4
   //            = maxStat_(maxStat_+1)(2maxStat_+1)/6(maxStat_+1) + maxStat_^2/4
   //            = maxStat_(2maxStat_+1)/6 + maxStat_^2/4
   //            = (4maxStat_^2 + 2maxStat_ - 3maxX^2)/12
   //            = (maxStat_^2 + 2maxStat_)/12
   stateDist[1 + numInd_] = {stat + statScale_*(rlfloat_t(maxStat_/2.0) - flatStat),
                             rlfloat_t(statScale_*statScale_*maxStat_*(maxStat_ + 2)/12.0)};

   for (size_t i = 1; i <= numInd_; ++i) {
      // Deterministic
      stateDist[i] = {premise[i], 0}; // Usually stay still
   }
   if (premise[0] < length_ - 0.5 and stateDist[0].mean >= length_ - 0.5) { // Entering prize
      for (size_t i = 1; i <= numInd_; ++i) {
	 // Bernoulli distribution: E = p, V = p(1-p)
	 stateDist[i] = {rlfloat_t(1.0/(maxStat_+1)), rlfloat_t(1.0/(maxStat_+1)*maxStat_/(maxStat_+1))};
      }
   } else if (premise[0] >= length_ - 0.5 and stateDist[0].mean >= length_ - 0.5) { // Staying prize
      bool any1s = false;
      bool any0s = false;
      for (size_t i = numInd_; i >= 2; --i) {
	 stateDist[i] = {premise[i-1], 0};

	 if (premise[i] > 0.5) {
	    any1s = true;
	 } else {
	    any0s = true;
	 }
      }

      if ((!any1s and premise[1] < 0.5) or
	  (!any0s and premise[1] >= 0.5)) {
	 stateDist[1] = {1, 0};
      } else {
	 stateDist[1] = {0, 0};
      }
   } else if (premise[0] >= length_ - 0.5 and nextPos < length_ - 0.5) {
      for (size_t i = 1; i <= numInd_; ++i) {
	 stateDist[i] = {0, 0}; // If leaving the prize state, flip to 0
      }
   }   
}

void GoRightUncertain::getStatePredSample(const State& premise, act_t action, State& predictedState) const {
   predictedState.clear();
   int direction = action*2 - 1;
   if ((premise[0] < 0.5 and action == 0) or
       (premise[0] >= length_ - 0.5 and action == 1)) {
      direction = 0;
   }
   
   predictedState.resize(2 + numInd_);
   rlfloat_t nextPos = premise[0] + direction;
   predictedState[0] = nextPos;

   rlfloat_t stat = premise[1 + numInd_];
   size_t flatStat = round(stat/statScale_);
   long long r = rng_.randomInt();
   predictedState[1 + numInd_] = stat + statScale_*(rlfloat_t(r%(maxStat_ + 1)) - flatStat);

   for (size_t i = 1; i <= numInd_; ++i) {
      predictedState[i] = premise[i]; // Usually stay still
   }
   if (premise[0] < length_ - 0.5 and predictedState[0] >= length_ - 0.5) { // Entering prize
      for (size_t i = 1; i <= numInd_; ++i) {
	 predictedState[i] = (rng_.randomFloat() < 1.0/(maxStat_ + 1));
      }
   } else if (premise[0] >= length_ - 0.5 and predictedState[0] >= length_ - 0.5) { // Staying prize
      bool any1s = false;
      bool any0s = false;
      for (size_t i = numInd_; i >= 2; --i) {
	 predictedState[i] = premise[i-1];

	 if (premise[i] > 0.5) {
	    any1s = true;
	 } else {
	    any0s = true;
	 }
      }

      if ((!any1s and premise[1] < 0.5) or
	  (!any0s and premise[1] >= 0.5)) {
	 predictedState[1] = 1;
      } else {
	 predictedState[1] = 0;
      }
   } else if (premise[0] >= length_ - 0.5 and nextPos < length_ - 0.5) {
      for (size_t i = 1; i <= numInd_; ++i) {
	 predictedState[i] = 0; // If leaving the prize state, flip to 0
      }
   }   
}

void GoRightUncertain::getStateBounds(const StateBound& premise, const vector<act_t>& action, StateBound& predictedBounds) const {
   const rlfloat_t* statBound[]{&premise[0].lower, &premise[0].upper};
   rlfloat_t statPred[2];
   predictedBounds.resize(2 + numInd_);
   if (action.size() == 1) {
      for (size_t i = 0; i < 2; ++i) {
	 int direction = action[0]*2 - 1;
	 if ((*statBound[i] < 0.5 and action[0] == 0) or
	     (*statBound[i] >= length_ - 0.5 and action[0] == 1)) {
	    direction = 0;
	 }
	 statPred[i] = *statBound[i] + direction;
      }
   } else { // Don't know the action
      if (*statBound[0] >= 0.5) {
	 statPred[0] = *statBound[0] - 1;
      } else {
	 statPred[0] = *statBound[0];
      }

      if (*statBound[1] >= length_ - 0.5) {
	 statPred[1] = *statBound[1];
      } else {
	 statPred[1] = *statBound[1] + 1;
      }
   }

   // Keep the stat offset
   rlfloat_t stat = premise[1 + numInd_].upper;
   size_t flatStat = round(stat/statScale_)*statScale_;
   rlfloat_t statOffset = stat - flatStat;
   predictedBounds[1 + numInd_] = {statOffset, statOffset + statScale_*maxStat_};
   
   for (size_t i = 1; i <= numInd_; ++i) {
      predictedBounds[i] = premise[i]; // Usually stay the same
   }
   if (*statBound[1] >= length_ - 0.5 and statPred[1] >= length_ - 0.5) { // Maybe stay prize 
      bool all0s = true;
      bool maybeAll0s = true;
      bool all1s = true;
      bool maybeAll1s = true;
      for (size_t i = numInd_; i >= 2; --i) {
	 predictedBounds[i] = premise[i-1];

	 if (premise[i].lower >= 0.5) {
	    maybeAll0s = false;
	    all0s = false;
	 } else if (premise[i].upper >= 0.5) {
	    all0s = false;
	 }

	 if (premise[i].upper < 0.5) {
	    maybeAll1s = false;
	    all1s = false;
	 } else if (premise[i].lower < 0.5) {
	    all1s = false;
	 }
      }

      if ((all1s and premise[1].lower >= 0.5) or
	  (all0s and premise[1].upper < 0.5)) {
	 predictedBounds[1] = {1, 1};
      } else if ((maybeAll1s and premise[1].upper >= 0.5) or
		 (maybeAll0s and premise[1].lower < 0.5)) {
	 predictedBounds[1] = {0, 1};
      } else {
	 predictedBounds[1] = {0, 0};
      }
      
      if (statPred[0] < length_ - 0.5) { // Might be leaving
	 for (size_t i = 1; i <= numInd_; ++i) {
	    predictedBounds[i].lower = 0;
	 }      
      }
      
      if (*statBound[0] < length_ - 0.5) { // Might be arriving
	 for (size_t i = 1; i <= numInd_; ++i) {
	    predictedBounds[i].upper = 1;
	 }      
      }
   } else if (statPred[1] >= length_ - 0.5) { // Not staying prize, but might be entering the prize
      for (size_t i = 1; i <= numInd_; ++i) {
	 predictedBounds[i] = {0, 1};
      }
   } else if (statPred[0] < length_ - 0.5) { // Not at prize
      for (size_t i = 1; i <= numInd_; ++i) {
	 predictedBounds[i] = {0, 0};
      }	 
   } 

   predictedBounds[0] = {statPred[0], statPred[1]};
}

rlfloat_t GoRightUncertain::getRewardPrediction(const State& premise, act_t action) const {
   StateBound premiseBound;
   for (auto d : premise) {
      premiseBound.push_back({d, d});
   }
   Bound rewardBound;
   getRewardBounds(premiseBound, vector<act_t>(1, action), rewardBound);
   return rewardBound.lower; // Arbitrary because lower = upper here
}

void GoRightUncertain::getRewardBounds(const StateBound& premise, const vector<act_t>& action, Bound& rewardBound) const {
   if (action.size() == 1 and action[0] == 0) {  // going left
      rewardBound = {0, 0};
   } else {
      bool allOn = true;
      bool allMaybeOn = true;
      for (size_t i = 1; i <= numInd_; ++i) {
	 allOn = allOn and (premise[i].lower >= 0.5);
	 allMaybeOn = allMaybeOn and (premise[i].upper >= 0.5);
      }

      rlfloat_t highRwd = prizeMult_*ceil((1 - pow(0.9, length_ + maxStat_))/pow(0.9, length_ + maxStat_ + 1));
      
      if (action.size() == 1 and action[0] == 1) { // Definitely going right
	 if (allOn) {             // Definitely getting the prize
	    rewardBound = {highRwd, highRwd};
	 } else if (allMaybeOn) { // Maybe getting the prize
	    rewardBound = {-1, highRwd};
	 } else {                 // Not getting the prize
	    rewardBound = {-1, -1};
	 }
      } else {                                     // Don't know the action
	 if (allOn) {             // Definitely getting the prize when going right
	    rewardBound = {0, highRwd};
	 } else if (allMaybeOn) { // Maybe getting the prize when going right
	    rewardBound = {-1, highRwd};
	 } else {                 // Not getting the prize when going right
	    rewardBound = {-1, 0};
	 }
      }
   }
}

rlfloat_t GoRightUncertain::getTermPrediction(const State&, act_t) const {
   return 0;
}

void GoRightUncertain::getTermBounds(const StateBound&, const vector<act_t>&, Bound& termBound) const {
   termBound = {0, 0};
}
