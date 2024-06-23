#include "FastIncModelTree.hpp"
#include "dout.hpp"

using namespace std;

FastIncModelTree::FastIncModelTree(size_t inDim, act_t numActions, RNG& initRNG, const Params& params) :
   inDim_{inDim},
   numActions_{numActions},
   root_{new Decision(inDim_, numActions_, "X")},
   numLeaves_{1},
   maxLeaves_(params.getInt("max_leaves")),
   confidence_(params.getFloat("split_confidence")),
   tieThreshold_(params.getFloat("tie_threshold")),
   rng_(initRNG.randomInt()) {
}

FastIncModelTree::Decision::Decision(size_t inDim, act_t numActions, string locStr) :
   locStr{locStr},
   left{nullptr},
   right{nullptr},
   splitCount{0},
   threshRoots{inDim, nullptr},
   actionSplits{numActions},
   discriminator{nullptr} {
}

FastIncModelTree::Threshold::Threshold(rlfloat_t threshold, Threshold* parent) :
   threshold{threshold},
   left{nullptr},
   right{nullptr},
   parent{parent} {
}

FastIncModelTree::Stats::Stats() :
   sum{0},
   sumSq{0},
   min{-numeric_limits<rlfloat_t>::infinity()},   
   max{numeric_limits<rlfloat_t>::infinity()},
   count{0} {
}

FastIncModelTree::~FastIncModelTree() {
   delete root_;
}

FastIncModelTree::Decision::~Decision() {
   delete discriminator;
   for (auto r : threshRoots) {
      delete r;
   }
   if (left) {
      delete left;
   }
   if (right) {
      delete right;
   }
}

FastIncModelTree::Threshold::~Threshold() {
   if (left) {
      delete left;
   }
   if (right) {
      delete right;
   }
}

void FastIncModelTree::addExample(Example* ex) {
   addExampleHelper(root_, ex);
   delete ex;
}

void FastIncModelTree::addExampleHelper(Decision* n, Example* ex) {
   const State& premise = ex->getPremiseState();
   act_t action = ex->getAction();
   
   if (n->discriminator) {
      if (n->discriminator->isRight(premise, action)) {
	 addExampleHelper(n->right, ex);
      } else {
	 addExampleHelper(n->left, ex);
      }
   } else {
      rlfloat_t outcome = ex->getOutcome()[0];
      n->predStats.sum += outcome;
      n->predStats.sumSq += outcome*outcome;
      n->predStats.count += 1;
      n->splitCount += 1;
      if (n->predStats.count == 1) {
	 DOUT << n->locStr << " first example: max and min = " << outcome << endl;
	 n->predStats.max = outcome;
	 n->predStats.min = outcome;
      } else {
	 n->predStats.max = max(n->predStats.max, outcome);
	 n->predStats.min = min(n->predStats.min, outcome);
	 DOUT << n->locStr << " updating: outcome = " << outcome << " min = " << n->predStats.min << " max = " << n->predStats.max << endl;
      }

      if (numLeaves_ < maxLeaves_) {
	 for (size_t i = 0; i < n->threshRoots.size(); ++i) {
	    rlfloat_t key = premise[i];
	    // Round to control memory growth
	    key = round(premise[i]*1000)/1000;
	    insertThreshold(n->threshRoots[i], nullptr, key, outcome);
	 }

	 for (act_t a = 0; a < numActions_; ++a) {
	    SplitStats& stats = n->actionSplits[a];
	    if (ex->getAction() == a) {
	       stats.right.sum += outcome;
	       stats.right.sumSq += outcome*outcome;
	       stats.right.count += 1;
	       if (stats.right.count == 1) {
		  stats.right.max = outcome;
		  stats.right.min = outcome;
	       } else {
		  stats.right.max = max(stats.right.max, outcome);
		  stats.right.min = min(stats.right.min, outcome);
	       }
	    } else {
	       stats.left.sum += outcome;
	       stats.left.sumSq += outcome*outcome;
	       stats.left.count += 1;
	       if (stats.left.count == 1) {
		  stats.left.max = outcome;
		  stats.left.min = outcome;
	       } else {
		  stats.left.max = max(stats.left.max, outcome);
		  stats.left.min = min(stats.left.min, outcome);
	       }
	    }
	 }
      }
   }
}

void FastIncModelTree::insertThreshold(Threshold*& n, Threshold* parent, rlfloat_t key, rlfloat_t outcome) {
   if (!n) { // Insert right here
      n = new Threshold(key, parent);
      n->stats.left.sum = outcome;
      n->stats.left.sumSq = outcome*outcome;
      n->stats.left.count = 1;
      n->stats.left.max = outcome;
      n->stats.left.min = outcome;
      n->hereAndRightMin = outcome;
      n->hereAndRightMax = outcome;
      DOUT << "Creating " << n << " left stats: counts " << n->stats.left.count << " " << n ->stats.right.count << " rmax " << n->stats.right.max << endl;
   } else {
      if (key < n->threshold) {
	 n->stats.left.sum += outcome;
	 n->stats.left.sumSq += outcome*outcome;
	 n->stats.left.count += 1;
	 n->stats.left.max = max(n->stats.left.max, outcome);
	 n->stats.left.min = min(n->stats.left.min, outcome);	 
	 DOUT << "Updating (l) " << n << " left stats: counts " << n->stats.left.count << " " << n ->stats.right.count << " rmax " << n->stats.right.max << endl;
	 insertThreshold(n->left, n, key, outcome);
      } else if (key > n->threshold) {
	 n->stats.right.sum += outcome;
	 n->stats.right.sumSq += outcome*outcome;
	 n->stats.right.count += 1;
	 n->hereAndRightMin = min(n->hereAndRightMin, outcome);
	 n->hereAndRightMax = max(n->hereAndRightMax, outcome);	 
	 if (n->stats.right.count == 1) {
	    n->stats.right.min = outcome;
	    n->stats.right.max = outcome;
	 } else {
	    n->stats.right.max = max(n->stats.right.max, outcome);
	    n->stats.right.min = min(n->stats.right.min, outcome);
	 }
	 DOUT << "Updating (r) " << n << " left stats: counts " << n->stats.left.count << " " << n ->stats.right.count << " rmax " << n->stats.right.max << endl;
	 insertThreshold(n->right, n, key, outcome);
      } else { // equal
	 n->stats.left.sum += outcome;
	 n->stats.left.sumSq += outcome*outcome;
	 n->stats.left.count += 1;
	 n->stats.left.max = max(n->stats.left.max, outcome);
	 n->stats.left.min = min(n->stats.left.min, outcome);	 
	 n->hereAndRightMin = min(n->hereAndRightMin, outcome);
	 n->hereAndRightMax = max(n->hereAndRightMax, outcome);
	 DOUT << "Updating (e) " << n << " left stats: counts " << n->stats.left.count << " " << n ->stats.right.count << " rmax " << n->stats.right.max << endl;
      }
   }
}

void FastIncModelTree::split() {
   if (numLeaves_ < maxLeaves_) {
      split(root_);
   }
   DOUT << "Num Leaves: " << numLeaves_ << endl;
}

void FastIncModelTree::split(Decision* n) {
   if (n->discriminator) {
      split(n->left);
      split(n->right);
   } else if (n->splitCount >= 2) { // Don't split nodes we never ask about
      if (numLeaves_ < maxLeaves_) {
	 DOUT << "Considering Leaf " << n->locStr << endl;
	 
	 size_t bestThreshDim = n->threshRoots.size();
	 rlfloat_t bestThresh = -numeric_limits<rlfloat_t>::infinity();
	 rlfloat_t bestThreshScore = -numeric_limits<rlfloat_t>::infinity();
	 SplitStats bestThreshStats;
	 rlfloat_t nextBestThreshScore = -numeric_limits<rlfloat_t>::infinity();
	 for (size_t i = 0; i < n->threshRoots.size(); ++i) {
	    if (n->threshRoots[i]) {
	       DOUT << "Dimension " << i << endl;
	       rlfloat_t best = -numeric_limits<rlfloat_t>::infinity();
	       rlfloat_t bestScore = -numeric_limits<rlfloat_t>::infinity();
	       SplitStats bestStats;
	       findBestThreshold(n->threshRoots[i], best, bestScore, bestStats);
	       
	       if (bestScore > bestThreshScore) {
		  nextBestThreshScore = bestThreshScore;
		  bestThresh = best;
		  bestThreshScore = bestScore;
		  bestThreshDim = i;
		  bestThreshStats = bestStats;
	       } else if (bestScore > nextBestThreshScore) {
		  nextBestThreshScore = bestScore;
	       }
	    }
	 }
	 
	 DOUT << "bestThresh dim: " << bestThreshDim << " thresh: " << bestThresh << " score: " << bestThreshScore << endl;
	 DOUT << "nextBestThreshScore: " << nextBestThreshScore << endl;
	 
	 act_t bestAct = numActions_;
	 rlfloat_t bestActScore = -numeric_limits<rlfloat_t>::infinity();
	 rlfloat_t nextBestActScore = -numeric_limits<rlfloat_t>::infinity();
	 SplitStats bestActStats;
	 for (act_t a = 0; a < n->actionSplits.size(); ++a) {
	    if (n->actionSplits[a].left.count > 0 and n->actionSplits[a].right.count > 0) {
	       rlfloat_t sdr = getSDR(n->actionSplits[a]);
	       if (sdr > bestActScore) {
		  nextBestActScore = bestActScore;
		  bestAct = a;
		  bestActScore = sdr;
		  bestActStats = n->actionSplits[a];
	       } else if (sdr > nextBestActScore) {
		  nextBestActScore = sdr;
	       }
	    }
	 }
	 DOUT << "bestAct: " << bestAct << " score: " << bestActScore << endl;
	 DOUT << "nextBestActScore: " << bestActScore << endl;
	 
	 Discriminator* bestSplit;
	 rlfloat_t bestScore;
	 SplitStats bestStats;
	 rlfloat_t nextBestScore;
	 if (bestThreshScore > bestActScore) {
	    bestSplit = new PropThreshDiscriminator(bestThreshDim, bestThresh);
	    bestScore = bestThreshScore;
	    bestStats = bestThreshStats;
	    
	    if (nextBestThreshScore > bestActScore) {
	       nextBestScore = nextBestThreshScore;
	    } else {
	       nextBestScore = bestActScore;
	    }
	 } else {
	    bestSplit = new OneHotActionDiscriminator(bestAct);
	    bestScore = bestActScore;
	    bestStats = bestActStats;
	    
	    if (nextBestActScore > bestThreshScore) {
	       nextBestScore = nextBestActScore;
	    } else {
	       nextBestScore = bestThreshScore;
	    }
	 }
	 
	 if (bestScore > 0) {
	    rlfloat_t ratio = nextBestScore/bestScore;
	    rlfloat_t bound = sqrt(log(1/confidence_)/(2*n->predStats.count));
	    
	    DOUT << "bestScore: " << bestScore << " nextBestScore: " << nextBestScore << endl;
	    DOUT << "ratio: " << ratio << " bound: " << bound << " count: " << n->predStats.count << endl;
	    
	    if (ratio + bound < 1 or bound < tieThreshold_) {
	       DOUT << "Splitting " << n->locStr << ": " << *bestSplit << endl;
	       n->discriminator = bestSplit;
	       
	       n->left = new Decision(inDim_, numActions_, n->locStr+"L");
	       n->left->predStats = bestStats.left;
	       if (n->left->predStats.min == -numeric_limits<rlfloat_t>::infinity() or
		   n->left->predStats.max == numeric_limits<rlfloat_t>::infinity()) {
		  DOUT << "New node " << n->left << " " << n->locStr << " has un-updated bounds" << endl;
	       }
	       
	       n->right = new Decision(inDim_, numActions_, n->locStr+"R");
	       n->right->predStats = bestStats.right;
	       if (n->right->predStats.min == -numeric_limits<rlfloat_t>::infinity() or
		   n->right->predStats.max == numeric_limits<rlfloat_t>::infinity()) {
		  DOUT << "New node " << n->right << " " << n->locStr << " has un-updated bounds" << endl;
	       }
	       
	       for (auto& r : n->threshRoots) {
		  delete r;
		  r = nullptr;
	       }
	       ++numLeaves_;
	    } else {
	       delete bestSplit;
	    }
	 } else {
	    delete bestSplit;
	 }
      }
   } else {
      DOUT << "Not ready to split: " << n->locStr << " " << n->splitCount << " " << n->predStats.count << endl;
   }
}

void FastIncModelTree::findBestThreshold(Threshold* root,
					 rlfloat_t& bestThreshold,
					 rlfloat_t& bestScore,
					 SplitStats& bestStats) const {
   SplitStats totalStats;
   totalStats.right.sum = root->stats.left.sum + root->stats.right.sum;
   totalStats.right.sumSq = root->stats.left.sumSq + root->stats.right.sumSq;
   totalStats.right.count = root->stats.left.count + root->stats.right.count;
   findBestThresholdHelper(root,
			   bestThreshold,
			   bestScore,
			   bestStats,
			   totalStats);   
}

void FastIncModelTree::findBestThresholdHelper(Threshold* n,
					       rlfloat_t& bestThreshold,
					       rlfloat_t& bestScore,
					       SplitStats& bestStats,
					       SplitStats totalStats) const {
   rlfloat_t rmin = totalStats.right.min;
   if (totalStats.right.min == -numeric_limits<rlfloat_t>::infinity()) {
      totalStats.right.min = n->hereAndRightMin;
   } else if (n->hereAndRightMin != -numeric_limits<rlfloat_t>::infinity()) {
      totalStats.right.min = min(totalStats.right.min, n->hereAndRightMin);
   }
   rlfloat_t rmax = totalStats.right.max;
   if (totalStats.right.max == numeric_limits<rlfloat_t>::infinity()) {
      totalStats.right.max = n->hereAndRightMax;
   } else if (n->hereAndRightMax != numeric_limits<rlfloat_t>::infinity()) {
      totalStats.right.max = max(totalStats.right.max, n->hereAndRightMax);
   }
   if (n->left) {
      findBestThresholdHelper(n->left, bestThreshold, bestScore, bestStats, totalStats);
   }

   totalStats.right.min = rmin;
   totalStats.right.max = rmax;
   if (totalStats.right.min == -numeric_limits<rlfloat_t>::infinity()) {
      totalStats.right.min = n->stats.right.min;
   } else if (n->stats.right.min != -numeric_limits<rlfloat_t>::infinity()) {
      totalStats.right.min = min(totalStats.right.min, n->stats.right.min);
   }
   if (totalStats.right.max == numeric_limits<rlfloat_t>::infinity()) {
      totalStats.right.max = n->stats.right.max;
   } else if (n->stats.right.max != numeric_limits<rlfloat_t>::infinity()) {
      totalStats.right.max = max(totalStats.right.max, n->stats.right.max);
   }   

   if (totalStats.left.min == -numeric_limits<rlfloat_t>::infinity()) {
      totalStats.left.min = n->stats.left.min;
   } else if (n->stats.left.min != -numeric_limits<rlfloat_t>::infinity()) {
      totalStats.left.min = min(totalStats.left.min, n->stats.left.min);
   }
   if (totalStats.left.max == numeric_limits<rlfloat_t>::infinity()) {
      totalStats.left.max = n->stats.left.max;
   } else if (n->stats.left.max != numeric_limits<rlfloat_t>::infinity()) {
      totalStats.left.max = max(totalStats.left.max, n->stats.left.max);
   }
   DOUT << "Pre right count: " << totalStats.right.count << " Left count: " << n->stats.left.count << endl;
   totalStats.left.sum += n->stats.left.sum;
   totalStats.left.sumSq += n->stats.left.sumSq;
   totalStats.left.count += n->stats.left.count;
   totalStats.right.sum -= n->stats.left.sum;
   totalStats.right.sumSq -= n->stats.left.sumSq;
   totalStats.right.count -= n->stats.left.count;

   // Only split on features that are knowable in this decision leaf
   DOUT << "Threshold " << n->threshold << endl;
   if (totalStats.right.count > 0) { 
      rlfloat_t sdr = getSDR(totalStats);
      n->cachedSDR = sdr; // Save this so we can prune the threshold tree later
      DOUT << " SDR: " << sdr << " bestSDR: " << bestScore << endl;
      DOUT << "LMin: " << totalStats.left.min << " LMax: " << totalStats.left.max << endl;
      DOUT << "LCount: " << totalStats.left.count << endl;
      DOUT << " RMin: " << totalStats.right.min << " RMax: " << totalStats.right.max << endl;
      DOUT << "RCount: " << totalStats.right.count << endl;
      DOUT << n << " " << n->left << " " << n->right << " " << n->parent << endl;
      if (sdr > bestScore) {
	 bestThreshold = n->threshold;
	 bestScore = sdr;
	 bestStats = totalStats;
      }
   } else {
      n->cachedSDR = -1; // Impossible value to indicate that it wasn't computed this time
   }
   
   if (n->right) {
      // Similar with min and max
      totalStats.right.min = rmin;
      totalStats.right.max = rmax;
      findBestThresholdHelper(n->right, bestThreshold, bestScore, bestStats, totalStats);
   }
}

FastIncModelTree::Threshold* FastIncModelTree::getSuccessor(Threshold* n) const {
   if (n->right) {
      Threshold* s = n;   
      while (s->left) {
	 s = s->left;
      }
      return s;
   } else {
      Threshold* s = n->parent;
      Threshold* c = n;
      while (s and c == s->right) {
	 c = s;
	 s = s->parent;
      }
      return s;
   }
}

rlfloat_t FastIncModelTree::getSDR(const SplitStats& stats) const {
   size_t total = stats.left.count + stats.right.count;
   rlfloat_t totalRecip = 1.0/total;
   rlfloat_t sdNoSplit = getStdDev(total,
				   stats.left.sum + stats.right.sum,
				   stats.left.sumSq + stats.right.sumSq);
   rlfloat_t sdLeft = getStdDev(stats.left.count, stats.left.sum, stats.left.sumSq);
   rlfloat_t sdRight = getStdDev(stats.right.count, stats.right.sum, stats.right.sumSq);
   DOUT << " sdNoSplit: " << sdNoSplit << " sdLeft: " << sdLeft << " sdRight: " << sdRight << endl;
   rlfloat_t sdr = sdNoSplit - (rlfloat_t(stats.left.count)*totalRecip*sdLeft +
				rlfloat_t(stats.right.count)*totalRecip*sdRight);
   return sdr;
}

rlfloat_t FastIncModelTree::getStdDev(size_t count, rlfloat_t sum, rlfloat_t sqSum) const {
   rlfloat_t countRecip = 1.0/count;
   return sqrt(countRecip*(sqSum - countRecip*sum*sum));
}

void FastIncModelTree::getPrediction(const State& premise, act_t action, State& pred) const {
   Decision* n = getNode(premise, action);
   
   pred.clear();
   pred.push_back(n->predStats.sum/n->predStats.count);
}

rlfloat_t FastIncModelTree::getPredBounds(const State& premise, act_t action, vector<Bound>& predBounds) {
   Decision* n = getNode(premise, action);
   
   predBounds.clear();
   predBounds.push_back({n->predStats.min, n->predStats.max});
   return n->predStats.sum/n->predStats.count;
}

void FastIncModelTree::getPredBounds(const StateBound& premise, const vector<act_t>& action, vector<Bound>& predictionBounds) const {
   getDontKnowPrediction(root_, premise, action, predictionBounds);

   vector<int> inBound(numActions_, 0);
   for (auto a : action) {
      inBound[a] = 1;
   }
}

void FastIncModelTree::getPredDist(const State& premise, act_t action, vector<Normal>& dist) const {
   Decision* n = getNode(premise, action);
   
   dist.clear();
   rlfloat_t var = (n->predStats.sumSq - n->predStats.sum*n->predStats.sum/n->predStats.count)/(n->predStats.count - 1);
   if (var > 1e-6) {
      dist.push_back({n->predStats.sum/n->predStats.count, var});
   } else {
      dist.push_back({n->predStats.sum/n->predStats.count, 0});
   }
}

void FastIncModelTree::getPredSample(const State& premise, act_t action, State& sample) const {
   Decision* n = getNode(premise, action);

   sample.clear();
   rlfloat_t var = (n->predStats.sumSq - n->predStats.sum*n->predStats.sum/n->predStats.count)/(n->predStats.count - 1);
   rlfloat_t s;
   if (var > 1e-6) {
      s = rng_.gaussian(n->predStats.sum/n->predStats.count, sqrt(var));
   } else {
      s = n->predStats.sum/n->predStats.count;
   }
   sample.push_back(s);   
}

FastIncModelTree::Decision* FastIncModelTree::getNode(const State& premise, act_t action) const {
   Decision* n = root_;
   while (n->discriminator) {
      if (n->discriminator->isRight(premise, action)) {
	 n = n->right;
      } else {
	 n = n->left;
      }
   }
   return n;
}

void FastIncModelTree::getDontKnowPrediction(Decision* n,
					     const StateBound& premise,
					     const vector<act_t>& action,
					     vector<Bound>& predictionBounds) const {
   predictionBounds.clear();
   if (!n->discriminator) {
      DOUT << "Leaf: " << n->locStr << endl;
      predictionBounds.push_back({n->predStats.min, n->predStats.max});
      if (n->predStats.min == numeric_limits<rlfloat_t>::infinity() or
	  n->predStats.max == numeric_limits<rlfloat_t>::infinity()) {
	 DOUT << n->locStr << " has un-updated bounds. Count: " << n->predStats.count << endl;
      }
   } else {      
      Discriminator::Answer decision = n->discriminator->isRight(premise, action);
      if (decision == Discriminator::NO) {         // Definitely left
	 getDontKnowPrediction(n->left, premise, action, predictionBounds);
      } else if (decision == Discriminator::YES) { // Definitely right
	 getDontKnowPrediction(n->right, premise, action, predictionBounds);
      } else {                                     // Don't know!
	 Bound unionBound;	 

	 vector<Bound > alteredBound = premise;
	 vector<act_t> alteredActBound = action;
	 n->discriminator->alterBound(alteredBound, alteredActBound, false);
	 vector<Bound > leftBounds;
	 getDontKnowPrediction(n->left, alteredBound, alteredActBound, leftBounds);
	 
	 alteredBound = premise;
	 alteredActBound = action;
	 n->discriminator->alterBound(alteredBound, alteredActBound, false);
	 vector<Bound > rightBounds;
	 getDontKnowPrediction(n->right, alteredBound, alteredActBound, rightBounds);
	 
	 rlfloat_t minS = min(leftBounds[0].lower, rightBounds[0].lower);
	 rlfloat_t maxS = max(leftBounds[0].upper, rightBounds[0].upper);
	 
	 unionBound = {minS, maxS};
	 
	 vector<const vector<Bound>* > bounds {&leftBounds, &rightBounds};
	 string boundNames[] {"left    ", "right   "};	    
	 for (size_t r = 0; r < bounds.size(); ++r) {
	    DOUT << boundNames[r] << " ";
	    for (size_t i = 0; i < bounds[r]->size(); ++i) {
	       DOUT << "(" << (*bounds[r])[i].lower << "," << (*bounds[r])[i].upper << ") ";
	    }
	    DOUT << endl;
	 }
	 DOUT << "union    " << "(" << unionBound.lower << "," << unionBound.upper << ")" << endl;
	 
	 predictionBounds.push_back({0, 0});
	 predictionBounds[0].lower = unionBound.lower;	    
	 predictionBounds[0].upper = unionBound.upper;

	 DOUT << "predict  ";	    
	 DOUT << "(" << predictionBounds[0].lower << "," << predictionBounds[0].upper << ") ";
	 DOUT << endl;	 
      }
   }
}
