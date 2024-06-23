#ifndef FAST_INC_MODEL_TREE
#define FAST_INC_MODEL_TREE

#include "Params.hpp"
#include "Example.hpp"
#include "Discriminator.hpp"
#include "RNG.hpp"
#include "RLTypes.hpp"

#include <vector>
#include <tuple>

// TODO: Could extend this to multi-dimensional targets and linear leaf models
class FastIncModelTree {
  public:
   FastIncModelTree(size_t inDim, act_t numActions, RNG& initRNG, const Params& params);
   virtual ~FastIncModelTree();

   virtual void addExample(Example* ex);
   virtual void split();

   // Predicts the expected value of the outcome from an input
   virtual void getPrediction(const State& premise, act_t action, State& pred) const;
   // Predicts the bound of the outcome value from an input
   virtual rlfloat_t getPredBounds(const State& premise, act_t action, std::vector<Bound>& predBounds);
   // Predicts the bound of the outcome value from a bound input
   virtual void getPredBounds(const StateBound& premise, const std::vector<act_t>& action, std::vector<Bound>& predictionBounds) const;
   // Gives a mean and variance of the outcome value from an input
   virtual void getPredDist(const State& premise, act_t action, std::vector<Normal>& dist) const;
   // Samples an outcome from an input
   virtual void getPredSample(const State& premise, act_t action, State& sample) const;

  protected:
   struct Stats {
      Stats();
      
      rlfloat_t sum;
      rlfloat_t sumSq;
      rlfloat_t min;
      rlfloat_t max;
      size_t count;
   };
   
   struct SplitStats {
      Stats left;
      Stats right;
   };
   
   struct Threshold {
      Threshold(rlfloat_t threshold, Threshold* parent);
      ~Threshold();

      rlfloat_t threshold;
      
      Threshold* left;
      Threshold* right;
      Threshold* parent;

      SplitStats stats;
      rlfloat_t hereAndRightMin;
      rlfloat_t hereAndRightMax;
      rlfloat_t cachedSDR;
   };
   
   struct Decision {
      Decision(size_t inDim, act_t numActions, std::string locStr);
      ~Decision();

      std::string locStr;
      
      Decision* left;
      Decision* right;

      Stats predStats;

      size_t splitCount;      
      std::vector<Threshold*> threshRoots;
      std::vector<SplitStats> actionSplits;
      
      Discriminator* discriminator;
   };

   size_t inDim_;
   act_t numActions_;
   Decision* root_;
   size_t numLeaves_;
   size_t maxLeaves_;
   rlfloat_t confidence_;
   rlfloat_t tieThreshold_;
   mutable RNG rng_;
   
   virtual void addExampleHelper(Decision* n, Example* ex);
   virtual void insertThreshold(Threshold*& n, Threshold* parent, rlfloat_t key, rlfloat_t outcome);

   virtual void split(Decision* n);
   virtual void findBestThreshold(Threshold* root,
				  rlfloat_t& bestThreshold,
				  rlfloat_t& bestScore,
				  SplitStats& bestStats) const;
   virtual void findBestThresholdHelper(Threshold* n,
					rlfloat_t& bestThreshold,
					rlfloat_t& bestScore,
					SplitStats& bestStats,
					SplitStats totalStats) const;
   virtual Threshold* getSuccessor(Threshold* n) const;
   virtual rlfloat_t getSDR(const SplitStats& stats) const;
   virtual rlfloat_t getStdDev(size_t count, rlfloat_t sum, rlfloat_t sqSum) const;

   virtual Decision* getNode(const State& premise, act_t action) const;

   virtual void getDontKnowPrediction(Decision* n,
				      const StateBound& premise,
				      const std::vector<act_t>& action,
				      std::vector<Bound>& predictionBounds) const;
};

#endif
