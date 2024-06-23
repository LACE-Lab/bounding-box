#ifndef TILE_CODING_Q_FUNCTION
#define TILE_CODING_Q_FUNCTION

#include "QFunction.hpp"
#include "RNG.hpp"

#include <vector>
#include <tuple>

class TileCodingQFunction : public QFunction {
  public:
   TileCodingQFunction(const std::vector<Bound>& dimBounds, const std::vector<size_t>& numDivisions, size_t numTilings, act_t numActions, RNG& initRng);
   virtual ~TileCodingQFunction() = default;

   virtual float getQ(const State& state, act_t action) const;
   virtual void getAllActQs(const State& state, std::vector<float>& qVals) const;

   virtual Bound getQBound(const StateBound& stateBound, act_t action) const;
   virtual void getAllActQBounds(const StateBound& state, std::vector<Bound>& qBounds) const;
   
   virtual void updateQ(const State& state, act_t action, float change);
   virtual float getStepSizeNormalizer() const;

  protected:
   struct IdxBound {
      size_t lower;
      size_t upper;
   };
   using CoordBound = std::vector<IdxBound>;
   
   std::vector<size_t> numDivisions_;
   std::vector<Bound> dimBounds_;
   std::vector<float> cellSize_;
   std::vector<std::vector<float> > offsets_;

   virtual void getCoordinates(const State& state, std::vector<std::vector<size_t> >& coords) const;
   virtual void getBounds(const StateBound& stateBound, std::vector<CoordBound>& bounds) const;   

   class GridWeightManager {
     public:
      GridWeightManager(const std::vector<size_t>& numDivisions, size_t numTilings, act_t numActions);
      ~GridWeightManager();
      float getQ(const std::vector<std::vector<size_t> >& coords, act_t action) const;
      void getAllActQs(const std::vector<std::vector<size_t> >& coords, std::vector<float>& qVals) const;
      Bound getQBound(const std::vector<CoordBound>& bounds, act_t action) const;
      void getAllActQBounds(const std::vector<CoordBound>& bounds, std::vector<Bound>& qBounds) const;      
      void updateQ(const std::vector<std::vector<size_t> >& coords, act_t action, float change);
     private:
      struct TrieNode {
	 ~TrieNode();
	 std::vector<TrieNode*> children;
	 size_t index;
      };
      
      size_t getIndex(const std::vector<size_t>& coord, size_t tiling) const;
      void getWeightBounds(TrieNode* n, const CoordBound& bound, size_t dim, size_t tiling, std::vector<Bound>& wBounds) const;
      
      std::vector<std::vector<float> > weights_;
      std::vector<size_t> numDivisions_;
      act_t numActions_;
      std::vector<TrieNode*> trieRoots_;
   };
   GridWeightManager weights_;
};

#endif
