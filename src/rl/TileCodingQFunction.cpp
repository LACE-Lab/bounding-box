#include "TileCodingQFunction.hpp"
#include "dout.hpp"

#include <algorithm>

using namespace std;

TileCodingQFunction::TileCodingQFunction(const vector<Bound>& dimBounds, const vector<size_t>& numDivisions, size_t numTilings, act_t numActions, RNG& initRNG) :
   numDivisions_{numDivisions},
   dimBounds_{dimBounds},
   weights_{numDivisions_, numTilings, numActions} {

   RNG rng(initRNG.randomInt());
   
   // get the cellSize for each dimension
   // cellSize: the size of the of block in the grid
   for (size_t i = 0; i < dimBounds.size(); ++i) {
      cellSize_.push_back((dimBounds[i].upper - dimBounds[i].lower)/numDivisions[i]);
   }

   // Randomly generate offsets for each tiling for each dimension
   offsets_.resize(numTilings);
   for (size_t t = 0; t < numTilings; ++t) {
      offsets_[t].resize(cellSize_.size()); // cellSize_.size is the number of dimensions
      for (size_t i = 0; i < cellSize_.size(); ++i) {
         // only add to offset when we are using the dimension
         if (numDivisions[i] > 1 and numTilings > 1) {
            offsets_[t][i] = -dimBounds[i].lower + rng.randomFloat()*cellSize_[i];
         } else {
            offsets_[t][i] = -dimBounds[i].lower;
         }
	      
      }
   }
}

TileCodingQFunction::GridWeightManager::GridWeightManager(const vector<size_t>& numDivisions, size_t numTilings, act_t numActions) :
   numDivisions_{numDivisions},
   numActions_{numActions} {
   size_t numFeats = numTilings;
   for (auto& d : numDivisions_) {
      if (d > 1) {
	 ++d;
	 numFeats *= d;
      }
   }
   weights_.resize(numFeats, vector<float>(numActions, 0));

   for (size_t i = 0; i < numTilings; ++i) {
      trieRoots_.push_back(new TrieNode);
      trieRoots_.back()->children.resize(numDivisions_[0], nullptr);
      DOUT << this << " Created root " << i << " " << trieRoots_.back() << " " << trieRoots_.back()->children.size() << endl;
   }
}

TileCodingQFunction::GridWeightManager::~GridWeightManager() {
   for (auto r : trieRoots_) {
      delete r;
   }
}

TileCodingQFunction::GridWeightManager::TrieNode::~TrieNode() {
   for (auto c : children) {
      if (c) {
	 delete c;
      }
   }
}

float TileCodingQFunction::getQ(const State& state, act_t action) const {
   vector<vector<size_t> > coords;
   getCoordinates(state, coords);
   DOUT << "Getting Q: ";
   for (auto& c : coords) {
      DOUT << "(";
      for (auto d : c) {
	 DOUT << d << ", ";
      }
      DOUT << ")";      
   }
   DOUT << endl;
   return weights_.getQ(coords, action);
}

void TileCodingQFunction::getAllActQs(const State& state, vector<float>& qVals) const {
   vector<vector<size_t> > coords;
   getCoordinates(state, coords);
   DOUT << "Getting All Qs: ";
   for (auto& c : coords) {
      DOUT << "(";
      for (auto d : c) {
	 DOUT << d << ", ";
      }
      DOUT << ")";      
   }
   DOUT << endl;
   weights_.getAllActQs(coords, qVals);   
}

void TileCodingQFunction::getCoordinates(const State& state, vector<vector<size_t> >& coords) const {
   coords.resize(offsets_.size()); // Number of tilings   
   for (size_t t = 0; t < offsets_.size(); ++t) { // Number of tilings
      coords[t].clear();
      for (size_t i = 0; i < dimBounds_.size(); ++i) { // Number of dimensions
         if (numDivisions_[i] > 1) {
	    float clippedState = min(dimBounds_[i].upper, max(dimBounds_[i].lower, state[i])); // Make the state to be within the bounds
	    int cellID = ceil((clippedState + offsets_[t][i])/cellSize_[i]) - 1;
	    if (cellID < 0) {
	       cellID = 0;
	    }
	    coords[t].push_back(cellID);
	 } else {
	    coords[t].push_back(0);
	 }
      }
   }
}

float TileCodingQFunction::GridWeightManager::getQ(const vector<vector<size_t> >& coords, act_t action) const {
   float q = 0;
   for (size_t i = 0; i < coords.size(); ++i) {
      size_t idx = getIndex(coords[i], i);
      q += weights_[idx][action];
   }
   return q;
}

void TileCodingQFunction::GridWeightManager::getAllActQs(const vector<vector<size_t> >& coords, vector<float>& qVals) const {
   qVals.clear();
   qVals.resize(numActions_, 0);
   for (size_t i = 0; i < coords.size(); ++i) {
      size_t idx = getIndex(coords[i], i);
      for (act_t a = 0; a < numActions_; ++a) {
	 qVals[a] += weights_[idx][a];
      }
   }
}

size_t TileCodingQFunction::GridWeightManager::getIndex(const vector<size_t>& coord, size_t tiling) const {
   size_t idx = tiling;
   for (size_t i = 0; i < coord.size(); ++i) {
      idx *= numDivisions_[i];
      idx += coord[i];
   }
   return idx;
}

Bound TileCodingQFunction::getQBound(const StateBound& stateBound, act_t action) const {
   vector<CoordBound > bounds;
   getBounds(stateBound, bounds);
   return weights_.getQBound(bounds, action);
}

void TileCodingQFunction::getAllActQBounds(const StateBound& stateBound, vector<Bound>& qBounds) const {
   vector<CoordBound > bounds;
   getBounds(stateBound, bounds);
   weights_.getAllActQBounds(bounds, qBounds);
}

void TileCodingQFunction::getBounds(const StateBound& state, vector<CoordBound >& bounds) const {
   bounds.resize(offsets_.size());
   // for each tiling
   for (size_t t = 0; t < offsets_.size(); ++t) {
      bounds[t].clear();
    
      // for each dimension
      for (size_t i = 0; i < dimBounds_.size(); ++i) {
         if (numDivisions_[i] > 1) {
	    auto [minVal, maxVal] = state[i];
	    float clippedMinVal = min(dimBounds_[i].upper, max(dimBounds_[i].lower, minVal));
	    float clippedMaxVal = min(dimBounds_[i].upper, max(dimBounds_[i].lower, maxVal));	 
	    int minID = ceil((clippedMinVal + offsets_[t][i])/cellSize_[i]) - 1;
	    if (minID < 0) {
	       minID = 0;
	    }
	    int maxID = ceil((clippedMaxVal + offsets_[t][i])/cellSize_[i]) - 1;
	    if (maxID < 0) {
	       maxID = 0;
	    }
	    bounds[t].push_back({size_t(minID), size_t(maxID)});
	 } else {
	    bounds[t].push_back({0,0});
	 }
      }
   }
}

Bound TileCodingQFunction::GridWeightManager::getQBound(const vector<CoordBound >& bounds, act_t action) const {
   Bound qBound{0, 0};
   vector<size_t> coord (bounds[0].size());
   
   for (size_t i = 0; i < bounds.size(); ++i) {
      const CoordBound& bound = bounds[i];
      Bound wr{numeric_limits<float>::infinity(), -numeric_limits<float>::infinity()};
      for (size_t j = 0; j < bound.size(); ++j) {
	 coord[j] = bound[j].lower;
      }

      while (coord.back() <= bound.back().upper) {
	 size_t idx = getIndex(coord, i);
	 float w = weights_[idx][action];
	 wr.lower = min(wr.lower, w);
	 wr.upper = max(wr.upper, w);
	 
	 size_t dim = 0;
	 ++coord[dim];
	 while (dim < coord.size()-1 and coord[dim] > bound[dim].upper) {
	    coord[dim] = bound[dim].lower;
	    ++dim;
	    ++coord[dim];
	 }
      }

      qBound.lower += wr.lower;
      qBound.upper += wr.upper;
   }

   return qBound;
}

void TileCodingQFunction::GridWeightManager::getAllActQBounds(const vector<CoordBound >& bounds, vector<Bound>& qBounds) const {
   qBounds.clear();
   qBounds.resize(numActions_, {0, 0});
   vector<size_t> coord (bounds[0].size());
   for (size_t i = 0; i < bounds.size(); ++i) {
      DOUT << "Getting all act bounds for tiling " << i << endl;
      const CoordBound& bound = bounds[i];

      vector<Bound> wrs(numActions_, {numeric_limits<float>::infinity(), -numeric_limits<float>::infinity()});
      TrieNode* n = trieRoots_[i];
      getWeightBounds(n, bound, 0, i, wrs);
      
      for (act_t a = 0; a < numActions_; ++a) {
	 qBounds[a].lower += wrs[a].lower;
	 qBounds[a].upper += wrs[a].upper;
      }
   }
}

void TileCodingQFunction::GridWeightManager::getWeightBounds(TrieNode* n, const CoordBound& bound, size_t dim, size_t tiling, vector<Bound>& wBounds) const {
   if (dim >= bound.size()) {   // Hit a leaf; time to update
      size_t idx = n->index;
      const vector<float>& ws = weights_[idx];
      for (act_t a = 0; a < numActions_; ++a) {
	 wBounds[a].lower = min(wBounds[a].lower, ws[a]);
	 wBounds[a].upper = max(wBounds[a].upper, ws[a]);	 
      }	 
   } else {                     // Internal node
      bool someZeros = false;
      for (size_t i = bound[dim].lower; i <= bound[dim].upper; ++i) {
	 if (n->children[i]) {     // Recur
	    getWeightBounds(n->children[i], bound, dim+1, tiling, wBounds);
	 } else {                  // No child to recur to
	    someZeros = true;
	 }
      }

      if (someZeros) {             // Account for any zero weights in the subtree
	 for (act_t a = 0; a < numActions_; ++a) {
	    wBounds[a].lower = min(wBounds[a].lower, 0.0f);
	    wBounds[a].upper = max(wBounds[a].upper, 0.0f);	 
	 }	 
      }
   }
}
   
void TileCodingQFunction::updateQ(const State& state, act_t action, float change) {
   vector<vector<size_t> > coords;
   getCoordinates(state, coords);

   DOUT << "Updating ";
   for (auto c : coords) {
      DOUT << "(";
      for (auto x : c) {
	 DOUT << x << " ";
      }
      DOUT << ") ";
   }
   DOUT << "Act " << action << " Change " << change << endl;
   weights_.updateQ(coords, action, change);   
}

void TileCodingQFunction::GridWeightManager::updateQ(const vector<vector<size_t> >& coords, act_t action, float change) {
   for (size_t i = 0; i < coords.size(); ++i) {
      size_t idx = getIndex(coords[i], i);
      float& w = weights_[idx][action];
      if (w == 0 and change != 0) {
	 TrieNode* n = trieRoots_[i];
	 for (size_t j = 0; j < coords[i].size(); ++j) {
	    if (n->children.size() == 0) {
	       n->children.resize(numDivisions_[j], nullptr);
	    }
	    TrieNode*& child = n->children[coords[i][j]];
	    if (!child) {
	       child = new TrieNode;
	    }
	    n = child;
	 }
	 if (n == nullptr) { // Should always be true??
	    n = new TrieNode;
	 }
	 n->index = idx;
      }
      w += change;
   }
}

float TileCodingQFunction::getStepSizeNormalizer() const {
   return offsets_.size(); // Number of tilings
}
