#include "Discriminator.hpp"

#include <limits>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <sstream>

using namespace std;

ostream& operator<<(ostream& out, const Discriminator& d) {
   out << d.toString();
   return out;
}

bool Discriminator::operator==(const Discriminator& other) const {
  return toString() == other.toString();
}

PropThreshDiscriminator::PropThreshDiscriminator(size_t property, rlfloat_t threshold) : 
   property(property),
   threshold(threshold) {
}

bool PropThreshDiscriminator::isRight(const State& state, act_t) const {
   bool result = state[property] > threshold;
   return result;
}

Discriminator::Answer PropThreshDiscriminator::isRight(const StateBound& state, const vector<act_t>&) const {
   if (state[property].lower > threshold) {
      return YES;
   } else if (state[property].upper <= threshold) {
      return NO;
   } else {
      return DONTKNOW;
   }
}

void PropThreshDiscriminator::alterBound(StateBound& bound, vector<act_t>&, bool isRight) const {
   if (isRight) {    // > threshold
      bound[property].lower = max(threshold, bound[property].lower);
   } else {          // <= threshold
      bound[property].upper = min(threshold, bound[property].upper);
   }
}

string PropThreshDiscriminator::toString() const {
   string s = "[Property " + to_string(property) + " > " + to_string(threshold) + "]";
   return s;
}

Discriminator* PropThreshDiscriminator::clone() const
{
   return new PropThreshDiscriminator(property, threshold);
}

OneHotActionDiscriminator::OneHotActionDiscriminator(act_t act) :
   act(act) {
}

bool OneHotActionDiscriminator::isRight(const State&, act_t action) const {
   return action == act;
}

Discriminator::Answer OneHotActionDiscriminator::isRight(const StateBound&, const vector<act_t>& action) const {
   if (action.size() == 1 and action[0] == act) {
      return YES;
   } else if (find(action.begin(), action.end(), act) != action.end()) {
      return DONTKNOW;
   } else { // act is not amongst the possible actions
      return NO;
   }
}

void OneHotActionDiscriminator::alterBound(StateBound&, vector<act_t>& actSet, bool isRight) const {
   if (isRight) {
      actSet.clear();
      actSet.push_back(act);
   } else {
      auto iter = find(actSet.begin(), actSet.end(), act);
      if (iter != actSet.end()) {
	 actSet.erase(iter);
      }
   }
}

   
string OneHotActionDiscriminator::toString() const {
   string s = "[Action is " + to_string(act) + "]";
   return s;
}

Discriminator* OneHotActionDiscriminator::clone() const
{
   return new OneHotActionDiscriminator(act);
}

