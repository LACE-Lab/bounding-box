#ifndef DISCRIMINATOR_HPP
#define DISCRIMINATOR_HPP

#include "Example.hpp"
#include "RLTypes.hpp"

#include <ostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>

//Abstract class
class Discriminator
{
  public:
   enum Answer {NO = 0, YES = 1, DONTKNOW = 2};

   virtual ~Discriminator() = default;
   virtual bool isRight(const State& state, act_t action) const = 0;
   virtual Answer isRight(const StateBound& state, act_t action) const {return isRight(state, std::vector<act_t>(1, action));}
   virtual Answer isRight(const StateBound& state, const std::vector<act_t>& action) const = 0;   
   virtual void alterBound(StateBound& bound, std::vector<act_t>& actSet, bool isRight) const = 0;
   virtual std::string toString() const = 0;
   virtual bool operator==(const Discriminator& other) const;

   virtual Discriminator* clone() const = 0;

  protected:
   std::string saveString;
};

std::ostream& operator<<(std::ostream& out, const Discriminator& d);

//For a leaf
class NullDiscriminator : public Discriminator
{
  public:
   virtual bool isRight(const State&, act_t) const {return false;}
   virtual Answer isRight(const StateBound&, const std::vector<act_t>&) const {return DONTKNOW;}
   virtual void alterBound(StateBound&, std::vector<act_t>&, bool) const {}
   virtual std::string toString() const {return "[Null]";}
   virtual Discriminator* clone() const {return new NullDiscriminator();}
};

class PropThreshDiscriminator : public Discriminator
{
  protected:
   std::size_t property;
   rlfloat_t threshold;

  public:
   PropThreshDiscriminator(std::size_t property, rlfloat_t threshold);
   virtual bool isRight(const State& state, act_t action) const;
   virtual Answer isRight(const StateBound& state, const std::vector<act_t>& action) const;
   virtual void alterBound(StateBound& bound, std::vector<act_t>& actSet, bool isRight) const;
   virtual std::string toString() const;
   virtual Discriminator* clone() const;
};

class OneHotActionDiscriminator : public Discriminator
{
  public:
   OneHotActionDiscriminator(act_t act);
   virtual bool isRight(const State& state, act_t action) const;
   virtual Answer isRight(const StateBound& state, const std::vector<act_t>& action) const;
   virtual void alterBound(StateBound& bound, std::vector<act_t>& actSet, bool isRight) const;
   virtual std::string toString() const;
   virtual Discriminator* clone() const;

  protected:
   act_t act;
   State dummy;
};

#endif
