#ifndef Q_FUNCTION
#define Q_FUNCTION

#include "RLTypes.hpp"

#include <vector>
#include <tuple>

class QFunction {
  public:
   virtual ~QFunction() = default;
   virtual float getQ(const State& state, act_t action) const = 0;
   virtual void getAllActQs(const State& state, std::vector<float>& qVals) const = 0;
   
   virtual Bound getQBound(const StateBound& stateBound, act_t action) const = 0;
   virtual void getAllActQBounds(const StateBound& state, std::vector<Bound>& qBounds) const = 0;   

   virtual void updateQ(const State& state, act_t action, float change) = 0;
   virtual float getStepSizeNormalizer() const = 0;
};

class SumQ : public QFunction
{
  public:
   SumQ(const std::vector<QFunction*>& qFuncs, act_t numActions);
   virtual ~SumQ();
   
   virtual float getQ(const State& state, act_t action) const;
   virtual void getAllActQs(const State& state, std::vector<float>& qVals) const;

   virtual Bound getQBound(const StateBound& stateBound, act_t action) const;
   virtual void getAllActQBounds(const StateBound& state, std::vector<Bound>& qBounds) const;
   
   virtual void updateQ(const State& state, act_t action, float change);   
   virtual float getStepSizeNormalizer() const;

  protected:
   std::vector<QFunction*> qFuncs_;
   act_t numActions_;
};

#endif
