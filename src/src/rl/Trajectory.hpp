#ifndef TRAJECTORY
#define TRAJECTORY

#include "RLTypes.hpp"

#include <vector>

class Trajectory
{
  protected:
   std::vector<State> obsData;
   std::vector<act_t> actionData;
   std::vector<rlfloat_t> rewardData;
   std::vector<bool> gameOverData;

  public:
   Trajectory(const State& initialState, bool initGameOver=false, act_t initAction=0);
   virtual ~Trajectory() = default;
   virtual void addStep(act_t action, rlfloat_t reward, const State& state, bool gameOver=false);
   virtual std::size_t getSize() const;
   virtual const State& getPremiseState(std::size_t t) const;
   virtual act_t getAction(std::size_t t) const;
   virtual act_t getPrevAction(std::size_t t) const;
   virtual rlfloat_t getReward(std::size_t t) const;
   virtual bool getGameOver(std::size_t t) const;
   virtual bool getResultGameOver(std::size_t t) const;
   virtual const State& getResultState(std::size_t t) const;
   virtual const State& getCurState() const;
   virtual act_t getCurPrevAction() const;
   virtual void shrinkToFit();
};

#endif
