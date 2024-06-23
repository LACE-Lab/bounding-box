#include "Trajectory.hpp"
#include <iostream>

using namespace std;

Trajectory::Trajectory(const State& initialState, bool initGameOver, act_t initAction)
{
   actionData.push_back(initAction);   
   obsData.push_back(initialState);
   gameOverData.push_back(initGameOver);
}

void Trajectory::addStep(act_t action, rlfloat_t reward, const State& state, bool gameOver)
{
   actionData.push_back(action);
   rewardData.push_back(reward);
   obsData.push_back(state);
   gameOverData.push_back(gameOver);
}

size_t Trajectory::getSize() const
{
   return obsData.size() - 1;
}

const State& Trajectory::getPremiseState(size_t t) const
{
   return obsData[t];
}

act_t Trajectory::getAction(size_t t) const
{
   return actionData[t+1];
}

act_t Trajectory::getPrevAction(size_t t) const
{
   return actionData[t];
}

rlfloat_t Trajectory::getReward(size_t t) const
{
   return rewardData[t];
}

bool Trajectory::getGameOver(size_t t) const
{
   return gameOverData[t];
}

bool Trajectory::getResultGameOver(size_t t) const
{
   return gameOverData[t+1];
}

const State& Trajectory::getResultState(size_t t) const
{
   return obsData[t+1];
}

const State& Trajectory::getCurState() const
{
   return obsData.back();
}

act_t Trajectory::getCurPrevAction() const
{
   return actionData.back();
}

void Trajectory::shrinkToFit()
{
   obsData.shrink_to_fit();
   actionData.shrink_to_fit();
   rewardData.shrink_to_fit();
   gameOverData.shrink_to_fit();
}
