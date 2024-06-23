#include "Example.hpp"

using namespace std;

Example::Example(const Trajectory& traj, size_t timestep) :
   traj(traj),
   timestep(timestep)
{

}

const State& Example::getPremiseState() const
{
   return traj.getPremiseState(timestep);
}

const State& Example::getOutcome() const
{
   return outcome;
}

act_t Example::getAction() const
{
   return traj.getAction(timestep);
}

const Trajectory* Example::getTraj() const
{
   return &traj;
}

FullVecExample::FullVecExample(const Trajectory& traj, size_t timestep) :
   Example(traj, timestep)   
{
   outcome = traj.getResultState(timestep);
}

PropertyExample::PropertyExample(const Trajectory& traj, size_t timestep, size_t propertyIndex) :
   Example(traj, timestep),
   propertyIndex(propertyIndex)
{
   outcome.resize(1);
   outcome[0] = traj.getResultState(timestep)[propertyIndex];
}

PropertyChangeExample::PropertyChangeExample(const Trajectory& traj, size_t timestep, size_t propertyIndex) :
   Example(traj, timestep),
   propertyIndex(propertyIndex)
{
   outcome.resize(1);
   outcome[0] = traj.getResultState(timestep)[propertyIndex] - traj.getPremiseState(timestep)[propertyIndex];
}

RewardExample::RewardExample(const Trajectory& traj, size_t timestep) :
   Example(traj, timestep)
{
   outcome.resize(1);
   outcome[0] = traj.getReward(timestep);
}

TerminalExample::TerminalExample(const Trajectory& traj, size_t timestep) :
   Example(traj, timestep)
{
   outcome.resize(1);
   outcome[0] = traj.getResultGameOver(timestep);
}

FullOutcomeExample::FullOutcomeExample(const Trajectory& traj, size_t timestep) :
   Example(traj, timestep)
{
   outcome = traj.getResultState(timestep);
   outcome.push_back(traj.getReward(timestep));
   outcome.push_back(traj.getResultGameOver(timestep));
}
