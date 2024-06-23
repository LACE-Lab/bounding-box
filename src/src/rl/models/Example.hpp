#ifndef EXAMPLE_HPP
#define EXAMPLE_HPP

#include "Trajectory.hpp"
#include "RLTypes.hpp"

#include <vector>

class Example
{
  protected:
   const Trajectory& traj;
   unsigned short timestep;
   std::vector<rlfloat_t> outcome; // contains the values we're tracking

  public:
   Example(const Trajectory& traj, std::size_t timeStep);
   virtual ~Example() {}
   virtual const State& getPremiseState() const;
   virtual const State& getOutcome() const;
   virtual act_t getAction() const;
   virtual const Trajectory* getTraj() const;
};

class FullVecExample : public Example
{
  public:
   FullVecExample(const Trajectory& traj, std::size_t timeStep);
};

class PropertyExample : public Example
{
  protected:
   std::size_t propertyIndex;

  public:
   PropertyExample(const Trajectory& traj, std::size_t timeStep, std::size_t propertyIndex);
};

class PropertyChangeExample : public Example
{
  protected:
   std::size_t propertyIndex;

  public:
   PropertyChangeExample(const Trajectory& traj, std::size_t timeStep, std::size_t propertyIndex);
};

class RewardExample : public Example
{
  public:
   RewardExample(const Trajectory& traj, std::size_t timeStep);
};

class TerminalExample : public Example
{
  public:
   TerminalExample(const Trajectory& traj, std::size_t timeStep);
};

class FullOutcomeExample : public Example
{
  public:
   FullOutcomeExample(const Trajectory& traj, std::size_t timeStep);
};

#endif
