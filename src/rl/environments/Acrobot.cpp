#include "Acrobot.hpp"
#include "dout.hpp"
#include <cmath>
#include <math.h>
#include <algorithm>

using namespace std;

Acrobot::Acrobot(bool distractor, RNG& initRNG) :
   distractor_{distractor},
   rng_(initRNG.randomInt()) {
}

void Acrobot::getStatePrediction(const State& premise, size_t action, State& predictions) const {
   // Action 0 - apply -1 torque to the actuated joint
   // Action 1 - apply 0 torque to the actuated joint
   // Action 2 - apply 1 torque to the actuated joint
   
   // 0 theta1 - in radians (min: -pi, max: pi)
   // 1 theta2 - in radians (min: -pi, max: pi)
   // 2 Angular velocity of theta1 (min: -4pi, max: 4pi)
   // 3 Angular velocity of theta2 (min: -8pi, max: 8pi)
   predictions.resize(4);
   rlfloat_t torque = action;
   --torque;

   // Max magnitude of velocities of theta1 and theta2
   rlfloat_t maxVel1 = 4*M_PI;
   rlfloat_t maxVel2 = 9*M_PI;

   vector<rlfloat_t> t = {0, 0.2};
   size_t lenT = t.size();
   
   vector<vector<rlfloat_t>> yOut(4, vector<rlfloat_t>(5, 0)); // The new state at each timestep
   vector<rlfloat_t> augmented = premise;
   if (distractor_) {
      augmented.pop_back(); // Remove the distractor
   }
   augmented.push_back(torque); // premise + action
   yOut[0] = augmented;
   
   vector<rlfloat_t> k1(5);
   vector<rlfloat_t> k2(5);
   vector<rlfloat_t> k3(5);
   vector<rlfloat_t> k4(5);

   // Runge-Kutta approximation of the ODE
   for (size_t i = 0; i < lenT - 1; i++){
      rlfloat_t dt = 0.2;
      rlfloat_t dt2 = dt / 2.0;
      vector <rlfloat_t> y0 = yOut[i];
      // cout << "ks: " << endl;
      k1 = dsdt(y0);
      k2 = dsdt(vectorAdd(y0, scalarMult(k1, dt2)));
      k3 = dsdt(vectorAdd(y0, scalarMult(k2, dt2)));
      k4 = dsdt(vectorAdd(y0, scalarMult(k3, dt)));
      vector <rlfloat_t> intermediate = (vectorAdd(k1, vectorAdd(scalarMult(k2, 2), vectorAdd(scalarMult(k3, 2), k4))));

      yOut[i+1] = vectorAdd(y0, scalarMult(intermediate, (dt/6.0)));
   }

   // Make sure the state values are within the bounds
   predictions[0] = wrap(yOut[lenT-1][0], -M_PI, M_PI);
   predictions[1] = wrap(yOut[lenT-1][1], -M_PI, M_PI);

   predictions[2] = min(max(yOut[lenT-1][2], -maxVel1), maxVel1);
   predictions[3] = min(max(yOut[lenT-1][3], -maxVel2), maxVel2);

   if (distractor_) {
      predictions.push_back((rng_.randomFloat() - 0.5)*8*M_PI); 
   }
}

void Acrobot::getStateBounds(const State& premise, size_t action, State& predictedState, StateBound& predictedBounds) const {
   getStatePrediction(premise, action, predictedState);
   predictedState.clear();
   for (auto d : predictedState) {
      predictedBounds.push_back({d, d});
   }
}

void Acrobot::getStatePredSample(const State& premise, size_t action, State& predictions) const {
   getStatePrediction(premise, action, predictions);
}

rlfloat_t Acrobot::getRewardPrediction(const State&, size_t) const {
   return -1;
}

rlfloat_t Acrobot::getRewardBounds(const State& premise, size_t action, Bound& rewardBound) const {
   rlfloat_t rwd = getRewardPrediction(premise, action);
   rewardBound = {rwd, rwd};
   return rwd;
}

rlfloat_t Acrobot::getRewardPredSample(const State& premise, size_t action) const {
   return getRewardPrediction(premise, action);
}

rlfloat_t Acrobot::getTermPrediction(const State& premise, size_t action) const {
   State nextState;
   getStatePrediction(premise, action, nextState);
   DOUT << "Termination check: " << -cos(nextState[0]) - cos(nextState[1] + nextState[0]) << endl;
   bool terminates = (-cos(nextState[0]) - cos(nextState[1]+nextState[0])) > 1.0;
   return terminates ? 1.0 : 0.0;
}

rlfloat_t Acrobot::getTermBounds(const State& premise, size_t action, Bound& termBound) const {
   rlfloat_t term = getTermPrediction(premise, action);
   termBound = {term, term};
   return term;
}

bool Acrobot::getTermPredSample(const State& premise, size_t action) const {
   return getTermPrediction(premise, action);
}

// Multiply a vector by a scalar
State Acrobot::scalarMult(const vector <rlfloat_t>& vec, const rlfloat_t sca) const{
   vector <rlfloat_t> newVec(5);
   for (int i = 0; i < 5; ++i) {
        newVec[i] = vec[i] * sca;
    }
   return newVec;
}

// Add a vector by a scalar
vector<rlfloat_t> Acrobot::scalarAdd(const vector <rlfloat_t>& vec, rlfloat_t sca) const{
   vector <rlfloat_t> newVec(5);
   for (int i = 0; i < 5; ++i) {
        newVec[i] = vec[i] + sca;
    }
   return newVec;
}

// Adds 2 vectors
vector<rlfloat_t> Acrobot::vectorAdd(const vector <rlfloat_t>& vec1, const vector <rlfloat_t>& vec2) const{
   vector <rlfloat_t> newVec(5);
   for (int i = 0; i < 5; ++i) {
        newVec[i] = vec1[i] + vec2[i];
    }
   return newVec;
}

// calculates the derivates of every element in the state vector
vector<rlfloat_t> Acrobot::dsdt(const vector <rlfloat_t>& state) const{
   rlfloat_t l1 = 1.0;
   rlfloat_t m1 = 1.0;
   rlfloat_t m2 = 1.0;
   rlfloat_t lc1 = 0.5;
   rlfloat_t lc2 = 0.5;
   rlfloat_t i1 = 1.0;
   rlfloat_t i2 = 1.0;
   rlfloat_t g = 9.8;
   rlfloat_t a = state[4];

   rlfloat_t theta1 = state[0];
   rlfloat_t theta2 = state[1];
   rlfloat_t dtheta1 = state[2];
   rlfloat_t dtheta2 = state[3];

   rlfloat_t d1 = m1 * pow(lc1, 2) + m2 * (pow(l1, 2) + pow(lc2, 2) + 2 * l1 * lc2 * cos(theta2)) + i1 + i2;
   // cout << "d1: " << d1 << endl;
   rlfloat_t d2 = m2 * (pow(lc2, 2) + l1 * lc2 * cos(theta2)) + i2; // different
   // cout << "d2: " << d2 << endl;
   rlfloat_t phi2 = m2 * lc2 * g * cos(theta1 + theta2 - M_PI / 2.0);
   // cout << "phi2: " << phi2 << endl;
   rlfloat_t phi1 = -m2 * l1 * lc2 * pow(dtheta2, 2) * sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - M_PI / 2) + phi2;
   // cout << "phi1: " << phi1 << endl;
   rlfloat_t ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * pow(dtheta1,2) * sin(theta2) - phi2) / (m2 * pow(lc2,2) + i2 - pow(d2,2) / d1);
   // cout << "ddtheta2: " << ddtheta2 << endl;
   
   // rlfloat_t ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * pow(lc2,2) + i2 - pow(d2,2) / d1);
   rlfloat_t ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;
   // cout << "ddtheta1: " << ddtheta2 << endl;

   vector<rlfloat_t> derivs = {dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0};
   return derivs;
}

// Make sure the radians are within -pi and pi
rlfloat_t Acrobot::wrap(rlfloat_t val, const rlfloat_t min, const rlfloat_t max) const{
   rlfloat_t diff = max - min;
   while (val > max){
      val = val - diff;
   }
   while (val < min){
      val = val + diff;
   }
   return val;
}
