#include "NNModel.hpp"
#include "dout.hpp"

#include <memory>
#include <algorithm>

using namespace std;

NNModel::NNModel(size_t inDim, size_t targetDim, act_t numActions, const vector<Bound>& dimBounds, RNG& rng, TrainingType trainType, const Params& params) :
   inDim_{inDim},
   numActions_{numActions},
   targetDim_{targetDim},   
   batchSize_(params.getInt("batch_size")),
   rng_{rng.randomInt()},
   trainType_{trainType},
   varianceSmoothing_(params.getFloat("variance_smoothing")),
   predictChange_(params.getInt("predict_change")) {

   netInDim_ = inDim_ + numActions_;
   if (trainType_ == iqn) {
      netInDim_ += 1;
   }

   size_t netOutDim;
   if (trainType_ == mse or trainType_ == iqn) {
      netOutDim = 1;
   } else if (trainType_ == bound) {
      netOutDim = 3;
   } else if (trainType_ == gaussian) {
      netOutDim = 2;
   }

   vector<size_t> sizes({size_t(params.getInt("hidden_size"))});
   for (size_t i = 0; i < inDim + 2; ++i) {
      nets_.push_back(make_shared<Net>(netInDim_, sizes, netOutDim));
      optimizers_.push_back(make_shared<torch::optim::Adam>(nets_.back()->parameters(), params.getFloat("nn_step_size")));
   }

   for (auto r : dimBounds) {
      // Center on 0
      inputShift_.push_back(-(r.lower + r.upper)/2);
      // Bound -1, 1
      inputScale_.push_back(2/(r.upper - r.lower));
   }
}

NNModel::BBILinear::BBILinear(torch::nn::Linear realModule) :
   realModule_{register_module("realModule_", realModule)},
   inDim_{realModule_->weight.size(1)},
   targetDim_{realModule_->weight.size(0)} {
}

torch::Tensor NNModel::BBILinear::forward(torch::Tensor x) {
   return realModule_->forward(x);
}

torch::Tensor NNModel::BBILinear::bbiForward(torch::Tensor bound) {
   torch::Tensor negWeights = 1^posWeights_;
   torch::Tensor boundMins = bound.index({0, torch::indexing::Slice()});
   torch::Tensor boundMaxs = bound.index({1, torch::indexing::Slice()});
   
   // To get a lower bound on the output   
   // use the min input for positive weights and max input for negative weights
   torch::Tensor minInputs = at::mul(boundMins, posWeights_) + at::mul(boundMaxs, negWeights);
   // To get an upper bound, do the reverse
   torch::Tensor maxInputs = at::mul(boundMins, negWeights) + at::mul(boundMaxs, posWeights_);
   torch::Tensor in = at::stack({minInputs, maxInputs});

   // Element-wise multiplication with weights, then sum to get the dot products
   // (And don't forget the bias!)
   torch::Tensor out = (in*realModule_->weight).sum(2) + realModule_->bias;
      
   return out;
 }

void NNModel::BBILinear::updateBookkeeping() {
   posWeights_ = realModule_->weight > 0;
}

NNModel::Net::Net(size_t inDim, const vector<size_t>& layerDims, size_t targetDim) :
   layers_{register_module("layers_", torch::nn::Sequential())} {
   size_t prevSize = inDim;
   for (auto d : layerDims) {
      layers_->push_back(BBILinear(torch::nn::Linear(prevSize, d)));
      layers_->push_back(BBIReLU(torch::nn::ReLU()));
      prevSize = d;
   }
   layers_->push_back(BBILinear(torch::nn::Linear(prevSize, targetDim)));
}

torch::Tensor NNModel::Net::forward(torch::Tensor x) {
   return layers_->forward(x);
}

torch::Tensor NNModel::Net::bbiForward(torch::Tensor bound) {
   for (size_t i = 0; i < layers_->size(); ++i) {
      bound = (*layers_)[i]->as<BBIModule>()->bbiForward(bound);
   }
   return bound;
}

void NNModel::Net::updateBookkeeping() {
   for (size_t i = 0; i < layers_->size(); ++i) {
      (*layers_)[i]->as<BBIModule>()->updateBookkeeping();
   }
}

NNModel::~NNModel() {
   for (const auto& exList : allExamples_) {
      for (auto ex : exList) {
	 delete ex;
      }
   }
}

void NNModel::addExample(const Trajectory& traj, size_t t) {
   vector<Example*> newEx;
   for (size_t i = 0; i < inDim_; ++i) {
      if (predictChange_) {
	 newEx.push_back(new PropertyChangeExample(traj, t, i));
      } else {
	 newEx.push_back(new PropertyExample(traj, t, i));
      }
   }
   newEx.push_back(new RewardExample(traj, t));
   newEx.push_back(new TerminalExample(traj, t));
   allExamples_.push_back(newEx);
}

void NNModel::removeTrajectory(const Trajectory* traj) {
   auto exIter = allExamples_.begin();
   while (exIter != allExamples_.end()) {
      if ((*exIter)[0]->getTraj() == traj) {
	 for (auto ex : *exIter) {
	    delete ex;
	 }
         exIter = allExamples_.erase(exIter);
      } else {
         ++exIter;
      }
   }
}

void NNModel::updatePredictions() {
   for (auto net : nets_) {
      net->train();
   }
   
   vector<double> batchInVec;
   vector<vector<double> > batchTargetVecs(nets_.size());
   for (size_t i = 0; i < batchSize_; ++i) {
      size_t randIdx = rng_.randomFloat() * allExamples_.size();
      const vector<Example*>& exList = allExamples_[randIdx];

      vector<double> inVec;
      prepareInputVector(exList[0]->getPremiseState(), exList[0]->getAction(), inVec);
      if (trainType_ == iqn) {
	 inVec.push_back(rng_.randomFloat());
      }
      
      DOUT << "Batch input " << i << ": ";
      for (auto x : inVec) {
	 DOUT << x << " ";
      }
      DOUT << endl;

      batchInVec.insert(batchInVec.end(), inVec.begin(), inVec.end());

      for (size_t i = 0; i < nets_.size(); ++i) {
	 double outcome = exList[i]->getOutcome()[0];
	 batchTargetVecs[i].push_back(outcome);
      }
   }

   for (size_t i = 0; i < nets_.size(); ++i) {
      DOUT << "Model " << i << endl;

      optimizers_[i]->zero_grad();

      torch::Tensor batchIn = torch::tensor(batchInVec);
      batchIn = at::reshape(batchIn, {-1, static_cast<long long>(netInDim_)});      

      torch::Tensor batchOut = nets_[i]->forward(batchIn);      
      DOUT << "Batch output: " << endl;
      DOUT << batchOut << endl;
      
      torch::Tensor batchTarget = torch::tensor(batchTargetVecs[i]);
      batchTarget = at::unsqueeze(batchTarget, 1);

      DOUT << "Batch target: " << endl;
      DOUT << batchTarget << endl;

      torch::Tensor loss;
      if (trainType_ == mse) {
	 loss = torch::mse_loss(batchOut, batchTarget);
      } else if (trainType_ == bound) {
	 torch::Tensor meanPred = at::unsqueeze(batchOut.index({torch::indexing::Slice(), 1}), 1);
	 torch::Tensor mseLoss = torch::mse_loss(meanPred, batchTarget);
	 DOUT << "Batch MSE: " << mseLoss.item<double>() << endl;      
	 
	 torch::Tensor maxPred = at::unsqueeze(batchOut.index({torch::indexing::Slice(), 2}), 1);
	 torch::Tensor maxErr = batchTarget - maxPred;
	 torch::Tensor q95Loss = torch::sum(torch::maximum(-0.05*maxErr, 0.95*maxErr));
	 DOUT << "Batch Q95: " << q95Loss.item<double>() << endl;      
	 
	 torch::Tensor minPred = at::unsqueeze(batchOut.index({torch::indexing::Slice(), 0}), 1);
	 torch::Tensor minErr = batchTarget - minPred;
	 torch::Tensor q5Loss = torch::sum(torch::maximum(-0.95*minErr, 0.05*minErr));      
	 DOUT << "Batch Q05: " << q5Loss.item<double>() << endl;
      
	 loss = mseLoss + q95Loss + q5Loss;
      } else if (trainType_ == iqn) {
	 torch::Tensor pred = batchOut;
	 torch::Tensor quantiles = at::unsqueeze(batchIn.index({torch::indexing::Slice(), int(netInDim_-1)}), 1);
	 torch::Tensor residual  = batchTarget - pred;
	 loss = torch::sum(torch::maximum((quantiles-1)*residual, quantiles*residual));	 
      } else if (trainType_ == gaussian) {
	 torch::Tensor meanPred = at::unsqueeze(batchOut.index({torch::indexing::Slice(), 0}), 1);
	 torch::Tensor varPred = at::unsqueeze(at::relu(batchOut.index({torch::indexing::Slice(), 1})) + varianceSmoothing_, 1);
	 DOUT << "varPred:" << endl;
	 DOUT << varPred;
	 torch::Tensor residual = meanPred - batchTarget;	 
	 loss = torch::sum(0.5*at::log(varPred) + (residual*residual)/(2*varPred));
      }

      DOUT << "Loss: " << loss.item<double>() << endl;
      loss.backward();
      optimizers_[i]->step();

      if (trainType_ == bound) {
	 nets_[i]->updateBookkeeping();
      }
   }
}

void NNModel::getStateBounds(const StateBound& premise, const vector<act_t>& action, StateBound& predictedBounds) const {
   c10::InferenceMode guard;
   for (size_t i = 0; i < inDim_; ++i) {
      nets_[i]->eval();
   }

   torch::Tensor in = prepareInput(premise, action);   
   predictedBounds.clear();

   for (size_t i = 0; i < inDim_; ++i) {
      torch::Tensor out = nets_[i]->bbiForward(in);
      predictedBounds.push_back({out[0][0].item<rlfloat_t>(), out[1][2].item<rlfloat_t>()});
      // Early in training, the quantiles might not be fully baked yet
      if (predictedBounds.back().upper < predictedBounds.back().lower) {
	 DOUT << "Invalid bound in dim " << i << ": " << predictedBounds.back().lower << "," << predictedBounds.back().upper << "." << endl;
	 swap(predictedBounds.back().lower, predictedBounds.back().upper);
      }
      
      if (predictChange_) {
	 predictedBounds.back().lower += premise[i].lower;
	 predictedBounds.back().upper += premise[i].upper;
      }
   }
}

void NNModel::getRewardBounds(const StateBound& premise, const vector<act_t>& action, Bound& rewardBound) const {
   c10::InferenceMode guard;
   nets_[inDim_]->eval();

   torch::Tensor in = prepareInput(premise, action);

   torch::Tensor out = nets_[inDim_]->bbiForward(in);
   rewardBound = {out[0][0].item<rlfloat_t>(), out[1][2].item<rlfloat_t>()};

   // Early in training, the quantiles might not be fully baked yet
   if (rewardBound.upper < rewardBound.lower) {
      swap(rewardBound.lower, rewardBound.upper);
   }   
}

void NNModel::getTermBounds(const StateBound& premise, const vector<act_t>& action, Bound& termBound) const {
   c10::InferenceMode guard;
   nets_[inDim_+1]->eval();

   torch::Tensor in = prepareInput(premise, action);

   torch::Tensor out = nets_[inDim_+1]->bbiForward(in);
   termBound = {out[0][0].item<rlfloat_t>(), out[1][2].item<rlfloat_t>()};

   // Early in training, the quantiles might not be fully baked yet
   if (termBound.upper < termBound.lower) {
      swap(termBound.lower, termBound.upper);
   }   
}

void NNModel::getStatePrediction(const State& premise, act_t action, State& predictions) const {
   c10::InferenceMode guard;
   for (size_t i = 0; i < inDim_; ++i) {
      nets_[i]->eval();
   }

   torch::Tensor in = prepareInput(premise, action);

   predictions.clear();
   for (size_t i = 0; i < inDim_; ++i) {
      torch::Tensor out = nets_[i]->forward(in);
      if (trainType_ == bound) {
	 predictions.push_back(out[0][1].item<rlfloat_t>());
      } else {
	 predictions.push_back(out[0][0].item<rlfloat_t>());
      }

      if (predictChange_) {
	 predictions.back() += premise[i];
      }
   }
}

void NNModel::getStateBounds(const State& premise, act_t action, State& predictedState, StateBound& predictedBounds) const {
   c10::InferenceMode guard;
   for (size_t i = 0; i < inDim_; ++i) {
      nets_[i]->eval();
   }

   torch::Tensor in = prepareInput(premise, action);
   predictedState.clear();
   predictedBounds.clear();
   
   for (size_t i = 0; i < inDim_; ++i) {
      torch::Tensor out = nets_[i]->forward(in);
      predictedState.push_back({out[0][1].item<rlfloat_t>()});
      predictedBounds.push_back({out[0][0].item<rlfloat_t>(), out[0][2].item<rlfloat_t>()});

      // Early in training, the quantiles might not be fully baked yet
      if (predictedBounds.back().upper < predictedBounds.back().lower) {
	 DOUT << "Invalid bound in dim " << i << ": " << predictedBounds.back().lower << "," << predictedBounds.back().upper << "." << endl;
	 swap(predictedBounds.back().lower, predictedBounds.back().upper);
      }

      if (predictChange_) {
	 predictedBounds.back().lower += premise[i];
	 predictedBounds.back().upper += premise[i];
      }      
   }
}

rlfloat_t NNModel::getRewardPrediction(const State& premise, act_t action) const {
   c10::InferenceMode guard;
   nets_[inDim_]->eval();

   torch::Tensor in = prepareInput(premise, action);
   torch::Tensor out = nets_[inDim_]->forward(in);

   if (trainType_ == bound) {
      return out[0][1].item<rlfloat_t>();
   } else {
      return out[0][0].item<rlfloat_t>();
   }
}

rlfloat_t NNModel::getRewardBounds(const State& premise, act_t action, Bound& rewardBound) const {
   c10::InferenceMode guard;
   nets_[inDim_]->eval();

   torch::Tensor in = prepareInput(premise, action);
   torch::Tensor out = nets_[inDim_]->forward(in);
   rewardBound = {out[0][0].item<rlfloat_t>(), out[0][2].item<rlfloat_t>()};

   // Early in training, the quantiles might not be fully baked yet
   if (rewardBound.upper < rewardBound.lower) {
      swap(rewardBound.lower, rewardBound.upper);
   }
   
   return out[0][1].item<rlfloat_t>();
}

rlfloat_t NNModel::getTermPrediction(const State& premise, act_t action) const {
   c10::InferenceMode guard;
   nets_[inDim_+1]->eval();

   torch::Tensor in = prepareInput(premise, action);
   torch::Tensor out = nets_[inDim_+1]->forward(in);

   if (trainType_ == bound) {
      return out[0][1].item<rlfloat_t>();
   } else {
      return out[0][0].item<rlfloat_t>();
   }
}

rlfloat_t NNModel::getTermBounds(const State& premise, act_t action, Bound& termBound) const {
   c10::InferenceMode guard;
   nets_[inDim_+1]->eval();

   torch::Tensor in = prepareInput(premise, action);
   torch::Tensor out = nets_[inDim_+1]->forward(in);
   termBound = {out[0][0].item<rlfloat_t>(), out[0][2].item<rlfloat_t>()};

   // Early in training, the quantiles might not be fully baked yet
   if (termBound.upper < termBound.lower) {
      swap(termBound.lower, termBound.upper);
   }

   return out[0][1].item<rlfloat_t>();   
}   

void NNModel::getStateDistribution(const State& premise, act_t action, StateNormal& stateDist) const {
   c10::InferenceMode guard;
   for (size_t i = 0; i < inDim_; ++i) {
      nets_[i]->eval();
   }

   stateDist.clear();

   if (trainType_ == gaussian) {
      torch::Tensor in = prepareInput(premise, action);
      for (size_t i = 0; i < inDim_; ++i) {
	 torch::Tensor out = nets_[i]->forward(in);
	 rlfloat_t mean = out[0][0].item<rlfloat_t>();
	 rlfloat_t var = at::relu(out[0][1]).item<rlfloat_t>() + varianceSmoothing_;
	 stateDist.push_back({mean, var});

	 if (predictChange_) {
	    stateDist.back().mean += premise[i];
	 }
      }
   } else { // Not supported otherwise
      vector<rlfloat_t> pred;
      getStatePrediction(premise, action, pred);
      for (size_t i = 0; i < inDim_; ++i) {
	 stateDist.push_back({pred[i], 0});

	 if (predictChange_) {
	    stateDist.back().mean += premise[i];
	 }
      }
   }
}

void NNModel::getRewardDistribution(const State& premise, act_t action, Normal& rewardDist) const {
   c10::InferenceMode guard;
   nets_[inDim_]->eval();

   if (trainType_ == gaussian) {
      torch::Tensor in = prepareInput(premise, action);
      torch::Tensor out = nets_[inDim_]->forward(in);
      rlfloat_t mean = out[0][0].item<rlfloat_t>();
      rlfloat_t var = at::relu(out[0][1]).item<rlfloat_t>() + varianceSmoothing_;
      rewardDist = {mean, var};
   } else {
      rewardDist = {getRewardPrediction(premise, action), 0};
   }
}

void NNModel::getTermDistribution(const State& premise, act_t action, Normal& termDist) const {
   c10::InferenceMode guard;
   nets_[inDim_+1]->eval();

   if (trainType_ == gaussian) {
      torch::Tensor in = prepareInput(premise, action);
      torch::Tensor out = nets_[inDim_+1]->forward(in);
      rlfloat_t mean = out[0][0].item<rlfloat_t>();
      rlfloat_t var = at::relu(out[0][1]).item<rlfloat_t>() + varianceSmoothing_;
      termDist = {mean, var};
   } else {
      termDist = {getTermPrediction(premise, action), 0};
   }   
}

void NNModel::getStatePredSample(const State& premise, act_t action, State& predictions) const {
   c10::InferenceMode guard;
   for (size_t i = 0; i < inDim_; ++i) {
      nets_[i]->eval();
   }

   predictions.clear();

   if (trainType_ == gaussian) {
      StateNormal stateDist;
      getStateDistribution(premise, action, stateDist);
      for (size_t i = 0; i < inDim_; ++i) {
	 predictions.push_back(rng_.gaussian(stateDist[i].mean, sqrt(stateDist[i].var)));

	 if (predictChange_) {
	    predictions.back() += premise[i];
	 }
      }      
   } else if (trainType_ == iqn) {
      vector<double> inputVec;
      prepareInputVector(premise, action, inputVec);
      for (size_t i = 0; i < inDim_; ++i) {
	 inputVec.push_back(rng_.randomFloat());
	 torch::Tensor in = at::unsqueeze(torch::tensor(inputVec), 0);
	 torch::Tensor out = nets_[i]->forward(in);
	 predictions.push_back(out[0][0].item<rlfloat_t>());
	 inputVec.pop_back();

	 if (predictChange_) {
	    predictions.back() += premise[i];
	 }
      }      
   } else {
      getStatePrediction(premise, action, predictions);
   }
}

rlfloat_t NNModel::getRewardPredSample(const State& premise, act_t action) const {
   c10::InferenceMode guard;
   nets_[inDim_]->eval();

   if (trainType_ == gaussian) {
      Normal dist;
      getRewardDistribution(premise, action, dist);
      return rng_.gaussian(dist.mean, sqrt(dist.var));
   } else if (trainType_ == iqn) {
      vector<double> inputVec;
      prepareInputVector(premise, action, inputVec);
      inputVec.push_back(rng_.randomFloat());      
      torch::Tensor in = at::unsqueeze(torch::tensor(inputVec), 0);
      torch::Tensor out = nets_[inDim_]->forward(in);
      return out[0][0].item<rlfloat_t>();
   } else {
      return getRewardPrediction(premise, action);
   }
}

bool NNModel::getTermPredSample(const State& premise, act_t action) const {
   c10::InferenceMode guard;
   nets_[inDim_+1]->eval();

   if (trainType_ == gaussian) {
      Normal dist;
      getTermDistribution(premise, action, dist);
      return rng_.gaussian(dist.mean, sqrt(dist.var));
   } else if (trainType_ == iqn) {
      vector<double> inputVec;
      prepareInputVector(premise, action, inputVec);
      inputVec.push_back(rng_.randomFloat());
      torch::Tensor in = at::unsqueeze(torch::tensor(inputVec), 0);
      torch::Tensor out = nets_[inDim_+1]->forward(in);
      return out[0][0].item<rlfloat_t>() >= 0.5;
   } else {
      return getTermPrediction(premise, action) >= 0.5;
   }
}

void NNModel::prepareInputVector(const State& premise, act_t action, vector<double>& inVec) const {
   for (size_t i = 0; i < inDim_; ++i) {
      inVec.push_back((premise[i] + inputShift_[i])*inputScale_[i]);
   }
   inVec.resize(inDim_ + numActions_);
   for (size_t i = inDim_; i < inDim_ + numActions_; ++i) {
      inVec[i] = -1;
   }
   inVec[inDim_ + action] = 1;   
}

torch::Tensor NNModel::prepareInput(const State& premise, act_t action) const {
   vector<double> inVec;
   prepareInputVector(premise, action, inVec);
   return at::unsqueeze(torch::tensor(inVec), 0);
}

void NNModel::prepareTargetVector(const State& result, rlfloat_t reward, rlfloat_t term, vector<double>& targetVec) const {
   targetVec.clear();
   for (auto d : result) {
      targetVec.push_back(d);
   }
   targetVec.resize(targetDim_ + 2);
   targetVec[targetDim_] = reward;
   targetVec[targetDim_ + 1] = term;
}

torch::Tensor NNModel::prepareTarget(const State& result, rlfloat_t reward, rlfloat_t term) const {
   vector<double> targetVec;
   prepareTargetVector(result, reward, term, targetVec);
   return at::unsqueeze(torch::tensor(targetVec), 0);
}

torch::Tensor NNModel::prepareInput(const StateBound& premise, const vector<act_t>& action) const {
   torch::Tensor in = -torch::ones({3, static_cast<long long>(inDim_ + numActions_)});
   for (size_t i = 0; i < inDim_; ++i) {
      in[0][i] = (premise[i].lower + inputShift_[i])*inputScale_[i];
      in[1][i] = (premise[i].upper + inputShift_[i])*inputScale_[i];
   }

   for (auto act : action) {
      in[1][inDim_ + act] = 1;
   }
   
   if (action.size() == 1) {
      in[0][inDim_ + action[0]] = 1;
   }
   
   return in;
}
