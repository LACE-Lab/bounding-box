#include "QLearner.hpp"
#include "TileCodingQFunction.hpp"
#include "Trajectory.hpp"
#include "NNModel.hpp"
#include "IncDTModel.hpp"
#include "RLTypes.hpp"
#include "MountainCar.hpp"
#include "Acrobot.hpp"
#include "GoRight.hpp"
#include "dout.hpp"
#include "Params.hpp"
#include "RNG.hpp"

#include <iostream>
#include <ctime>
#include <string>
#include <fstream>
#include <cxxopts.hpp>
#include <chrono>
#include <algorithm>
#include <torch/torch.h>

using namespace std;

void parseParams(int argc, char* argv[], Params& params) {
   // Consider switching to CLI11?
   cxxopts::Options options("planning", "Run experiments with selective MVE planning.");
   options.add_options()
      // General
      ("n,num_frames", "Number of Frames", cxxopts::value<size_t>()->default_value("300000"))
      ("d,seed", "Seed", cxxopts::value<size_t>()->default_value("0"))
      ("game", "Game Environment Used", cxxopts::value<string>()->default_value("GR"))
      ("o,output", "Output filename", cxxopts::value<std::string>()->default_value("test"))
      ("gen_config", "Generate a config file for this run", cxxopts::value<bool>()->default_value("false"))
      ("c, config", "Filename of config file to use for settings", cxxopts::value<string>())

      // Go Right
      ("gor_length", "Length of GoRight hallway", cxxopts::value<size_t>()->default_value("10"))
      ("gor_num_ind", "Number of GoRight reward indicators", cxxopts::value<size_t>()->default_value("2"))
      ("gor_prize_mult", "GoRight prize reward multiplier", cxxopts::value<double>()->default_value("1.0"))

      // RL
      ("p,planner", "Planner to use", cxxopts::value<string>()->default_value("P"))
      ("a,step_size", "Step size", cxxopts::value<double>()->default_value("1e-1"))
      ("e,exploration_rate", "The behavior policy's exploration rate", cxxopts::value<double>()->default_value("1"))
      ("g,discount", "Discount Factor", cxxopts::value<double>()->default_value("0.9"))
      ("sparse_weights", "Use a sparse representation of the q-function weights", cxxopts::value<bool>()->default_value("false"))
      ("h,horizon", "Horizon", cxxopts::value<size_t>()->default_value("5"))
      ("m,temperature", "Temperature", cxxopts::value<double>()->default_value("1e-1"))
      ("y,decay", "Decay Factor", cxxopts::value<double>()->default_value("1"))
      ("k,num_samples", "Number of MC Samples", cxxopts::value<size_t>()->default_value("10"))

      // Decision Tree
      ("update_every", "Split every", cxxopts::value<size_t>()->default_value("100"))
      ("max_leaves", "Maximum leaves", cxxopts::value<size_t>()->default_value(to_string(numeric_limits<long long>::max())))
      ("predict_change", "Predict the change of state rather than the next state", cxxopts::value<bool>()->default_value("false"))
      ("split_confidence", "Confidence level for incremental splits", cxxopts::value<double>()->default_value("0.05"))
      ("tie_threshold", "Similarity level at which two splits are considered tied", cxxopts::value<double>()->default_value("0.05"))

      // Neural Network
      ("use_nn", "Use a NN model", cxxopts::value<bool>()->default_value("false"))
      ("hidden_size", "Size of the hidden layer in NN models", cxxopts::value<size_t>()->default_value("128"))
      ("nn_step_size", "Step size for the NN optimizer", cxxopts::value<double>()->default_value("1e-3"))
      ("batch_size", "Batch size for NN training", cxxopts::value<size_t>()->default_value("16"))
      ("use_gaussian", "Use heteroschedastic Gaussian instead of IQN for sampling", cxxopts::value<bool>()->default_value("false"))
      ("variance_smoothing", "Smoothing value for heteroschedastic variance", cxxopts::value<double>()->default_value("1e-6"));
      
   auto result = options.parse(argc, argv);

   vector<string> strNames({"game",
	                    "planner",
                   	    "output"});

   vector<string> floatNames({"gor_prize_mult",
	                      "split_confidence",
			      "tie_threshold",
                   	      "nn_step_size",
			      "variance_smoothing",
			      "step_size",
			      "exploration_rate",
			      "discount",
			      "temperature",
			      "decay"});

   vector<string> sizeNames({"gor_length",
	                    "gor_num_ind",
			    "num_frames",
			    "seed",
			    "update_every",
			    "max_leaves",
			    "hidden_size",
			    "batch_size",
			    "horizon",
			    "num_samples"});

   vector<string> boolNames({"predict_change",
			     "use_nn",
			     "use_gaussian",
			     "sparse_weights"});	    
   
   if (result.count("config")) {
      ifstream configIn(result["config"].as<string>());

      if (configIn.is_open()) {
	 string name;
	 string dummy;
	 // TODO: Falls apart with unrecognized variable name
	 while (configIn >> name) {	   
	    configIn >> dummy; // Consume the =
	    if (find(strNames.begin(), strNames.end(), name) != strNames.end()) {
	       string val;
	       configIn >> val;
	       params.setStr(name, val);	   
	    } else if (find(floatNames.begin(), floatNames.end(), name) != floatNames.end()) {
	       double val;
	       configIn >> val;
	       params.setFloat(name, val);
	    } else if (find(sizeNames.begin(), sizeNames.end(), name) != sizeNames.end()) {
	       size_t val;
	       configIn >> val;
	       params.setInt(name, val);
	    } else if (find(boolNames.begin(), boolNames.end(), name) != boolNames.end()) {
	       bool val;
	       configIn >> val;
	       params.setInt(name, val);
	    } else {
	       cerr << "Warning: ignoring unrecognized config file option: " << name << endl;
	       cerr << "in given config file: " << result["config"].as<string>() << endl;
	    }
	 }
      } else {
	 cerr << "Could not open config file for reading: " << result["config"].as<string>() << endl;
	 exit(1);
      }
   }

   // If config file is given, explicitly provided parameters override
   for (auto name : strNames) {
      if (result.count(name) or !params.isSet(name)) {
	 params.setStr(name, result[name].as<string>());
      }
   }
   for (auto name : floatNames) {
      if (result.count(name) or !params.isSet(name)) {
	 params.setFloat(name, result[name].as<double>());
      }
   }
   for (auto name : sizeNames) {
      if (result.count(name) or !params.isSet(name)) {
	 params.setInt(name, result[name].as<size_t>());
      }
   }
   for (auto name : boolNames) {
      if (result.count(name) or !params.isSet(name)) {
	 params.setInt(name, result[name].as<bool>());
      }
   }

   // Planner choice overrides some parameter values, even if explicitly provided
   string planner = params.getStr("planner");

   if (planner == "S" or
       planner == "IS" or
       planner == "E" or
       planner == "IE" or
       planner == "A" or
       planner == "P" or
       planner == "Q") {
      params.setInt("num_samples", 1);
      params.setFloat("temperature", numeric_limits<double>::infinity());
   }

   if (planner == "RR" or
       planner == "IRR" or
       planner == "RV" or
       planner == "IRV") {
      params.setInt("inc_rwd", 1);
      params.setInt("inc_state", 0);
   } else if (planner == "SRR" or
	      planner == "ISRR" or
	      planner == "SRV" or
	      planner == "ISRV") {
      params.setInt("inc_rwd", 1);
      params.setInt("inc_state", 1);
   } else { // SR, SV, ISR, ISV, and others that don't matter
      params.setInt("inc_rwd", 0);
      params.setInt("inc_state", 1);
   }

   if (planner == "MCTV" or
       planner == "IMCTV" or
       planner == "RV" or
       planner == "IRV" or
       planner == "SRV" or
       planner == "ISRV") {
      params.setInt("use_variance", 1);
   } else {
      params.setInt("use_variance", 0);
   }

   if (planner == "ITDR" or
       planner == "TDR" or
       planner == "IMCTDR" or
       planner == "MCTDR" or
       planner == "IDTOR" or
       planner == "DTOR" or
       planner == "IMCTDOR" or
       planner == "MCTDOR") {
      params.setInt("directional_range", 1);
   } else {
      params.setInt("directional_range", 0);
   }
   
   if (planner == "ITOR" or
       planner == "TOR" or
       planner == "IMCTOR" or
       planner == "MCTOR" or
       planner == "ITDOR" or
       planner == "TDOR" or
       planner == "IMCTDOR" or
       planner == "MCTDOR") {
      params.setInt("reject_overlap", 1);
   } else {
      params.setInt("reject_overlap", 0);
   }

   // Generate a config file if requested
   if (result["gen_config"].as<bool>()) {
      string configFilename = params.getStr("output") + ".config";
      ofstream configOut(configFilename);
      if (configOut.is_open()) {
	 for (auto name : strNames) {
	    configOut << name << " = " << params.getStr(name) << endl;
	 }
	 for (auto name : floatNames) {
	    configOut << name << " = " << params.getFloat(name) << endl;
	 }
	 for (auto name : sizeNames) {
	    configOut << name << " = " << params.getInt(name) << endl;
	 }
	 for (auto name : boolNames) {
	    configOut << name << " = " << params.getInt(name) << endl;
	 } 
	 configOut.close();
      } else {
	 cerr << "Failed to open the output config file: " << params.getStr("output") + ".config" << endl;
	 exit(1);
      }
   }
}

int main(int argc, char* argv[]) {
   Params params;
   parseParams(argc, argv, params);

   enum PlanningAlg {qlearning, unselective, state, target, monteCarlo};

   string planner = params.getStr("planner");
   PlanningAlg alg;
   NNModel::TrainingType trainType;
   if (planner == "Q") {
      alg = qlearning;
   } else if (planner == "P" or
	      planner == "A" or
	      planner == "IE" or
	      planner == "E") {
      alg = unselective;
      trainType = NNModel::mse;
   } else if (planner == "ISV" or
	      planner == "SV" or
	      planner == "IRV" or
	      planner == "RV" or
	      planner == "ISRV" or
	      planner == "SRV") {
      alg = state;
      trainType = NNModel::gaussian;
   } else if (planner == "ISR" or
	      planner == "SR" or
	      planner == "IRR" or
	      planner == "RR" or
	      planner == "ISRR" or
	      planner == "SRR") {
      alg = state;
      trainType = NNModel::bound;
   } else if (planner == "ITR" or
	      planner == "TR" or
	      planner == "ITDR" or
	      planner == "TDR" or
	      planner == "ITOR" or
	      planner == "TOR" or
	      planner == "ITDOR" or
	      planner == "TDOR") {
      alg = target;
      trainType = NNModel::bound;
   } else if (planner == "IS" or
	      planner == "S" or
	      planner == "IMCTV" or
	      planner == "MCTV" or
	      planner == "IMCTR" or
	      planner == "MCTR" or
	      planner == "IMCTDR" or
	      planner == "MCTDR" or
	      planner == "IMCTOR" or
	      planner == "MCTOR" or
	      planner == "IMCTDOR" or
	      planner == "MCTDOR") {
      alg = monteCarlo;
      if (params.getInt("use_gaussian")) {
	 trainType = NNModel::gaussian;
      } else {
	 trainType = NNModel::iqn;
      }
   } else {
      cerr << "Planner " << planner << " not recognized.";
      exit(1);
   }
   
   streambuf* coutbuf = cout.rdbuf();
   ofstream outFile(params.getStr("output") + ".result");
   if (!outFile.is_open()) {
      cerr << "Failed to open the output file: " << params.getStr("output") + ".result" << endl;
      exit(1);
   }
   cout.rdbuf(outFile.rdbuf());


   RNG initRNG(params.getInt("seed"));
   RNG rng(initRNG.randomInt());
   torch::manual_seed(initRNG.randomInt());

   PredictionModel* env;
   BBIPredictionModel* uncertainEnv = nullptr;   
   QFunction* qFunc;
   size_t stateDim;
   act_t numActions;

   vector<Bound> dimRanges;
   string game = params.getStr("game");
   if(game == "MC") {
      env = new MountainCar();
      stateDim = 2;
      numActions = 3;
      dimRanges = {{-1.2, 0.6}, {-0.07, 0.07}};
      qFunc = new TileCodingQFunction(dimRanges, // feature ranges
				      {8, 8}, // numDivisions
				      8, // num tilings
				      numActions,
				      initRNG);
   } else if(game == "A") {
      env = new Acrobot(false, initRNG);
      stateDim = 4;
      numActions = 3;
      dimRanges = {{-M_PI, M_PI}, {-M_PI, M_PI}, {-4*M_PI, 4*M_PI}, {-9*M_PI, 9*M_PI}};
      vector<size_t> numDivisions({6, 7, 6, 7});

      vector<QFunction*> qFuncs;      
      // all 4 dimensions
      qFuncs.push_back(new TileCodingQFunction(dimRanges, numDivisions, 12, numActions, rng));
      // choose 3 dimensions / exclude 1 dimension
      for (size_t i = 0; i < stateDim; ++i) {
         vector<size_t> curNumDivisions = numDivisions;
         curNumDivisions[i] = 1;
         qFuncs.push_back(new TileCodingQFunction(dimRanges, curNumDivisions, 3, numActions, rng));
      }      
      // choose 2 dimensions / exclude 2 dimension
      for (size_t i = 0; i < stateDim; ++i) {
         for (size_t j = i+1; j < stateDim; ++j) {
            vector<size_t> curNumDivisions = numDivisions;
            curNumDivisions[i] = 1;
            curNumDivisions[j] = 1;
            qFuncs.push_back(new TileCodingQFunction(dimRanges, curNumDivisions, 2, numActions, rng));
	 } 
      }      
      // choose 1 dimension 
      for (size_t i = 0; i < stateDim; ++i) {
         vector<size_t> curNumDivisions(stateDim, 1);
         curNumDivisions[i] = numDivisions[i];
         qFuncs.push_back(new TileCodingQFunction(dimRanges, curNumDivisions, 3, numActions, rng));
      }      
      qFunc = new SumQ(qFuncs, numActions);
   } else if (game == "AD") {
      env = new Acrobot(true, initRNG);
      stateDim = 5;
      numActions = 3;
      dimRanges = {{-M_PI, M_PI}, {-M_PI, M_PI}, {-4*M_PI, 4*M_PI}, {-9*M_PI, 9*M_PI}, {-9*M_PI, 9*M_PI}};
      vector<size_t> numDivisions({6, 7, 6, 7, 7});

      vector<QFunction*> qFuncs;         
      // all 5 dimensions
      qFuncs.push_back(new TileCodingQFunction(dimRanges, numDivisions, 20, numActions, rng));
      // choose 4 dimensions / exclude 1 dimension
      for (size_t i = 0; i < stateDim; ++i) {
         vector<size_t> curNumDivisions = numDivisions;
         curNumDivisions[i] = 1;
         qFuncs.push_back(new TileCodingQFunction(dimRanges, curNumDivisions, 4, numActions, rng));
      }      
      // choose 3 dimensions / exclude 2 dimension
      for (size_t i = 0; i < stateDim; ++i) {
         for (size_t j = i+1; j < stateDim; ++j) {
            vector<size_t> curNumDivisions = numDivisions;
            curNumDivisions[i] = 1;
            curNumDivisions[j] = 1;
            qFuncs.push_back(new TileCodingQFunction(dimRanges, curNumDivisions, 2, numActions, rng));
	 } 
      }      
      // choose 2 dimensions / exclude 3 dimension
      for (size_t i = 0; i < stateDim; ++i) {
         for (size_t j = i+1; j < stateDim; ++j) {
            vector<size_t> curNumDivisions(stateDim, 1);
            curNumDivisions[i] = numDivisions[i];
            curNumDivisions[j] = numDivisions[j];
            qFuncs.push_back(new TileCodingQFunction(dimRanges, curNumDivisions, 2, numActions, rng));
	 } 
      }      
      // choose 1 dimension / exclude 4 dimension
      for (size_t i = 0; i < stateDim; ++i) {
         vector<size_t> curNumDivisions(stateDim, 1);
         curNumDivisions[i] = numDivisions[i];
         qFuncs.push_back(new TileCodingQFunction(dimRanges, curNumDivisions, 4, numActions, rng));
      }      
      qFunc = new SumQ(qFuncs, numActions);
   } else {// game == GR
      size_t length = params.getInt("gor_length");
      size_t numInd = params.getInt("gor_num_ind");
      env = new GoRight(params);
      uncertainEnv = new GoRightUncertain(initRNG, params);

      if (params.getStr("planner") == "A") {
	 stateDim = 3 + numInd;
      } else {
	 stateDim = 2 + numInd;
      }
      numActions = 2;
      
      dimRanges.push_back({rlfloat_t(-0.5), rlfloat_t(length + 0.5)});
      for (size_t i = 0; i < numInd; ++i) {
	 dimRanges.push_back({rlfloat_t(-0.25), rlfloat_t(1.25)});
      }
      size_t maxStat = 2;
      rlfloat_t statScale = length/maxStat;
      dimRanges.push_back({rlfloat_t(-0.5*statScale), rlfloat_t(statScale*(maxStat + 0.5))});

      vector<size_t> numDiv;
      numDiv.push_back(length + 1);
      for (size_t i = 0; i < numInd; ++i) {
	 numDiv.push_back(2);
      }
      numDiv.push_back(maxStat + 1);
      
      qFunc = new TileCodingQFunction(dimRanges,
					numDiv,
					1,
					numActions,
					initRNG);

      // Tile coding ignores the last dim no matter what
      // But the model might use it
      if (stateDim == 3 + numInd) {
	 dimRanges.push_back({rlfloat_t(-0.5*statScale), rlfloat_t(statScale*(maxStat + 0.5))});
      }
   }

   uniform_int_distribution<act_t> actDist(0, numActions-1);

   QLearner* agent = new QLearner(qFunc,
				  numActions,
				  initRNG,
				  params);
   
   LearnedModel* model;
   if (params.getInt("use_nn")) {
      model = new NNModel(stateDim,    // inDim
			  stateDim,    // targetDim
			  numActions,
			  dimRanges,
			  initRNG,
			  trainType,
			  params);
   } else {
      model = new IncDTModel(stateDim,
			     numActions,
			     initRNG,
			     params);
   }
   
   BBIPredictionModel* planningModel;
   if (planner[0] == 'I') {
      planningModel = uncertainEnv;
   } else {
      planningModel = dynamic_cast<BBIPredictionModel*>(model);
   }
		  
   bool modelUpdated = false;
      
   vector<Trajectory*> data;   
   rlfloat_t epReward = 0;
   rlfloat_t epReturn = 0;
   rlfloat_t totalDiscount = 0;

   double totalTime = 0;
   double totalPlanTime = 0;
   size_t totalFrames = 0;
   size_t framesSinceSplit = 0;
//   size_t ep = 0;

   size_t horizon = params.getInt("horizon");

   size_t padW = 15;
   size_t col = 1;
   cout << setw(padW) << to_string(col) + "_totFrames";
   ++col;
   cout << setw(padW) << to_string(col) + "_epFPS";
   ++col;
   cout << setw(padW) << to_string(col) + "_totalFPS";
   ++col;
   cout << setw(padW) << to_string(col) + "_epPlanFPS";
   ++col;
   cout << setw(padW) << to_string(col) + "_totalPlanFPS";
   ++col;
   cout << setw(padW) << to_string(col) + "_epScore";
   ++col;
   cout << setw(padW) << to_string(col) + "_epReturn";
   ++col;
   cout << setw(padW) << to_string(col) + "_epFrames";
   ++col;
   cout << setw(padW) << to_string(col) + "_evalScore";
   ++col;
   cout << setw(padW) << to_string(col) + "_evalReturn";
   ++col;
   cout << setw(padW) << to_string(col) + "_evalFrames";
   ++col;
   cout << setw(padW) << to_string(col) + "_effHoriz";
   ++col;
   
   string errNames[] {"StateErr", "RwdErr", "TermErr", "PredErr", "TargErr", "UncErr", "NumInf", "Num-Inf", "uErrMin", "uErrLQ", "uErrMed", "uErrUQ", "uErrMax", "utCorr"};
   for (auto name : errNames) {
      for (size_t h = 2; h <= horizon; ++h) {
	 cout << setw(padW) << to_string(col) + "_" + name + "_h" + to_string(h);
	 ++col;
      }
      cout << setw(padW) << to_string(col) + "_" + name;
      ++col;
   }
   
   cout << endl;
   
   while (totalFrames < size_t(params.getInt("num_frames"))) {
      vector<double> uncSum(horizon-1, 0);
      vector<double> tgtSum(horizon-1, 0);
      vector<double> uncXtgt(horizon-1, 0);
      vector<double> uncXunc(horizon-1, 0);
      vector<double> tgtXtgt(horizon-1, 0);
      vector<size_t> nonInf(horizon-1, 0);

      vector<rlfloat_t> stateError(horizon-1, 0);
      vector<rlfloat_t> rwdError(horizon-1, 0);
      vector<rlfloat_t> termError(horizon-1, 0);
      vector<rlfloat_t> predError(horizon-1, 0);
      vector<rlfloat_t> targetError(horizon-1, 0);
      vector<vector<rlfloat_t> > uncertaintyErrors(horizon-1);
      vector<rlfloat_t> uncertaintyError(horizon-1, 0);
      vector<size_t> numInf(horizon-1, 0);
      vector<size_t> numNegInf(horizon-1, 0);

      size_t learnFrames = 0;

      vector<vector<rlfloat_t>*> errs({&stateError, &rwdError, &termError, &predError});
      vector<vector<size_t>*> infCounts({&numInf, &numNegInf});
      
      
      double effectiveHorizon = 0;
      
      for (unsigned char eval = 0; eval < 2; ++eval) {
	 size_t numFrames = 0;
	 epReward = 0;
	 epReturn = 0;
	 totalDiscount = 1;

	 State curState;
	 if (game == "MC") {
	    curState = {0,0};
	 } else if (game == "A") {
	    curState = {0,0,0,0};
	 } else if (game == "AD") {
	    curState = {0,0,0,0,0};
	 } else if (game == "GR") {
	    curState.push_back(rng.randomFloat()*0.5 - 0.25);

	    size_t length = params.getInt("gor_length");
	    size_t numInd = params.getInt("gor_num_ind");
	    for (size_t i = 0; i < numInd; ++i) {
	       curState.push_back(0);
	    }

	    size_t numStat = 3;	    
	    rlfloat_t statScale = double(length)/(numStat - 1);
	    rlfloat_t prevStat = int(rng.randomFloat()*numStat)*statScale;
	    rlfloat_t initStat = int(rng.randomFloat()*numStat)*statScale;
	    rlfloat_t statOffset = rng.randomFloat()*statScale/2 - statScale/4;
	    curState.push_back(statOffset + initStat);
	    curState.push_back(statOffset + prevStat);
	 }
	 
	 bool terminated = false;

	 if (!eval) {
	    data.push_back(new Trajectory(curState, terminated));
	 }

	 auto epStart = chrono::high_resolution_clock::now();
	 double epPlanTime = 0;
	 
	 for (size_t t = 0; t < 500 and !terminated; ++t) {
	    act_t action;
	    Trajectory& curTraj = *data.back();

            if (!eval) {
	       if (rng.randomFloat() < params.getFloat("exploration_rate")) {
		  action = rng.randomFloat()*numActions;
	       } else {
		  DOUT << "greedy (learned Q-function)" << endl;
		  action = agent->getGreedyAction(curState);
	       }
	    } else {
	       action = agent->getGreedyAction(curState);
	    }

	    State resultState;

	    if (eval) {
	       DOUT << "Eval ";
	    }
	    DOUT << "Frame: " << totalFrames << " Episode Step " << t << ": ";
	    for (auto d : curState) {
	       DOUT << d << " ";
	    }
	    DOUT << " action: " << action << endl;

	    env->getStatePrediction(curState, action, resultState);
         
	    rlfloat_t r = env->getRewardPrediction(curState, action);
	    epReward += r;
	    epReturn += totalDiscount*r;
	    totalDiscount *= params.getFloat("discount");

	    terminated = env->getTermPrediction(curState, action) > 0.5;

	    if (!eval) {
	       curTraj.addStep(action, r, resultState, terminated);

	       QLearner::Measurements measurements;

	       auto planStart = chrono::high_resolution_clock::now();
	       
	       if (planner == "P") {
		  agent->mveUpdate(curTraj, t, env, measurements);
	       } else {
		  if (alg == qlearning or
		      (planningModel != uncertainEnv and !modelUpdated) or
		      horizon == 1) {
		     agent->qUpdate(curTraj, t);
		  } else if (alg == unselective) {
		     agent->mveUpdate(curTraj, t, planningModel, env, measurements);
		  } else if (alg == target) {
		     agent->targetRangeSMVEUpdate(curTraj, t, planningModel, env, uncertainEnv, measurements);
		  } else if (alg == state) {
		     agent->oneStepUncertaintySMVEUpdate(curTraj, t, planningModel, env, uncertainEnv, measurements);
		  } else { // (alg == monteCarlo) {
		     agent->monteCarloSMVEUpdate(curTraj, t, planningModel, env, uncertainEnv, measurements);
		  } 
		  
		  model->addExample(curTraj, t);
	       }

	       auto planEnd = chrono::high_resolution_clock::now();
	       auto planTime = chrono::duration_cast<chrono::duration<double>>(planEnd - planStart).count();
	       epPlanTime += planTime;
	       totalPlanTime += planTime;
	       
	       double totalWeight;
	       if (measurements.weights.size() > 0) {
		  totalWeight = measurements.weights[0];
	       } else {
		  totalWeight = 1;
	       }
	       double weightedHorizon = totalWeight;
	       for (size_t h = 1; h < measurements.stateError.size(); ++h) {
		  if (measurements.uncertainties[h] != numeric_limits<double>::infinity()) {
		     uncSum[h-1] += measurements.uncertainties[h];
		     tgtSum[h-1] += fabs(measurements.targetError[h]);
		     uncXtgt[h-1] += measurements.uncertainties[h]*fabs(measurements.targetError[h]);
		     uncXunc[h-1] += measurements.uncertainties[h]*measurements.uncertainties[h];
		     tgtXtgt[h-1] += measurements.targetError[h]*fabs(measurements.targetError[h]);
		     ++nonInf[h-1];
		  }
		  
		  for (auto d : measurements.stateError[h]) {
		     stateError[h-1] += (d*d)/stateDim;
		     predError[h-1] += (d*d)/(stateDim+2);
		  }
		  rlfloat_t sqRErr = measurements.rwdError[h]*measurements.rwdError[h];
		  rwdError[h-1] += sqRErr;
		  predError[h-1] += sqRErr/(stateDim+2);
		  rlfloat_t sqTErr = measurements.termError[h]*measurements.termError[h];
		  termError[h-1] += sqTErr;
		  predError[h-1] += sqTErr/(stateDim+2);
		  targetError[h-1] += fabs(measurements.targetError[h]);
		  if (uncertainEnv) { // If we have an oracle
		     uncertaintyErrors[h-1].push_back(measurements.uncertaintyError[h]);
		     if (measurements.uncertaintyError[h] == numeric_limits<double>::infinity()) {
			++numInf[h-1];
		     } else if (measurements.uncertaintyError[h] == -numeric_limits<double>::infinity()) {
			++numNegInf[h-1];
		     } else {
			uncertaintyError[h-1] += fabs(measurements.uncertaintyError[h]);
		     }
		  } else {
		     DOUT << "Pushing back 0" << endl;
		     uncertaintyErrors[h-1].push_back(0);
		  }
		  totalWeight += measurements.weights[h];
		  weightedHorizon += measurements.weights[h]*(h+1);
	       }

	       effectiveHorizon += weightedHorizon/totalWeight;
	       DOUT << "Effective horizon: " << weightedHorizon/totalWeight << endl;
	       
	       ++totalFrames;
	       ++framesSinceSplit;

	       if (planner != "P" and
		   planner != "Q" and
		   planningModel != uncertainEnv and
		   horizon > 1 and
		   framesSinceSplit >= size_t(params.getInt("update_every"))) {
		  DOUT << "Updating Model " << totalFrames << endl;
		  model->updatePredictions();
		  framesSinceSplit = 0;
		  modelUpdated = true;
	       }
	    }
	    curState = resultState;
	    ++numFrames;
	    if (!eval) {
	       ++learnFrames;
	    }
	 }

	 auto epEnd = chrono::high_resolution_clock::now();
	 auto epTime = chrono::duration_cast<chrono::duration<double>>(epEnd - epStart).count();
	 totalTime += epTime;
	 
	 if (!eval) {
	    cout << setw(padW) << totalFrames;
	    cout << setw(padW) << double(numFrames)/epTime;
	    cout << setw(padW) << double(totalFrames)/totalTime;
	    cout << setw(padW) << double(numFrames)/epPlanTime;
	    cout << setw(padW) << double(totalFrames)/totalPlanTime;
	 }
	 cout << setw(padW) << epReward;
	 cout << setw(padW) << epReturn;
	 cout << setw(padW) << numFrames;
	 
	 if (eval) {
	    cout << setw(padW) << effectiveHorizon/learnFrames;

	    rlfloat_t total = 0;

	    for (size_t e = 0; e < errs.size(); ++e) { // stateErr, rwdErr, termErr
	       total = 0;
	       for (size_t h = 0; h < horizon-1; ++h) {		  
		  cout << setw(padW) << sqrt((*errs[e])[h]/learnFrames);
		  total += (*errs[e])[h]/learnFrames;
	       }
	       if (horizon > 1) {
		  cout << setw(padW) << sqrt(total/(horizon-1));
	       } else {
		  cout << setw(padW) << 0;
	       }
	    }

	    total = 0;
	    for (size_t h = 0; h < horizon-1; ++h) {
	       cout << setw(padW) << targetError[h]/learnFrames;
	       total += targetError[h]/learnFrames;
	    }
	    if (horizon > 1) {
	       cout << setw(padW) << sqrt(total/(horizon-1));
	    } else {
	       cout << setw(padW) << 0;
	    }

	    total = 0;
	    for (size_t h = 0; h < horizon-1; ++h) {
	       cout << setw(padW) << uncertaintyError[h]/(learnFrames - numInf[h] - numNegInf[h]);
	       total += uncertaintyError[h]/(learnFrames - numInf[h] - numNegInf[h]);
	    }
	    if (horizon > 1) {
	       cout << setw(padW) << sqrt(total/(horizon-1));
	    } else {
	       cout << setw(padW) << 0;
	    }
	    
	    for (size_t c = 0; c < infCounts.size(); ++c) {
	       total = 0;
	       for (size_t h = 0; h < horizon-1; ++h) {
		  cout << setw(padW) << (*infCounts[c])[h];
		  total += (*infCounts[c])[h];
	       }
	       cout << setw(padW) << total;
	    }

	    vector<rlfloat_t> quantiles({0, 0.25, 0.5, 0.75, 1});
	    for (auto q : quantiles) {
	       vector<rlfloat_t> allErrs;
	       for (size_t h = 0; h < horizon-1; ++h) {
		  if (uncertaintyErrors[h].size() > 0) {
		     sort(uncertaintyErrors[h].begin(), uncertaintyErrors[h].end());
		     size_t idx = (uncertaintyErrors[h].size()-1)*q;		  
		     cout << setw(padW) << uncertaintyErrors[h][idx];
		     allErrs.insert(allErrs.end(), uncertaintyErrors[h].begin(), uncertaintyErrors[h].end());
		  } else {
		     cout << setw(padW) << 0;
		  }
	       }

	       if (allErrs.size() > 0) { 
		  sort(allErrs.begin(), allErrs.end());
		  size_t idx = (allErrs.size()-1)*q;
		  cout << setw(padW) << allErrs[idx];
	       } else {
		  cout << setw(padW) << 0;
	       }
	    }
	    
	    rlfloat_t totalUncSum = 0;
	    rlfloat_t totalTgtSum = 0;
	    rlfloat_t totalUncXTgt = 0;
	    rlfloat_t totalUncXUnc = 0;
	    rlfloat_t totalTgtXTgt = 0;
	    size_t totalNonInf = 0;
	    for (size_t h = 0; h < horizon-1; ++h) {
	       rlfloat_t corrNum = nonInf[h]*uncXtgt[h] - uncSum[h]*tgtSum[h];
	       rlfloat_t corrDenUnc = nonInf[h]*uncXunc[h] - uncSum[h]*uncSum[h];
	       if (corrDenUnc < 0) { // Can only happen because of floating point error
		  corrDenUnc = 0;
	       }
	       rlfloat_t corrDenTgt = nonInf[h]*tgtXtgt[h] - tgtSum[h]*tgtSum[h];
	       if (corrDenTgt < 0) { // Can only happen because of floating point error
		  corrDenTgt = 0;
	       }
	       rlfloat_t corrDen = sqrt(corrDenUnc) * sqrt(corrDenTgt);
	       DOUT << "nonInf[h] " << nonInf[h] << " uncXunc[h] " << uncXunc[h] << " uncSum[h] " << uncSum[h] << " tgtXtgt[h] " << tgtXtgt[h] << " tgtSum[h] " << tgtSum[h] << endl;
	       DOUT << "sqrt1: " << nonInf[h]*uncXunc[h] - uncSum[h]*uncSum[h] << " sqrt2: " << nonInf[h]*tgtXtgt[h] - tgtSum[h]*tgtSum[h] << endl;
	       DOUT << "corrNum: " << corrNum << " corrDen: " << corrDen << " corr: " << corrNum/corrDen << endl;
	       
	       if (corrDen != 0) {
		  cout << setw(padW) << corrNum/corrDen;
	       } else {
		  cout << setw(padW) << 0;
	       }

	       totalUncSum += uncSum[h];
	       totalTgtSum += tgtSum[h];
	       totalUncXTgt += uncXtgt[h];
	       totalUncXUnc += uncXunc[h];
	       totalTgtXTgt += tgtXtgt[h];
	       totalNonInf += nonInf[h];
	    }
	    rlfloat_t corrNum = totalNonInf*totalUncXTgt - totalUncSum*totalTgtSum;
	    rlfloat_t corrDenUnc = totalNonInf*totalUncXUnc - totalUncSum*totalUncSum;
	    if (corrDenUnc < 0) { // Can only happen because of floating point error
	       corrDenUnc = 0;
	    }
	    rlfloat_t corrDenTgt = totalNonInf*totalTgtXTgt - totalTgtSum*totalTgtSum;
	    if (corrDenTgt < 0) { // Can only happen because of floating point error
	       corrDenTgt = 0;
	    }
	    rlfloat_t corrDen = sqrt(corrDenUnc) * sqrt(corrDenTgt);

	    DOUT << "nonInf " << totalNonInf << " uncXunc " << totalUncXUnc << " uncSum " << totalUncSum << " tgtXtgt " << totalTgtXTgt << " tgtSum " << totalTgtSum << endl;
	    DOUT << "sqrt1: " << totalNonInf*totalUncXUnc - totalUncSum*totalUncSum << " sqrt2: " << totalNonInf*totalTgtXTgt - totalTgtSum*totalTgtSum << endl;
	    DOUT << "corrNum: " << corrNum << " corrDen: " << corrDen << " corr: " << corrNum/corrDen << endl;
	    
	    if (corrDen != 0) {
	       cout << setw(padW) << corrNum/corrDen;
	    } else {
	       cout << setw(padW) << 0;
	    }
	 }
      }
      cout << endl;
   }
   delete env;
   delete uncertainEnv;
   delete agent;
   for (auto traj : data) {
      delete traj;
   }
   delete model;


   std::cout.rdbuf(coutbuf);
   outFile.close();
}
