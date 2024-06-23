cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(DontKnowBranches)

option(DEBUG_OUT "Turn on debug output" OFF)

set(CMAKE_CXX_FLAGS "-std=c++17 -g -O3 -pedantic -Wall -Wextra")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

find_package(cxxopts REQUIRED)
find_package(Torch REQUIRED)

add_executable(planning
  src/planning.cpp
  src/rl/TileCodingQFunction.cpp
  src/rl/Trajectory.cpp
  src/rl/PredictionModel.cpp
  src/rl/QFunction.cpp
  src/rl/QLearner.cpp
  src/rl/environments/Acrobot.cpp
  src/rl/environments/GoRight.cpp
  src/rl/environments/MountainCar.cpp
  src/rl/models/Discriminator.cpp
  src/rl/models/Example.cpp
  src/rl/models/FastIncModelTree.cpp
  src/rl/models/IncDTModel.cpp
  src/rl/models/NNModel.cpp
  src/util/Params.cpp
  src/util/RNG.cpp
)

if (DEBUG_OUT)
   target_compile_definitions(planning PRIVATE "DEBUG")
endif()

target_link_libraries(planning "${TORCH_LIBRARIES}")

target_include_directories(planning PRIVATE src/ src/rl src/rl/environments src/rl/models src/util)
