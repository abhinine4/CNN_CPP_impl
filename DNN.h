#pragma once

#include "Eigen3/Eigen/Dense"
#include "Config.h"
#include "RNG.h"

#include "Layer.h"
#include "Layer/Convolutional.h"
#include "Layer/FullyConnected.h"
#include "Layer/Pooling.h"

#include "Output.h"
#include "Output/MSE.h"
#include "Output/CrossEntropy.h"

#include "Optimizer.h"
#include "Optimizer/SGD.h"
#include "Optimizer/Momentum.h"
#include "Optimizer/AdaGrad.h"

#include "Callback.h"
#include "Callback/VerboseCallback.h"

#include "NeuralNet.h"