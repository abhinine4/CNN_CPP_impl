#pragma once

#include "../Eigen3/Eigen/Core"
#include <iostream>
#include "../Config.h"
#include "../Callback.h"
#include "../NeuralNet.h"

class VerboseCallback : public Callback{
    public:
        void post_training_batch(const Network* net, const Matrix& x, const Matrix& y){
            const Scalar loss = net->get_output()->loss();
            std::cout << "[Epoch " << m_epoch_id <<", batch " << m_batch_id <<",] Loss = " << loss <<std::endl;
        }

        void post_training_batch(const Network* net, const Matrix& x, const IntegerVector& y){
            const Scalar loss = net->get_output()->loss();
            std::cout << "[Epoch " << m_epoch_id <<", batch " << m_batch_id <<",] Loss = " << loss <<std::endl;
        }
};