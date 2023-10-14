#pragma once

#include "Eigen3/Eigen/Core"
#include "Config.h"
#include "NeuralNet.h"

class Callback{
    protected:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::RowVectorXi IntegerVector;

    public:
        int m_nbatch;
        int m_batch_id;
        int m_nepoch;
        int m_epoch_id;

        Callback() : m_nbatch(0), m_nepoch(0), m_epoch_id(0){}

        virtual ~Callback() {}

        virtual void pre_training_batch(const Network* net, const Matrix& x, const Matrix& y){}
        virtual void pre_training_batch(const Network* net, const Matrix& x, const IntegerVector& y){}

        virtual void post_training_batch(const Network* net, const Matrix& x, Matrix& y){}
        virtual void post_training_batch(const Network* net, const Matrix& x, const IntegerVector& y){}

};