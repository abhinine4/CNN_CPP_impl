#pragma once

#include "Eigen3/Eigen/Core"
#include <stdexcept>
#include "Config.h"

class Output{
    protected:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::RowVectorXi IntegerVector;

    public:
        virtual ~Output(){}
        virtual void check_target_data(const Matrix& target){}
        virtual void check_target_data(const IntegerVector& target){
            throw std::invalid_argument("[class output]: This output type cannot take class labels as target data");

        }

        virtual void evaluate(const Matrix& prev_layer_data, const Matrix& target) = 0;
        virtual void evaluate(const Matrix& prev_layer_data, const IntegerVector& target){
            throw std::invalid_argument("[class output]: This output type cannot take class labels as target data");
        }

        virtual const Matrix& backprop_data() const = 0;
        virtual Scalar loss() const = 0;

        virtual std::string output_type() const = 0;

};