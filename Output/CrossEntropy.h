#pragma once

#include "../Eigen3/Eigen/Core"
#include "../Config.h"
#include <stdexcept>
#include "../Output.h"

class CrossEntropy : public Output{
    private:
        Matrix m_din;
    
    public:
        void check_target_data(const Matrix& target){
            const int nelem = target.size();
            const Scalar* target_data = target.data();

            for(int i=0; i < nelem; i++){
                if((target_data[i] != Scalar(0)) && (target_data[i] != Scalar(1))){
                    throw std::invalid_argument("[class CrossEntropy]: Target data should pnly contain zero or one");
                }
            }
        }

        void check_target_data(const IntegerVector& target){
            const int nobs = target.size();

            for(int i=0; i < nobs; i++){
                if((target[i] != 0) && (target[i] != 1)){
                    throw std::invalid_argument("[class CrossEntropy]: Target data should pnly contain zero or one");
                }
            }
        }

        void evaluate(const Matrix& prev_layer_data, const Matrix& target){
            const int nobs = prev_layer_data.cols();
            const int nvar = prev_layer_data.rows();

            if((target.cols() != nobs) && (target.rows() != nvar)){
                    throw std::invalid_argument("[class CrossEntropy]: Target data has incprrect dimensions");
                }

            m_din.resize(nvar, nobs);
            m_din.array() = (target.array() < Scalar(0.5)).select((Scalar(1) - prev_layer_data.array()).cwiseInverse(), -prev_layer_data.cwiseInverse());
        }

        void evaluate(const Matrix& prev_layer_data, const IntegerVector& target){
            const int nvar = prev_layer_data.rows();
            if(nvar != 1){
                throw std::invalid_argument("[class CrossEntropy]: Only one response variable is allowed when calss labels are used as target data");
            }

            const int nobs = prev_layer_data.cols();

            if((target.size() != nobs)){
                throw std::invalid_argument("[class CrossEntropy]: Tsarget data has incorrect dimensions");
            }

            m_din.resize(1, nobs);
            m_din.array() = (target.array() == 0).select((Scalar(1) - prev_layer_data.array()).cwiseInverse(), -prev_layer_data.cwiseInverse());
        }

        const Matrix& backprop_data() const{
            return m_din;
        }

        Scalar loss() const{
            return m_din.array().abs().log().sum() / m_din.cols();
        }
};