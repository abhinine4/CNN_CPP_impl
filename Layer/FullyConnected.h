#pragma once

#include "../Eigen3/Eigen/Core"
#include <vector>
#include <stdexcept>
#include "../Config.h"
#include "../Layer.h"
#include "../Utils/Random.h"

template<typename Activation>
class FullyConnected :public Layer {
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
        typedef Vector::AlignedMapType AlignedMapVec;

        Matrix m_weight;
        Vector m_bias;
        Matrix m_dw;
        Vector m_db;
        Matrix m_z;
        Matrix m_a;
        Matrix m_din;

    public:
        FullyConnected(const int in_size, const int out_size):
            Layer(in_size, out_size){}

        void init(const Scalar& mu, const Scalar& sigma, RNG& rng){

            m_weight.resize(this->m_in_size, this->m_out_size);
            m_bias.resize(this->m_out_size);
            m_dw.resize(this->m_in_size, this->m_out_size);
            m_db.resize(this->m_out_size);

            internal::set_normal_random(m_weight.data(), m_weight.size(), rng, mu, sigma);
            internal::set_normal_random(m_bias.data(), m_bias.size(), rng, mu, sigma);
        }

    void forward(const Matrix& prev_layer_data){
        const int nobs = prev_layer_data.cols();

        // z = w' * in + b
        m_z.resize(this->m_out_size, nobs);
        m_z.noalias() = m_weight.transpose() * prev_layer_data;
        m_z.colwise() += m_bias;

        m_a.resize(this->m_out_size, nobs);
        Activation::activate(m_z, m_a);
    }

    const Matrix& output() const{
        return m_a;
    }

    void backprop(const Matrix& prev_layer_data, const Matrix& next_layer_data){
        const int nobs = prev_layer_data.cols();
        Matrix& dLz = m_z;
        Activation::apply_jacobian(m_z, m_a, next_layer_data, dLz);
        m_dw.noalias() = prev_layer_data * dLz.transpose() / nobs;
        m_dw.noalias() = prev_layer_data * dLz.transpose() / nobs;

    }

    const Matrix& backprop_data() const{
        return m_din;
    }

    void update(Optimizer& opt){
        ConstAlignedMapVec dw(m_dw.data(), m_dw.size());
        ConstAlignedMapVec db(m_db.data(), m_db.size());
        AlignedMapVec w(m_weight.data(), m_weight.size());
        AlignedMapVec b(m_bias.data(), m_bias.size());

        opt.update(dw, w);
        opt.update(db, b);
    }

    virtual std::vector<Scalar> get_parameter() const {};
    virtual void set_parameter(const std::vector<Scalar>& param) {};
    virtual std::vector<Scalar> get_derivatives() const {};
};  
