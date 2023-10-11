#pragma once 
#include "../Layer.h"
#include <vector>
#include <stdexcept>
#include "../Layer.h"
#include "../Eigen3/Eigen/Core"

template<typename Activation>
class Pooling :public Layer{
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, eigen::Dynamic> Matrix;
        typedef Eigen::MatrixXi IntMatrix;
        
        const int m_channel_rows;
        const int m_channel_cols;
        const int m_in_channels;
        const int m_pool_rows;
        const int m_pool_cols;

        const int m_out_rows;
        const int m_out_cols;

        IntMatrix m_loc;
        Matrix m_z;
        Matrix m_a;
        Matrix m_din;

    public:
        Pooling(const int in_width, const int in_height, const int in_channel, const int pooling_width,const int pooling_height):
            Layer(in_width*in_height*in_channel,(in_width/pooling_width)*(in_height/pooling_height)*in_channel),
            m_channel_rows(in_height),
            m_channel_cols(in_width),
            m_in_channels(in_channel),
            m_pool_rows(pooling_height),
            m_pool_cols(pooling_width),
            m_out_rows(m_channel_rows/m_pool_rows),
            m_out_cols(m_channel_cols/m_pool_cols){}

        void init(const Scalar& mu, const Scalar& sigma, RNG& rng){}

        void forward(const Matrix& prev_layer_data){
            
        }




};