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
            const int nobs = prev_layer_data.cols();
            m_loc.resize(this->m_out_size, nobs);
            m_z.resize(this->m_out_size, nobs);

            int* loc_data = m_loc.data();
            const int channel_end = prev_layer_data.size();
            const int channel_stride = m_channel_rows * m_channel_cols;
            const int col_end_gap = m_channel_rows * m_pool_cols * m_ou_cols;
            const int col_stride = m_channel_rows * m_pool_cols;
            const int row_end_gap = m_out_rows * m_pool_rows;

            for(int channel_start = 0; channel_start < channel_end; channel_start += channel_stride){
                const int col_end = channel_start + col_end_gap;

                for(int col_start = channel_start; col_start < col_end; col_start += col_stride){
                    const int row_end = col_start + row_end_gap;

                    for(int row_start = col_start; row_start < row_end; row_start += m_pool_rows, loc_data++){
                        *loc_data = row_start;
                    }
                }
            }

            loc_data = m_loc.data();
            const int* const loc_end = loc_data + m_loc.size();
            Scalar* z_data = m_z.data();

            for(; loc_data < loc_end; loc_data++, z_data++){
                const int offset = *loc_data;
                *z_data = internal::find_block_max(src + offset, m_pool_rows, m_pool_cols, m_channel_rows, *loc_data);
                *loc_data += offset;
            }

            m_a.resize(this->m_out_size, nobs);
            Activation::activate(m_z, m_a);
        }

        const Matrix& output() const{
            return m_a;
        }

        void backprop(const matrix& prev_layer_data, const Matrix& next_layer_data){
            const int nobs = prev_layer_data.cols();
            Matrix& dLz = m_z;
            Activation::apply_jacobian(m_z, m_a, next_layer_data, dLz);

            m_din.resize(this->m_in_size, nobs);
            m_din.setZero();
            const int dLz_size = dLz.size();
            const Scalar* dLz_data = dLz.data();
            const int* loc_data = m_loc.data();
            Scalar* din_data = m_din.data();

            for(int i=0; i <dLz_size; i++){
                din_data[loc_data[i]] += dLz_data[i];
            }
        }

        const Matrix& backprop_data() const{
            return m_din;
        }

        const update(Optimizer& opt){}

};