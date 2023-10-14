#pragma once

#include "../Optimizer.h"
#include "../Eigen3/Eigen/Core"
#include "../Config.h"

class SGD : public Optimizer {
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
        typedef Vector::AlignedMapType AlignedMapVec;

    public:
        Scalar m_lrate;
        Scalar m_decay;

        SGD(const Scalar& lrate = Scalar(0.001), const Scalar& decay = Scalar(0)):
            m_lrate(lrate), m_decay(decay){}

        void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec){
            vec.noalias() -= m_lrate * (dvec + m_decay * vec);
        }

};