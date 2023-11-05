#pragma once

#include <string>
#include <vector>

#include "CloudUtils.h"
#include "PointsReg.h"

class ScaleICP : public PointsReg
{
private:
    /* data */
public:
    ScaleICP(/* args */);
    ~ScaleICP();

    std::tuple<Eigen::MatrixXf, float> updateSRTPoint2Plane(const Eigen::MatrixXf& mat_source_point,
                                                        const Eigen::MatrixXf& mat_target_point,
                                                        const Eigen::MatrixXf& mat_target_normal,
                                                        const bool& b_scaling,
                                                        const Eigen::Vector3f& interval_scaling = Eigen::Vector3f(0.9, 1.1, 1.0));
};