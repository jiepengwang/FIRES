#include "ScaleICP.h"

#include <boost/filesystem.hpp>
#include <fstream>
#include <algorithm>
#include <math.h>

#include "CloudUtils.h"
#include "PointsReg.h"
#include "PointsUtils.h"
#include "LogUtils.h"

ScaleICP::ScaleICP(/* args */)
{
}

ScaleICP::~ScaleICP()
{
}




std::tuple<Eigen::MatrixXf, float> ScaleICP::updateSRTPoint2Plane(const Eigen::MatrixXf& mat_source_point,
                                                                 const Eigen::MatrixXf& mat_target_point,
                                                                 const Eigen::MatrixXf& mat_target_normal,
                                                                 const bool& b_scaling,
                                                                 const Eigen::Vector3f& interval_scaling)
{
    
    const int num_rows = mat_source_point.rows(); // For ICP of data with num-rows dimension
	const int num_cols = mat_source_point.cols();

    Eigen::MatrixXf A(num_cols, 6);
    Eigen::VectorXf B(num_cols);

    // Calculate R, T
    for (int idx = 0; idx < num_cols; idx++) {
        const Eigen::Vector3f& pt_a = mat_source_point.col(idx);
        const Eigen::Vector3f& pt_b = mat_target_point.col(idx);
        const Eigen::Vector3f& pt_b_normal = mat_target_normal.col(idx);

        double dx = pt_b(0);
        double dy = pt_b(1);
        double dz = pt_b(2);

        double nx = pt_b_normal(0);
        double ny = pt_b_normal(1);
        double nz = pt_b_normal(2);

        // template point
        double sx = pt_a(0);
        double sy = pt_a(1);
        double sz = pt_a(2);

        if (std::isnan(dx + dy + dz + nx + ny + nz + sx + sy + sz)) {
            std::cout << "?????";
            std::cout << idx << std::endl;
            std::cout << dx << " " << dy << " " << dz << std::endl;
            std::cout << nx << " " << ny << " " << nz << std::endl;
            std::cout << sx << " " << sy << " " << sz << std::endl;
            std::exit(0);
        }
        // setup least squares system
        A(idx, 0) = 1.0 * (nz * sy - ny * sz);
        A(idx, 1) = 1.0 * (nx * sz - nz * sx);
        A(idx, 2) = 1.0 * (ny * sx - nx * sy);
        A(idx, 3) = 1.0 * nx;
        A(idx, 4) = 1.0 * ny;
        A(idx, 5) = 1.0 * nz;
        B(idx) = 1.0 * (nx * dx + ny * dy + nz * dz - nx * sx - ny * sy - nz * sz);
    }

    Eigen::VectorXf delta_p;
    delta_p = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);
    Eigen::Matrix3f R_inc = Eigen::Matrix3f::Identity();
    R_inc(0, 1) = -delta_p(2);
    R_inc(1, 0) = +delta_p(2);
    R_inc(0, 2) = +delta_p(1);
    R_inc(2, 0) = -delta_p(1);
    R_inc(1, 2) = -delta_p(0);
    R_inc(2, 1) = +delta_p(0);

    Eigen::Vector3f T_inc(delta_p(3), delta_p(4), delta_p(5));
    Eigen::Matrix3f U;
    Eigen::Matrix3f V;
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(R_inc, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U = svd.matrixU();
    V = svd.matrixV();

    R_inc = U * V.transpose();

    if (R_inc.determinant() < 0) {
        Eigen::Matrix3f tmp = Eigen::Matrix3f::Identity();
        tmp(2, 2) = R_inc.determinant();
        R_inc = V * tmp * U.transpose();
    }

    // Calculate scale
    float scale = 1.0;
    Eigen::Matrix4f trans_svd = Eigen::Matrix4f::Identity();
    if (b_scaling) {
        float a = 0.0;
        float b = 0.0;
        for (int i = 0; i < num_cols; i++) {
            Eigen::Vector3f source = R_inc * mat_source_point.col(i) + T_inc;
            Eigen::Vector3f target = mat_target_point.col(i);
            Eigen::Vector3f normal = mat_target_normal.col(i);
            float current_a = source.dot(normal);
            float current_b = target.dot(normal);
            a += current_a * current_a;
            b -= 2 * current_a * current_b;
        }
        scale = -0.5 * b / a;
        trans_svd.block<3,3>(0,0) = scale * R_inc;
        trans_svd.block<3,1>(0,3) = scale * T_inc;
    }
    else{
        trans_svd.block<3,3>(0,0) = R_inc;
        trans_svd.block<3,1>(0,3) = T_inc;
    }

    return { trans_svd, scale };
}