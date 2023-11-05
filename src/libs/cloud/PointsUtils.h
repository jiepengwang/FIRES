#pragma once
#include <Eigen/Core>
#include <vector>

/*
    Data type:
    1. std::vector
    2. Eigen
    3. No PCL
*/

typedef std::vector<Eigen::Vector3f> Points3Df;
typedef std::vector<Eigen::Vector3f> Points3D;

namespace PointsUtils
{
    Eigen::MatrixXf convertPointsToMat(const Points3Df& points);
    std::tuple<Eigen::Matrix3f, Eigen::Vector3f> computeCovMatNormalized(const Points3Df& points);
    std::tuple<Eigen::Matrix4f, Eigen::Vector3f> computeEigenVectorsAndValues(const Eigen::Matrix3f& mat_cov, const Eigen::Vector3f& centroid); 
    std::tuple<Eigen::Matrix4f, Eigen::Vector3f> computeEigenVectorsAndValues(const Points3Df& points);


    
    // Add ScaleICP funcs
    std::tuple<Eigen::Matrix4f, Eigen::Vector3f>  estimateInitialTransformationByPCA(const Points3D& point_source, const Points3D& points_target);

} // namespace PointsUtils