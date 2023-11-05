#include "PointsUtils.h"
#include <cmath>
#include <iostream>

#include <Eigen/Eigenvalues> 

Eigen::MatrixXf PointsUtils::convertPointsToMat(const Points3Df& points)
{
    int num_points = points.size();
    Eigen::MatrixXf mat_points(3, num_points);  // 3*N
    for (int i = 0; i < num_points; i++){
        mat_points.col(i) = points[i];
    }


    return mat_points;
}


std::tuple<Eigen::Matrix3f,Eigen::Vector3f> PointsUtils::computeCovMatNormalized(const Points3Df& points)
{
    Eigen::MatrixXf mat_points = convertPointsToMat(points);
    Eigen::Vector3f centroid = mat_points.rowwise().mean();
    Eigen::MatrixXf mat_points_demean = mat_points.colwise() - centroid;
    Eigen::Matrix3f mat_cov = mat_points_demean*mat_points_demean.transpose();

    // Normalize
    mat_cov *= 1.0/(float)points.size();
    return {mat_cov, centroid};
}


std::tuple<Eigen::Matrix4f, Eigen::Vector3f> PointsUtils::computeEigenVectorsAndValues(const Eigen::Matrix3f& mat_cov, const Eigen::Vector3f& centroid)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(mat_cov, Eigen::ComputeEigenvectors);
	Eigen::Matrix3f eigenVectors = eigen_solver.eigenvectors();
	Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();
    // std::cout << "The eigen values are " << eigenValuesPCA.transpose() << "\n";
	
	eigenVectors.col(2) = eigenVectors.col(0).cross(eigenVectors.col(1)); //校正主方向间垂直
	eigenVectors.col(0) = eigenVectors.col(1).cross(eigenVectors.col(2));
	eigenVectors.col(1) = eigenVectors.col(2).cross(eigenVectors.col(0));

	// TODO： check
	Eigen::Vector3f temp = eigenVectors.col(0);
	eigenVectors.col(0) = eigenVectors.col(2);
	eigenVectors.col(1) = -eigenVectors.col(1);
    eigenVectors.col(2) = temp;
	
	Eigen::Matrix4f trans_mat = Eigen::Matrix4f::Identity();
	trans_mat.block<3, 3>(0, 0) = eigenVectors.transpose();   //R.
	trans_mat.block<3, 1>(0, 3) = -1.0f * (eigenVectors.transpose()) * centroid;//  -R*t
    return {trans_mat, eigenValuesPCA};
}


std::tuple<Eigen::Matrix4f, Eigen::Vector3f> PointsUtils::computeEigenVectorsAndValues(const Points3Df& points)
{
    auto [mat_cov, centroid] = computeCovMatNormalized(points);
    auto [trans, eigen_values] = computeEigenVectorsAndValues(mat_cov, centroid);

    std::cout << "The transformation is \n" << trans << "\n";
    std::cout << "The eigen values are " << eigen_values.transpose() << "\n";
    return {trans, eigen_values};
}



std::tuple<Eigen::Matrix4f, Eigen::Vector3f>  PointsUtils::estimateInitialTransformationByPCA(const Points3D& points_source, const Points3D& points_target)
{
    auto [pca_trans_source, eigenvalues_source] = PointsUtils::computeEigenVectorsAndValues(points_source);
    auto [pca_trans_target, eigenvalues_target] = PointsUtils::computeEigenVectorsAndValues(points_target);

    Eigen::Vector3f ratio_eigen_values = eigenvalues_source.cwiseInverse().cwiseProduct(eigenvalues_target);
    ratio_eigen_values = ratio_eigen_values.cwiseSqrt();
    float scale = ratio_eigen_values.mean();
    Eigen::Vector3f scale_interval(ratio_eigen_values.minCoeff(), ratio_eigen_values.maxCoeff(), scale);
    std::cout << "The estimated scale_interval is " << scale_interval.transpose() << "\n";

    Eigen::Vector4f trans_scale(scale, scale, scale, 1.0);
    Eigen::Matrix4f trans_source = pca_trans_target.inverse() * trans_scale.asDiagonal()*pca_trans_source;
    return {trans_source, scale_interval}; 
}
