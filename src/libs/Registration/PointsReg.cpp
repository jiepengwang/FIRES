#include "PointsReg.h"
#include "CloudUtils.h"

#include <algorithm>
#include <omp.h>

PointsReg::PointsReg(/* args */)
{
}

PointsReg::~PointsReg()
{
}

void PointsReg::checkNaNsInPoints3D(Points3D points)
{
	int num_points = points.size();
	int dim_point = points[0].size();

	std::cout << "THe number of points before erase is " << num_points << "\n";
	for (int j = num_points-1; j>=0; j-- ){
		auto point = points[j];
		for (int i = 0; i < dim_point; i++){
			if (std::isnan(point[i])){
				points.erase(points.begin()+j);
				std::cout << "Erase point[" << j << "]\n";
				break;
			}
		}
	}
	std::cout << "THe number of points after erase is " << points.size() << "\n";	
}


void PointsReg::findICPCorrespondences(const Points3D& points_source, const Points3D& points_target, 
											Pairs& corres, const float& dist_trim)
{
	// Find corresponces between points for ICP
	// Pair (source, target)
	corres.clear();
	corres.resize(points_source.size());
	std::cout << "points_target size: " <<  points_target.size() << "\n";
	KDTree tree_nncontour_target(flann::KDTreeSingleIndexParams(15));
	buildKDTree(points_target, &tree_nncontour_target);
	
	// omp_set_num_threads(4);
    int i;
    #pragma omp parallel for private(i) default(none) shared(dist_trim, tree_nncontour_target, points_target, points_source, corres)
	for(i = 0; i < points_source.size(); i++){ //
		// printf("from thread = %d\n", omp_get_thread_num());
		std::vector<int> corres_knn;
		std::vector<float> dists_knn;
		int num_knn = 1; 
		searchKDTree(&tree_nncontour_target, points_source[i], corres_knn, dists_knn, num_knn);
		int ind_target = corres_knn[0];
		
		auto point_target = points_target[ind_target];
		auto point_source = points_source[i];
		float dist = (point_source-point_target).norm();

		if (dist_trim < 0){
			corres[i] = std::pair<int,int>(i, ind_target);
		}
		else if (dist < dist_trim){
			corres[i] = std::pair<int,int>(i, ind_target);
		}   	
	}// build trimmed ICP corres of contour neighbors
}

std::tuple<Eigen::MatrixXf,Eigen::MatrixXf> PointsReg::getCorresMats(const Points3D& points_source, const Points3D& points_target, const Pairs& corres)
{
    // Convert corres points to matrix
    int num_corres = corres.size();
    Eigen::MatrixXf mat_source(3, num_corres);
    Eigen::MatrixXf mat_target(3, num_corres);
	#pragma omp parallel for
    for (int i = 0; i < num_corres; i++){
        int ind_source = corres[i].first;
        int ind_target = corres[i].second;

        auto point_source = points_source[ind_source];
        auto point_target = points_target[ind_target];

        mat_source.block<3,1>(0,i) = point_source;
        mat_target.block<3,1>(0,i) = point_target;
    }
    return {mat_source, mat_target}; 
}

std::tuple<Eigen::MatrixXf,Eigen::MatrixXf,Eigen::MatrixXf>
		PointsReg::getCorresMatsWithNormal(const Points3D& points_source,
																			 const Points3D& points_target,
																			 const Points3D& normals_target,
																			 const Pairs& corres)
{
    // Convert corres points to matrix
    int num_corres = corres.size();
    Eigen::MatrixXf mat_source(3, num_corres);
    Eigen::MatrixXf mat_target(3, num_corres);
		Eigen::MatrixXf mat_target_normal(3, num_corres);
	#pragma omp parallel for
    for (int i = 0; i < num_corres; i++) {
        int ind_source = corres[i].first;
        int ind_target = corres[i].second;

        auto point_source = points_source[ind_source];
        auto point_target = points_target[ind_target];
				auto normal_target = normals_target[ind_target];

        mat_source.block<3,1>(0,i) = point_source;
        mat_target.block<3,1>(0,i) = point_target;
				mat_target_normal.block<3,1>(0,i) = normal_target;
    }
    return { mat_source, mat_target, mat_target_normal }; 
}


float PointsReg::calculateRMSE(const Points3D& points_source, const Points3D& points_target, 
							const Eigen::Matrix4f& trans)
{
	Points3D points_source_cpy = points_source;
	if (trans != Eigen::Matrix4f::Identity()){
		CloudUtils::transformPoints3D(points_source_cpy, trans);
	}
	
	Pairs corres;
	findICPCorrespondences(points_source_cpy, points_target, corres);
	
	auto [mat_source, mat_target] = getCorresMats(points_source_cpy, points_target, corres);

	Eigen::MatrixXf mat_diff = mat_source - mat_target;
	float err_SE = mat_diff.colwise().squaredNorm().sum();
	float err_RMSE = std::sqrt(err_SE/(float)(corres.size()));
	
	printf("RMSE: %0.4e\n", err_RMSE);

	return err_RMSE;
}

float PointsReg::calculateRMSE(const Eigen::MatrixXf& mat_source, const Eigen::MatrixXf& mat_target)
{
	int num_corres = mat_source.cols();
	Eigen::MatrixXf mat_diff = mat_source - mat_target;
	float err_SE = mat_diff.colwise().squaredNorm().sum();
	float err_RMSE = std::sqrt(err_SE/(float)(num_corres));
	
	return err_RMSE;
}