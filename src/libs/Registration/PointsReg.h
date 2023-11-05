#pragma once
#include <vector>
#include <Eigen/Core>
#include <flann/flann.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef std::vector<Eigen::Vector3f> Points;
typedef std::vector<Eigen::Vector2f> Points2D;
typedef std::vector<Eigen::Vector2i> Points2Di;
typedef std::vector<Eigen::Vector3f> Points3D;

typedef flann::Index<flann::L2<float> > KDTree;
typedef std::vector<std::pair<int, int>> Pairs;  


class PointsReg
{
private:
    /* data */
public:
    PointsReg(/* args */);
    ~PointsReg();

    // KD-Tree
    template<class T>
    void buildKDTree(const std::vector<T>& data, KDTree* kd_tree);
    template<class T>
    void searchKDTree(KDTree* tree, const T& input, 
							std::vector<int>& indices,
							std::vector<float>& dists, int nn);

    // Data type conversion
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr convertPointsToCloud(const Points3D& points);
    void checkNaNsInPoints3D(Points3D points);
	std::tuple<Eigen::MatrixXf,Eigen::MatrixXf> getCorresMats(const Points3D& points_source, const Points3D& points_target, const Pairs& corres);
	std::tuple<Eigen::MatrixXf,Eigen::MatrixXf,Eigen::MatrixXf> getCorresMatsWithNormal(const Points3D& points_source,
																																											const Points3D& points_target,
																																											const Points3D& normals_target,
																																											const Pairs& corres);


    // visualization of point cloud    
    void sortVector3D(const Points3D& vec,  Points3D& sorted_vec,  Eigen::VectorXi& ind);


	void findICPCorrespondences(const Points3D& points_source, const Points3D& points_target, Pairs& corres, const float& dist_trim = -1.0);

	float calculateRMSE(const Points3D& points_source, const Points3D& points_target, 
							const Eigen::Matrix4f& trans);
	float calculateRMSE(const Eigen::MatrixXf& mat_source, const Eigen::MatrixXf& mat_target);

};


// KD-Tree
template<class T>
void PointsReg::buildKDTree(const std::vector<T>& data, KDTree* tree)
{
	int rows, dim;
	rows = (int)data.size();
	dim = (int)data[0].size();
	std::vector<float> dataset(rows * dim);
	flann::Matrix<float> dataset_mat(&dataset[0], rows, dim);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < dim; j++)
			dataset[i * dim + j] = data[i][j];
	KDTree temp_tree(dataset_mat, flann::KDTreeSingleIndexParams(15));
	temp_tree.buildIndex();
	*tree = temp_tree;
}


template<class T>
void PointsReg::searchKDTree(KDTree* tree, const T& input, 
							std::vector<int>& indices,
							std::vector<float>& dists, int nn)
{
	int rows_t = 1;
	int dim = input.size();

	std::vector<float> query;
	query.resize(rows_t*dim);
	for (int i = 0; i < dim; i++)
		query[i] = input(i);
	flann::Matrix<float> query_mat(&query[0], rows_t, dim);

	indices.resize(rows_t*nn);
	dists.resize(rows_t*nn);
	flann::Matrix<int> indices_mat(&indices[0], rows_t, nn);
	flann::Matrix<float> dists_mat(&dists[0], rows_t, nn);

	tree->knnSearch(query_mat, indices_mat, dists_mat, nn, flann::SearchParams(128));
}



