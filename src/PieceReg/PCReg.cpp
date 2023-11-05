#include "PCReg.h"

#include <fstream>
#include <cmath>

#include <pcl/common/pca.h>
#include <pcl/common/common.h>

#include <pcl/filters/filter.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/normal_space.h>

#include <pcl/features/boundary.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>

#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "ICPReg.h"
#include "CloudUtils.h"
#include "LogUtils.h"


void PCReg::loadRegClouds(RegClouds& clouds, const std::string& path_dir, const std::string& side)
{
	std::string path_cloud = path_dir + "/" + side + ".ply";
	clouds.piece = CloudUtils::loadPLYFile(path_cloud);
	std::cout << "Cloud points: " << side << ", " << clouds.piece->size() << "\n";

	// Use pclbound or pclbound_refine
	if(!OPTREG::bPCLBoundRefine){  
		std::string path_bound = path_dir + "/" + side + "_pclbound.ply";
		clouds.boundary = CloudUtils::loadPLYFile(path_bound);
	}// use pclbound_refine
	else{
		std::string path_bound_refine = path_dir + "/" + side + "_pclbound_refine.ply";
		clouds.boundary = CloudUtils::loadPLYFile(path_bound_refine);
	}// use pclbound
}

/**
 * @brief 
 * Load clouds: top->source; bottom->target. 
 * Scale source cloud -> matrix: trans_scale_source
 */
void PCReg::loadClouds()
{
	loadRegClouds(_clouds_source, _path_clouds, "top");
	loadRegClouds(_clouds_target, _path_clouds, "bottom");

	_clouds_source.trans_scale_source = Eigen::Matrix4f::Identity();

	if (OPTREG::nInfoLevel >= 5){
		CloudUtils::visualizeClouds(_clouds_source.piece, _clouds_target.piece);
	}
}

void PCReg::alignClouds()
{
	/*
	Transform from source to target
	1. PCA align two clouds
	2. Scale target cloud to a unit box
	3. Align target and source scale
	4. Uniform scampling cloud to calculate alpha shape
	5. Transform clouds
	*/

	// 1. PCA
	Eigen::Matrix4f mat_pca_source, mat_pca_target;
	std::tie(_clouds_source.piece, mat_pca_source) = CloudUtils::calculatePCA(_clouds_source.piece);
	std::tie(_clouds_target.piece, mat_pca_target) = CloudUtils::calculatePCA(_clouds_target.piece);
	// CloudUtils::visualizeClouds(_clouds_source.piece, _clouds_target.piece);

	// 2. Normalize target
	auto scale_target_norm = CloudUtils::normalizeCloud(_clouds_target.piece);
	
	// 3. Scale source
	float scale_source = CloudUtils::alignCloudsScale(_clouds_source.piece, _clouds_target.piece);
	// CloudUtils::visualizeClouds(_clouds_source.piece, _clouds_target.piece);

	// Alphashape
	auto box1 = CloudUtils::getCloudBoundingBox(_clouds_source.piece);
	std::cout << box1.transpose() << "\n";
	float voxel_size = 1.0/300.0;
	auto cloud_sample_source = CloudUtils::uniformSampling(_clouds_source.piece, voxel_size);
	auto cloud_sample_target = CloudUtils::uniformSampling(_clouds_target.piece, voxel_size);
	std::cout << "The number of points in cloud_sample_source is " << cloud_sample_source->points.size() << ".\n"
				<< "The number of points in cloud_sample_target is " << cloud_sample_target->points.size() << ".\n";		

	auto trans_alphashape = CloudUtils::alignCloudsAlphaShape(cloud_sample_source, cloud_sample_target);
	pcl::transformPointCloudWithNormals(*_clouds_source.piece, *_clouds_source.piece, trans_alphashape);

	// Transform boundary
	Eigen::Matrix4f  trans_source = trans_alphashape * Eigen::Vector4f(scale_source, scale_source, scale_source, 1.0).asDiagonal()*mat_pca_source;
	Eigen::Matrix4f  trans_target = Eigen::Vector4f(scale_target_norm, scale_target_norm, scale_target_norm, 1.0).asDiagonal() * mat_pca_target;
	pcl::transformPointCloudWithNormals(*_clouds_source.boundary, *_clouds_source.boundary, trans_source);
	pcl::transformPointCloudWithNormals(*_clouds_target.boundary, *_clouds_target.boundary, trans_target);

}


bool isFileExistent(const std::string& path) 
{
    boost::filesystem::path bs_path(path);
	if (boost::filesystem::is_regular_file(bs_path)){
		std::cout << "File " << path << " is existent.\n";
		return true;
	}
	else{
		std::cout << "File " << path << " is not existent.\n";
		return false;
	}
}


void PCReg::scaleClouds(RegClouds& clouds, const float& scale)
{
	Eigen::MatrixX4f trans_scale = Eigen::Vector4f(scale,scale, scale, 1.0).asDiagonal();
	
	// Convert piece
	if (!clouds.piece->empty())
		pcl::transformPointCloudWithNormals(*clouds.piece, *clouds.piece, trans_scale);
		std::cout << "Scale piece.\n";

	// Convert boundary
	if (!clouds.boundary->empty())
		pcl::transformPointCloudWithNormals(*clouds.boundary, *clouds.boundary, trans_scale);
		std::cout << "Scale boundary.\n";
}


void PCReg::registerUsingICPWithBoundaryConstraint()
{
	// Data type conversion
	convertCloudsToPoints3Ds(_clouds_source, _points3Ds_source);
	convertCloudsToPoints3Ds(_clouds_target, _points3Ds_target);
	
	ICPReg icpReg(_reg_args, _points3Ds_source, _points3Ds_target);
	icpReg._path_clouds = _path_clouds;
	START_TIMER();
	auto [trans_icp, scale_] = icpReg.registerPoints3D_scaleicp(bUseScale);	
	INFO_MSG("[registerPoints3D_scaleicp()] %s.\n", END_TIMER());

	// save transformation matrix of source 
	INFO_MSG("The transformation calculated in ICP is %s.\n",TO_CSTR(trans_icp));
	CloudUtils::writeTransformMatrix((_path_clouds + "/reg/trans_aligntwosides_top.txt").c_str(), trans_icp*_clouds_source.trans_scale_source);

	pcl::transformPointCloudWithNormals(*_clouds_source.piece, *_clouds_source.piece, trans_icp);
	pcl::transformPointCloudWithNormals(*_clouds_source.boundary, *_clouds_source.boundary, trans_icp);

	CloudUtils::savePLYFile(_path_clouds + "/reg/source_pcl.ply", _clouds_source.piece);
	CloudUtils::savePLYFile(_path_clouds + "/reg/target_pcl.ply", _clouds_target.piece);

	CloudUtils::savePLYFile(_path_clouds + "/reg/source_bound_pcl.ply", _clouds_source.boundary);
	CloudUtils::savePLYFile(_path_clouds + "/reg/target_bound_pcl.ply", _clouds_target.boundary);

	std::ofstream fwriteS(_path_clouds + "/reg/scale_icpreg.txt");
	fwriteS << scale_ << " " << _clouds_source.trans_scale_source(0,0) << "\n";
	fwriteS.close();

	// Merge registered two sides for quick verification
	pcl::PointCloud<PointT>::Ptr cloud_temp(new pcl::PointCloud<PointT>());
	pcl::copyPointCloud((*_clouds_source.piece + *_clouds_target.piece), *cloud_temp);
	cloud_reg = cloud_temp;
	CloudUtils::savePLYFile(_path_clouds + "/reg/icp_merge.ply", cloud_temp);
}


Points3D PCReg::convertCloudToPoints3D(const pcl::PointCloud<PointT>::Ptr& cloud)
{
	Points3D points;
	for (auto& point : cloud->points){
		points.push_back(point.getVector3fMap());
	}
	return points;
}

void PCReg::convertCloudToPoints3D(pcl::PointCloud<PointT>::Ptr cloud, Points3D& points)
{
	for (auto& point : cloud->points)
		points.push_back(point.getVector3fMap());
}

void PCReg::convertCloudToNormals3D(pcl::PointCloud<PointT>::Ptr cloud, Points3D& normals)
{
	// Normal Estimation
	auto box = CloudUtils::getCloudBoundingBox(cloud);
	double search_radius = box[0]/100.0;

	if(cloud->points[0].normal_x == cloud->points[0].normal_y == cloud->points[0].normal_z == 0.0f){
		pcl::NormalEstimationOMP<PointT,PointT> normEst; 
		pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
		normEst.setInputCloud(cloud); 
		normEst.setRadiusSearch(search_radius);  // _reg_args.data_args.sampling_args.uniform_sampling_leaf_size*3
		normEst.setNumberOfThreads(4);
		normEst.setSearchMethod(tree);
		normEst.compute(*cloud); 
	}

	for (auto& point : cloud->points)
		normals.emplace_back(point.normal[0], point.normal[1], point.normal[2]);
}

void PCReg::convertCloudToPointsAndNormals3D(pcl::PointCloud<PointT>::Ptr cloud, Points3D& points, Points3D& normals)
{
	// Normal Estimation
	double search_radius = 0.6;

	if(cloud->points[0].normal_x == cloud->points[0].normal_y == cloud->points[0].normal_z == 0.0f){
		pcl::NormalEstimationOMP<PointT,PointT> normEst; 
		pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
		normEst.setInputCloud(cloud); 
		normEst.setRadiusSearch(search_radius);  // _reg_args.data_args.sampling_args.uniform_sampling_leaf_size*3
		normEst.setNumberOfThreads(4);
		normEst.setSearchMethod(tree);
		normEst.compute(*cloud); 
	}

	for (auto& point : cloud->points) {
		Eigen::Vector3f pt = point.getVector3fMap();
		Eigen::Vector3f normal(point.normal[0], point.normal[1], point.normal[2]);
		if (std::isnan(pt.dot(normal))) {
			continue;
		}
		points.emplace_back(pt);
		normals.emplace_back(normal);
	}
}

void PCReg::convertCloudsToPoints3Ds(RegClouds& clouds, RegPoints3Ds & points3Ds)
{
	// Convert piece
	if (!clouds.piece->empty())
		convertCloudToPointsAndNormals3D(clouds.piece, points3Ds.piece, points3Ds.piece_normal);

	// Convert boundary
	if (!clouds.boundary->empty())
		convertCloudToPoints3D(clouds.boundary, points3Ds.boundary);

}


template<class T>
void PCReg::buildKDTree(const std::vector<T>& data, KDTree* tree)
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
void PCReg::searchKDTree(KDTree* tree, const T& input, 
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


std::vector<Eigen::Vector2f> PCReg::extract2DContourByAlphaShape(pcl::PointCloud<PointT>::Ptr& cloud)
{
	auto box = CloudUtils::getCloudBoundingBox(cloud);
	float step = 100.0;  //150
	double alpha_value = box[0]/step;
	if (alpha_value < 0.9*(1.0/step))
		alpha_value = 1.0/step;
	int num_sample = 200;

	std::vector<Eigen::Vector2f> points_2Dcloud, points_2Dcontour;
	points_2Dcloud.resize(cloud->points.size());
	for( int i=0; i < cloud->points.size(); i++){
		points_2Dcloud[i] = cloud->points[i].getVector3fMap().head(2);
	}

	AlphaShape alphashape;
	Eigen::VectorXf angles;
	std::tie(points_2Dcontour, angles) = alphashape.getContourAngleSignature(points_2Dcloud, alpha_value, num_sample);

	return points_2Dcontour;
}



