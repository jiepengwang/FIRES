#include "CloudUtils.h"

#include <pcl/io/ply_io.h>
#include <pcl/common/io.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/random_sample.h>
#include <pcl/features/normal_3d_omp.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/visualization/pcl_visualizer.h>

// Plane segmentation
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/ModelCoefficients.h>

#include <omp.h>
#include "AlphaShape.h"
#include "IOUtils.h"
#include "LogUtils.h"



pcl::PointCloud<PointT>::Ptr CloudUtils::loadPLYFile(const std::string& file_path)
{
	std::cout << "[LoadPLYFile]\n";
	pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
	if (pcl::io::loadPLYFile(file_path, *cloud)){
		std::cerr << "ERROR: Cannot open file " << file_path << "! Aborting..." << std::endl;
		exit(-1);
	}
	removeNaNPoints(cloud);
	std::cout << "The number of points in cloud is " << cloud->points.size() << "\n";
	return cloud;
}


void CloudUtils::loadPLYFile(const std::string& file_path, pcl::PointCloud<PointT>::Ptr& cloud)
{
	std::cout << "[LoadPLYFile]\n";
	if (pcl::io::loadPLYFile(file_path, *cloud)){
		std::cerr << "ERROR: Cannot open file " << file_path << "! Aborting..." << std::endl;
		exit(-1);
	}
	
	removeNaNPoints(cloud);
	std::cout << "The number of points in cloud is " << cloud->points.size() << "\n";
	return;
}

std::vector<pcl::PointCloud<PointT>::Ptr> CloudUtils::loadPLYFiles(const std::vector<std::string>& path_files)
{
    std::vector<pcl::PointCloud<PointT>::Ptr> clouds;
    for (int i = 0; i < path_files.size(); i++){
        pcl::PointCloud<PointT>::Ptr cloud = CloudUtils::loadPLYFile(path_files[i]);
        clouds.push_back(cloud);
    }
    
    return clouds;
}


std::vector<pcl::PointCloud<PointT>::Ptr> CloudUtils::loadPLYFilesInDirectory(const std::string& path_dir)
{
	auto path_files = IOUtils::getFileListInDirectory(path_dir);
	std::vector<pcl::PointCloud<PointT>::Ptr> clouds = loadPLYFiles(path_files);
	
    return clouds;
}

void CloudUtils::savePLYFile(const std::string& file_path, const pcl::PointCloud<PointT>::Ptr& cloud)
{
	boost::filesystem::path bp_dir(file_path);

	if(!boost::filesystem::exists(bp_dir.parent_path())){
		boost::filesystem::create_directories(bp_dir.parent_path());
		std::cout << "Create the directory: " <<  bp_dir.parent_path().string() << ".\n";
	}

	if (pcl::io::savePLYFileBinary(file_path, *cloud)){
		std::cerr << "ERROR: Cannot open file " << file_path << "! Aborting..." << std::endl;
		exit(-1);
	}

	INFO_MSG("[Save PLY File] Path: %s; Points number: %d.\n", file_path.c_str(), cloud->points.size());
	return;
}

void CloudUtils::savePLYFile(const std::string& file_path, const std::vector<Eigen::Vector3f>& points)
{
	auto cloud = convertPointsToCloud(points);
	savePLYFile(file_path, cloud);
}


void CloudUtils::removeNaNPoints(pcl::PointCloud<PointT>::Ptr cloud)
{
	int num_points_origin = cloud->points.size();
	std::vector<int> nan_idx;
  	pcl::removeNaNFromPointCloud(*cloud, *cloud, nan_idx);
	cloud -> height = 1;
	cloud -> width  = static_cast<uint32_t>(cloud->points.size());
	
	if (cloud->points.size() != num_points_origin){
		std::cout << "nan_idx size is " << nan_idx.size() << "\n";
		std::cout << "cloud->points.size() is " << cloud->points.size() << "\n";
	}
	
	return;
}



std::tuple<pcl::PointCloud<PointT>::Ptr, Eigen::Matrix4f>  CloudUtils::calculatePCA(const pcl::PointCloud<PointT>::Ptr& cloud)
{
	Eigen::Vector4f pcaCentroid;
	pcl::compute3DCentroid(*cloud, pcaCentroid);
	Eigen::Matrix3f covariance;
	pcl::computeCovarianceMatrixNormalized(*cloud, pcaCentroid, covariance);
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
	Eigen::Matrix3f eigenVectors = eigen_solver.eigenvectors();
	Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();
	
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
	trans_mat.block<3, 1>(0, 3) = -1.0f * (eigenVectors.transpose()) *(pcaCentroid.head(3));//  -R*t

	pcl::PointCloud<PointT>::Ptr cloud_transed(new pcl::PointCloud<PointT>() );
	pcl::transformPointCloudWithNormals(*cloud, *cloud_transed, trans_mat);
	VERBOSE_MSG("[calculatePCA] Transformation is %s.\n", LogUtils::toString(trans_mat).c_str());

	return std::tuple<pcl::PointCloud<PointT>::Ptr, Eigen::Matrix4f>(cloud_transed, trans_mat);
}


float CloudUtils::normalizeCloud(const pcl::PointCloud<PointT>::Ptr& cloud, const bool& bShortEdge)
{
	// Scale the largest edge of bound box of cloud to 1
	auto box = getCloudBoundingBox(cloud);
	VERBOSE_MSG("[normalizeCloud] Box size is %s.\n", LogUtils::toString(box.transpose()).c_str());

	float max_edge = box[0];
	float scale;
	if(bShortEdge){
		max_edge = box[1];
		scale= 1.0/max_edge;
	}
	else 
		scale= 1.0/max_edge;
	VERBOSE_MSG("Max edge used to normalize cloud: %f.\n", max_edge);
	
	Eigen::MatrixX4f trans_scale = Eigen::Vector4f(scale, scale, scale,1.0).asDiagonal();
	pcl::transformPointCloudWithNormals(*cloud, *cloud, trans_scale);

	return scale;
}

pcl::PointCloud<PointT>::Ptr CloudUtils::scaleCloud(const pcl::PointCloud<PointT>::Ptr& cloud, const float& scale)
{
    // Scale cloud where the center of sclaing is origin
    Eigen::Vector4f trans_scale(scale, scale,scale, 1.0);
    pcl::PointCloud<PointT>::Ptr cloud_scale(new pcl::PointCloud<PointT>());
    pcl::transformPointCloudWithNormals(*cloud, *cloud_scale, trans_scale.asDiagonal());

    return cloud_scale;
}

void CloudUtils::transformPoints3D(Points3Df& points, const Eigen::Matrix4f& trans)
{
	int npc = (int)points.size();
	Eigen::Matrix3f R = trans.block<3, 3>(0, 0);
	Eigen::Vector3f t = trans.block<3, 1>(0, 3);
	
	#pragma omp parallel for shared(points, R, t)
	for (int cnt = 0; cnt < npc; cnt++) {
		Eigen::Vector3f temp = R * points[cnt] + t;
		points[cnt] = temp;
	}

	return;
}

void CloudUtils::transformPoints2D(Points2Df& points, const Eigen::Matrix4f& trans)
{
	int npc = (int)points.size();
	Eigen::Matrix2f R = trans.block<2, 2>(0, 0);
	Eigen::Vector2f t = trans.block<2, 1>(0, 2);
	
	#pragma omp parallel for shared(points, R, t)
	for (int cnt = 0; cnt < npc; cnt++) {
		Eigen::Vector2f temp = R * points[cnt] + t;
		points[cnt] = temp;
	}

	return;
}

void CloudUtils::transformCloud(const pcl::PointCloud<PointT>::Ptr& cloud, const pcl::PointCloud<PointT>::Ptr& cloud_trans, const Eigen::Matrix4f& trans)
{
	pcl::transformPointCloudWithNormals(*cloud, *cloud_trans, trans);
	return;
}

pcl::PointCloud<PointT>::Ptr CloudUtils::transformCloud(const pcl::PointCloud<PointT>::Ptr& cloud, const Eigen::Matrix4f& trans)
{
	pcl::PointCloud<PointT>::Ptr cloud_trans(new pcl::PointCloud<PointT>);
	pcl::transformPointCloudWithNormals(*cloud, *cloud_trans, trans);
	return cloud_trans;
}


pcl::PointCloud<PointT>::Ptr CloudUtils::uniformSampling(const pcl::PointCloud<PointT>::Ptr& cloud, const float& voxel_size)
{
	pcl::PointCloud<PointT>::Ptr cloud_sampled(new pcl::PointCloud<PointT>());
	pcl::UniformSampling<PointT> uniform_sampling;
	uniform_sampling.setInputCloud(cloud);
	uniform_sampling.setRadiusSearch(voxel_size);
	uniform_sampling.filter(*cloud_sampled);
	
	DEBUG_MSG("Points in raw cloud: %d; Points in sampled cloud: %d.\n", 
							cloud->points.size(), cloud_sampled->points.size());
	return cloud_sampled;
}

pcl::PointCloud<PointT>::Ptr CloudUtils::statisticalOutlierRemoval(const pcl::PointCloud<PointT>::Ptr& cloud, const int& num_neighbor, const float& std_dev)
{
    pcl::PointCloud<PointT>::Ptr cloud_filter(new  pcl::PointCloud<PointT>);

    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(num_neighbor);
    sor.setStddevMulThresh(std_dev);
    sor.filter(*cloud_filter);

    INFO_MSG("***** Inlier size: %d; Outlier size: %d. *****\n", cloud_filter->points.size(), cloud->points.size() - cloud_filter->points.size());
	
	if (OPTREG::nInfoLevel>= VISUAL_LEVEL){
        pcl::PointCloud<PointT>::Ptr cloud_outliers(new  pcl::PointCloud<PointT>);
        sor.setNegative(true);
        sor.filter(*cloud_outliers);
        CloudUtils::visualizeClouds(cloud_filter, cloud_outliers, 5.0);
    }
    
	return cloud_filter;
}

pcl::PointCloud<PointT>::Ptr CloudUtils::randomSampling(const pcl::PointCloud<PointT>::Ptr& cloud, const int& num_sample)
{
    pcl::PointCloud<PointT>::Ptr cloud_sample(new  pcl::PointCloud<PointT>);
    pcl::RandomSample<PointT> rs;
    rs.setInputCloud(cloud);
    rs.setSample(num_sample);
    rs.filter(*cloud_sample);

    return cloud_sample;
}


float CloudUtils::estimateVoxelSize(const pcl::PointCloud<PointT>::Ptr& cloud, const float& fNormVoxel, const int& num_rand_sample)
{
	auto cloud_rand_sample = randomSampling(cloud, num_rand_sample);
	auto [cloud_rand_sample_pca, trans_pca] = calculatePCA(cloud_rand_sample);

	auto box = getCloudBoundingBox(cloud_rand_sample_pca);
	DEBUG_MSG("Box: %s.\n", LogUtils::toString(box.transpose()).c_str());

	float fVoxel = box[1] * fNormVoxel;    // Use midum value
	DEBUG_MSG("Voxel size: %f.\n", fVoxel);

	return fVoxel;
}


// Tools: directory
Eigen::Vector3f CloudUtils::getCloudBoundingBox(const pcl::PointCloud<PointT>::Ptr& cloud)
{
    PointT minPt, maxPT;
    pcl::getMinMax3D(*cloud, minPt, maxPT);
    Eigen::Vector3f box = maxPT.getVector3fMap() - minPt.getVector3fMap();
    
    // DEBUG_MSG("Box size is %s.\n",  TO_CSTR(box.transpose()));
    return box;
}

// Tools: directory
Eigen::Vector3f CloudUtils::getCloudBoundingBox(const Points3Df& points)
{
	auto cloud = convertPointsToCloud(points);
	Eigen::Vector3f box = getCloudBoundingBox(cloud);

    return box;
}


// Tools: data type conversion
pcl::PointCloud<PointT>::Ptr CloudUtils::convertPointsToCloud(const std::vector<Eigen::Vector3f>& points)
{
	pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
	pcl::Vector3c color_point(255, 122, 0);
    for(auto& point : points){
        PointT point_cloud;
		point_cloud.getVector3fMap() = point;
		point_cloud.getBGRVector3cMap() = color_point;
		cloud->points.push_back(point_cloud);
    }

	return cloud;
}

Points3Df CloudUtils::convertPoints2DToHomo3D(const Points2Df& points2d)
{
	int num_points = points2d.size();
	Points3Df points3d;
	points3d.resize(num_points);

    for(int i=0; i<num_points; i++){
		points3d[i] = points2d[i].homogeneous();
    }

	return points3d;
}


Points3Df CloudUtils::convertCloudToPoints(const pcl::PointCloud<PointT>::Ptr& cloud)
{
	int num_points = cloud->points.size();
	Points3Df points;
	points.resize(num_points);
	for (int i = 0; i < num_points; i++){
		Eigen::Vector3f point = cloud->points[i].getVector3fMap();
		points[i] = point;
	}

	return points;
}

Points2Df CloudUtils::convertCloudToPoints2D(const pcl::PointCloud<PointT>::Ptr& cloud)
{
	int num_points = cloud->points.size();
	Points2Df points;
	points.resize(num_points);
	for (int i = 0; i < num_points; i++){
		Eigen::Vector2f point = cloud->points[i].getVector3fMap().head(2);
		points[i] = point;
	}

	return points;
}


void CloudUtils::convertCloudToPointsAndNormals3D(pcl::PointCloud<PointT>::Ptr cloud, Points3Df& points, Points3Df& normals)
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




std::tuple<pcl::PointCloud<PointT>::Ptr,pcl::PointCloud<PointT>::Ptr> 
CloudUtils::alignCloudsPCA(const pcl::PointCloud<PointT>::Ptr& cloud_source, 
                                const pcl::PointCloud<PointT>::Ptr& cloud_target)
{
	Eigen::Matrix4f tmp_mat4f;
    pcl::PointCloud<PointT>::Ptr cloud_pca_source(new pcl::PointCloud<PointT>()),
                                    cloud_pca_target(new pcl::PointCloud<PointT>());
    std::tie(cloud_pca_source, tmp_mat4f) = calculatePCA(cloud_source);
    std::tie(cloud_pca_target, tmp_mat4f) = calculatePCA(cloud_target);

	return {cloud_pca_source, cloud_pca_target};
}


std::tuple<pcl::PointCloud<PointT>::Ptr, pcl::PointCloud<PointT>::Ptr> 
CloudUtils::alignClouds(const pcl::PointCloud<PointT>::Ptr& cloud_source, 
                                const pcl::PointCloud<PointT>::Ptr& cloud_target)
{
    auto [cloud_pca_source, cloud_pca_target] = alignCloudsPCA(cloud_source, cloud_target);
	alignCloudsScale(cloud_pca_source, cloud_pca_target);
	visualizeClouds(cloud_pca_source, cloud_pca_target);
	
	alignCloudsAlphaShape(cloud_pca_source, cloud_pca_target);
	alignCloudsScale(cloud_pca_source, cloud_pca_target);
	visualizeClouds(cloud_pca_source, cloud_pca_target);

	return {cloud_pca_source, cloud_pca_target};
}

Eigen::Matrix4f CloudUtils::alignCloudsAlphaShape(pcl::PointCloud<PointT>::Ptr& cloud_source, 
                                                        pcl::PointCloud<PointT>::Ptr& cloud_target)
{
    // TODO: fix this problem later
	// Transform Target to Source ???
	auto box = getCloudBoundingBox(cloud_source);
	float step = 100.0;  //150
	double alpha_value = box[0]/step;
	if (alpha_value < 0.9*(1.0/step))
		alpha_value = 1.0/step;
	int num_sample = 200;

	std::vector<Eigen::Vector2f> source_ved2d, target_vec2d;
	for( int i=0; i < cloud_source->points.size(); i++){
		source_ved2d.push_back(cloud_source->points[i].getVector3fMap().head(2));
	}
	for( int i=0; i < cloud_target->points.size(); i++){
		target_vec2d.push_back(cloud_target->points[i].getVector3fMap().head(2));
	}

	AlphaShape alphashape;
	auto [contour_source, contour_target, trans_contour_source] = alphashape.registerContour(source_ved2d, target_vec2d, alpha_value, num_sample);

	if ( trans_contour_source.determinant() < 0){   // when flipping, the determinant will be less than 0
		trans_contour_source(2,2) = -1;	
	}
	std::cout << "THe transformation of contour is \n" << trans_contour_source << "\n";

	// Register contour: transform target point cloud to source point cloud using trans from alpha shape
	pcl::transformPointCloudWithNormals( *cloud_source, *cloud_source, trans_contour_source);

	return trans_contour_source;
}


float CloudUtils::alignCloudsScale(const pcl::PointCloud<PointT>::Ptr& cloud_source, 
						const pcl::PointCloud<PointT>::Ptr& cloud_target)
{
	auto box_source = getCloudBoundingBox(cloud_source);
    auto box_target = getCloudBoundingBox(cloud_target);

	std::cout << "box_source " << box_source.transpose() << "\n";
	std::cout << "box_target " << box_target.transpose() << "\n";

    Eigen::Vector3f scales = box_source.cwiseInverse().asDiagonal()*box_target;
    std::cout << "Scales in 3 directions are " << scales.transpose() << "\n";

    float scale= (scales[0]+scales[1])/2.0;
	if (std::isinf(scale))
	{
		scale = 1.0;
		INFO_MSG("Scale is infinity, so change to default value 1.0.\n");
	}
	
	std::cout << "Final scale: " << scale << "\n";

    Eigen::Vector4f mat4f_scale(scale, scale, scale, 1.0);
    pcl::transformPointCloudWithNormals(*cloud_source, *cloud_source, mat4f_scale.asDiagonal());

	return scale;
}


void CloudUtils::visualizeCloud(pcl::PointCloud<PointT>::Ptr cloud, const double& pointsize)
{
	pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer());
 

	viewer->addPointCloud<PointT> (cloud, "cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointsize, "cloud");


	auto box = getCloudBoundingBox(cloud);
	float axis_length = 1.2 * box.maxCoeff();
	DEBUG_MSG("Visualization axis length: %f.\n", axis_length);
	pcl::PointXYZ origin(0,0,0);
	pcl::PointXYZ axisX(axis_length,0,0), axisY(0,axis_length,0),axisZ(0,0,axis_length);
	viewer->addArrow(axisX, origin, 1.0, 0.0, 0.0, false, "arrow_x");
	viewer->addArrow(axisY, origin, 0.0, 1.0, 0.0, false, "arrow_y");
	viewer->addArrow(axisZ, origin, 0.0, 0.0, 1.0, false, "arrow_z");

	while (!viewer->wasStopped()){
		viewer->spinOnce(50);
		//boost::this_thread::sleep(boost::posix_time::microseconds(50));
	}
	viewer->close();

	return;
}

void CloudUtils::visualizeClouds(pcl::PointCloud<PointT>::Ptr cloud1, pcl::PointCloud<PointT>::Ptr cloud2, const double& pointsize_cloud2)
{
	pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer());
 
	pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud1_color (cloud1, 255,120,0), cloud2_color (cloud2, 0, 120, 255);
	viewer->addPointCloud<PointT> (cloud1, cloud1_color, "cloud1");
 	viewer->addPointCloud<PointT> (cloud2, cloud2_color, "cloud2");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointsize_cloud2, "cloud2");
	
	pcl::PointXYZ origin(0,0,0);
	pcl::PointXYZ axisX(1,0,0), axisY(0,1,0),axisZ(0,0,1);
	viewer->addArrow(axisX, origin, 1.0, 0.0, 0.0, false, "arrow_x");
	viewer->addArrow(axisY, origin, 0.0, 1.0, 0.0, false, "arrow_y");
	viewer->addArrow(axisZ, origin, 0.0, 0.0, 1.0, false, "arrow_z");

	while (!viewer->wasStopped()){
		viewer->spinOnce(50);
		//boost::this_thread::sleep(boost::posix_time::microseconds(50));
	}
	viewer->close();

	return;
}

void CloudUtils::sortVector3D(const std::vector<Eigen::Vector3f>& vec,  std::vector<Eigen::Vector3f>& sorted_vec,  Eigen::VectorXi& ind)
{
    ind = Eigen::VectorXi::LinSpaced(vec.size(),0,vec.size()-1); // [0 1 2 3 ... N-1]

    auto rule = [vec](int i, int j)->bool{
        if (vec[i][0] < vec[j][0])
            return true;
        else if (vec[i][0] > vec[j][0])
            return false;
        else if (vec[i][1] < vec[j][1])
            return true;
        else if (vec[i][1] > vec[j][1])
            return false;
        else if (vec[i][2] < vec[j][2])
            return true;
        else 
            return false;
        //return vec[i][0] < vec[j][0]; 
    };//

  std::sort(ind.data(),ind.data()+ind.size(),rule);
  sorted_vec.resize(vec.size());
  for(int i=0;i<vec.size();i++){
    sorted_vec[i]=vec[ind(i)];
  }
}



void CloudUtils::visualizeCorrespondences(const pcl::PointCloud<PointT>::Ptr cloud_source, const pcl::PointCloud<PointT>::Ptr cloud_target, 
											const std::vector<std::pair<int, int>>& matches, const double& matches_sample_rate)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_corres_source (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_corres_target (new pcl::PointCloud<pcl::PointXYZRGB>);

	// Sort matches
	int matches_sample_step = int(1.0 / matches_sample_rate);
	std::cout << "Sample one point among " << matches_sample_step << " points.\n";

	std::vector<Eigen::Vector3f> corres_source,  corres_target, corres_source_sorted;
	int ind_source, ind_target;
	for(int i=0; i<(matches.size()); ){	
		ind_source = matches[i].first;
		ind_target = matches[i].second;

		corres_source.push_back(cloud_source->points[ind_source].getVector3fMap());
		corres_target.push_back(cloud_target->points[ind_target].getVector3fMap());

		i += matches_sample_step;
	}
	std::cout << "The number of corres is " << matches.size() << " .\n";
	std::cout << "The number of sampled corres is " << corres_source.size() << " .\n";

	Eigen::VectorXi index_sorted;
	sortVector3D(corres_source, corres_source_sorted,index_sorted);
	std::cout << "The number of correspondences is " << matches.size() << std::endl;
	
	// add color
	int num_sampled_corres = corres_source.size();
	float step = 255.0 /(float)num_sampled_corres;
	float color_r = 255.0, color_b = .0, color_g = 0.0;
	for(int i=0; i < num_sampled_corres; i++)
	{
		color_g = i * step;
		pcl::PointXYZRGB  temp_source;
		temp_source.x = corres_source_sorted[i][0];
		temp_source.y = corres_source_sorted[i][1];
		temp_source.z = corres_source_sorted[i][2];
		temp_source.r = color_r;
		temp_source.g = color_g;
		temp_source.b = color_b;
		cloud_corres_source->push_back(temp_source);

		pcl::PointXYZRGB  temp_target;
		temp_target.x = corres_target[index_sorted[i]][0];
		temp_target.y = corres_target[index_sorted[i]][1];
		temp_target.z = corres_target[index_sorted[i]][2];
		temp_target.r = color_r;
		temp_target.g = color_g;
		temp_target.b = color_b;
		cloud_corres_target->push_back(temp_target);
	}
	std::cout << "Total points in source cloud is " << cloud_source->size() << " the number of matches is " << matches.size() << std::endl;
	
	pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer() ), viewer2 (new pcl::visualization::PCLVisualizer() ); 
	typename pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_color_source(cloud_source, 187,255,255), 
																		cloud_color_target(cloud_target, 187,255,255);
	viewer->addPointCloud<PointT> (cloud_source, cloud_color_source, "cloud_source");
    viewer2->addPointCloud<PointT> (cloud_target, cloud_color_target, "cloud_target");

	pcl::PointXYZ origin(0,0,0);
	pcl::PointXYZ axisX(1,0,0), axisY(0,1,0),axisZ(0,0,1);
	viewer->addArrow(axisX, origin, 1.0, 0.0, 0.0, false, "arrow_x");
	viewer->addArrow(axisY, origin, 0.0, 1.0, 0.0, false, "arrow_y");
	viewer->addArrow(axisZ, origin, 0.0, 0.0, 1.0, false, "arrow_z");

    
	viewer->addPointCloud<pcl::PointXYZRGB> (cloud_corres_source, "cloud_corres_source");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 12, "cloud_corres_source" );
	viewer2->addPointCloud<pcl::PointXYZRGB> (cloud_corres_target, "cloud_corres_target");
	viewer2->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 12, "cloud_corres_target" );

	while (!viewer->wasStopped()){
		viewer->spinOnce(50);
		viewer2->spinOnce(50);
		// boost::this_thread::sleep(boost::posix_time::microseconds(100));
	}
}


void CloudUtils::visualizeCorrespondences(const std::vector<Eigen::Vector3f>& points_source, const  std::vector<Eigen::Vector3f>& points_target, 
                                        const std::vector<std::pair<int, int>>& matches, const double& matches_sample_rate)
{
	auto cloud_source = convertPointsToCloud(points_source);
	auto cloud_target = convertPointsToCloud(points_target);

	visualizeCorrespondences(cloud_source, cloud_target, matches, matches_sample_rate);
}


// visualize points 
void CloudUtils::visualizeCloud(const std::vector<Eigen::Vector3f>& points, const double& pointsize)
{
	auto cloud = CloudUtils::convertPointsToCloud(points);
	CloudUtils::visualizeCloud(cloud, pointsize);
}

// visualize 2D points 
void CloudUtils::visualizeCloud(const std::vector<Eigen::Vector2f>& points2d, const double& pointsize)
{
	Points3Df points2d_homo = convertPoints2DToHomo3D(points2d);
	auto cloud = convertPointsToCloud(points2d_homo);

	visualizeCloud(cloud, pointsize);
}

// visualize points 
void CloudUtils::visualizeClouds(const std::vector<Eigen::Vector3f>& points1, 
									const std::vector<Eigen::Vector3f>& points2, const double& pointsize_points2)
{
	auto cloud1 = convertPointsToCloud(points1);
	auto cloud2 = convertPointsToCloud(points2);
	visualizeClouds(cloud1, cloud2, pointsize_points2);
}

void CloudUtils::visualizeClouds(const pcl::PointCloud<PointT>::Ptr cloud, const Points3Df& points3d, const double& pointsize_points3d)
{
	auto cloud2 = convertPointsToCloud(points3d);
	visualizeClouds(cloud, cloud2, pointsize_points3d);
}

// visualize points 
void CloudUtils::visualizeClouds(const std::vector<Eigen::Vector2f>& points1, 
									const std::vector<Eigen::Vector2f>& points2, const double& pointsize_points2)
{
	Points3Df points1_homo = convertPoints2DToHomo3D(points1);
	Points3Df points2_homo = convertPoints2DToHomo3D(points2);
	auto cloud1 = convertPointsToCloud(points1_homo);
	auto cloud2 = convertPointsToCloud(points2_homo);
	visualizeClouds(cloud1, cloud2, pointsize_points2);
}


std::string CloudUtils::extendFileNameInPath(const std::string& path, const std::string& str_expand)
{
	boost::filesystem::path bspath(path);
	std::string path_expand = bspath.parent_path().string() 
									+ "/" 			+ bspath.stem().string() 
									+ str_expand 	+ bspath.extension().string();
	
	return path_expand;
}

void CloudUtils::alignClouds(const std::string& path_reg, const std::string& path_gt)
{

	// Load ply file
	pcl::PointCloud<PointT>::Ptr cloud_reg(new pcl::PointCloud<PointT>()),
									cloud_gt(new pcl::PointCloud<PointT>());
	pcl::io::loadPLYFile(path_gt, *cloud_gt);
	pcl::io::loadPLYFile(path_reg, *cloud_reg);

	// Align clouds
	auto [cloud_align_reg, cloud_align_gt] = alignClouds(cloud_reg, cloud_gt);
	
	// Save ply file
	std::string path_reg_save = extendFileNameInPath(path_reg, "_align");
	std::string path_gt_save = extendFileNameInPath(path_gt, "_align");
	pcl::io::savePLYFile(path_gt_save, *cloud_align_gt);
	pcl::io::savePLYFile(path_reg_save, *cloud_align_reg);

	return;
}


void CloudUtils::writeTransformMatrix(const char* filepath, const Eigen::Matrix4f& transtemp)
{
	FILE* fid = fopen(filepath, "w");
	if(fid==NULL){
		ERROR_MSG("Fail to open file: %s.\n", filepath);
		exit(-1);
	}

	fprintf(fid, "%.10f %.10f %.10f %.10f\n", transtemp(0, 0), transtemp(0, 1), transtemp(0, 2), transtemp(0, 3));
	fprintf(fid, "%.10f %.10f %.10f %.10f\n", transtemp(1, 0), transtemp(1, 1), transtemp(1, 2), transtemp(1, 3));
	fprintf(fid, "%.10f %.10f %.10f %.10f\n", transtemp(2, 0), transtemp(2, 1), transtemp(2, 2), transtemp(2, 3));
	fprintf(fid, "%.10f %.10f %.10f %.10f\n", 0.0f, 0.0f, 0.0f, 1.0f);

	fclose(fid);
}


Eigen::Matrix4f CloudUtils::readTransformMatrix(const std::string& filepath)
{
	std::fstream fread(filepath, std::ios::in);
	if(!fread.is_open()){
		ERROR_MSG("Fail to open file: %s.\n", filepath.c_str());
		exit(-1);
	}
	Eigen::Matrix4f trans;

	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			fread >> trans(i,j);
		}
	}
	fread.close();
	return trans;
}

Eigen::Matrix4f CloudUtils::readFGRTransformMatrix(const std::string& filepath)
{
	std::fstream fread(filepath, std::ios::in);
	if(!fread.is_open()){
		ERROR_MSG("Fail to open file: %s.\n", filepath.c_str());
		exit(-1);
	}
	Eigen::Matrix4f trans;

	float temp;
	fread >> temp >> temp >> temp;
	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			fread >> trans(i,j);
		}
	}
	fread.close();
	return trans;
}



std::tuple<pcl::PointCloud<PointT>::Ptr,pcl::PointCloud<PointT>::Ptr>
CloudUtils::filterCloudByColor(const pcl::PointCloud<PointT>::Ptr& cloud, const Eigen::Vector3i& color_filter_kernel)
{
    int num_points = cloud->points.size();
    pcl::PointCloud<PointT>::Ptr cloud_inliers(new  pcl::PointCloud<PointT>),
                                    cloud_outliers(new  pcl::PointCloud<PointT>);
                                          
    for (int i = 0; i < num_points; i++)
    {
        auto color = cloud->points[i].getRGBVector3i();
        if (color[0] < color_filter_kernel[0] && 
             color[1] < color_filter_kernel[1] &&
             color[2] < color_filter_kernel[2]){
            cloud_outliers->points.push_back(cloud->points[i]);
        }
        else{
            cloud_inliers->points.push_back(cloud->points[i]);
        }
    }

    std::cout << "The number of points in cloud_inliers  is " << cloud_inliers->points.size() << "\n";
    std::cout << "The number of points in cloud_outliers is " << cloud_outliers->points.size() << "\n";
    

    return {cloud_inliers, cloud_outliers};
}

pcl::PointCloud<PointT>::Ptr CloudUtils::filterCloudByRadiusOutlierRemoval(const pcl::PointCloud<PointT>::Ptr& cloud, const double& search_radius, const int& num_neighbor)
{
    int num_points = cloud->points.size();
    double search_radius_ = search_radius; 
    // Remove isolated points
	pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>),  
                                    cloud_removed (new pcl::PointCloud<PointT>);
	pcl::RadiusOutlierRemoval<PointT> radius_filter;
	radius_filter.setInputCloud(cloud);
	radius_filter.setMinNeighborsInRadius(num_neighbor);
    radius_filter.setNegative(true); 
    int num_points_remove = 0;
 
    radius_filter.setRadiusSearch(search_radius_);
    radius_filter.filter(*cloud_removed);

    num_points_remove = cloud_removed->points.size();
    std::cout << "The number of cloud points is " <<  num_points 
                << ". The number of removed points is " <<  num_points_remove << "\n";

    
    

    radius_filter.setNegative(false); 
	radius_filter.filter(*cloud_filtered);
    
    std::cout << "The total number of points is " << cloud->points.size()
            << "The number of sampled points is " << cloud_filtered->points.size() << std::endl;

    CloudUtils::visualizeClouds(cloud, cloud_removed, 5);
    return cloud_filtered;
}


pcl::PointCloud<pcl::PointXYZ>::Ptr CloudUtils::extractPointsByIndices(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const pcl::Indices &indices)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_extract(new pcl::PointCloud<pcl::PointXYZ>);

    // Create the filtering object
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    boost::shared_ptr<std::vector<int>> index_ptr = boost::make_shared<std::vector<int>>(indices);

    // Extract the inliers
    extract.setInputCloud(cloud);
    extract.setIndices(index_ptr);
    extract.setNegative(false); // If set to true, you can extract point clouds outside the specified index
    extract.filter(*cloud_extract);

    return cloud_extract;
}

std::tuple<pcl::PointCloud<pcl::PointXYZ>::Ptr, Eigen::Matrix4f> CloudUtils::calculatePCA(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    Eigen::Vector4f pcaCentroid;
    pcl::compute3DCentroid(*cloud, pcaCentroid);
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*cloud, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectors = eigen_solver.eigenvectors();
    Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();

    eigenVectors.col(2) = eigenVectors.col(0).cross(eigenVectors.col(1)); //校正主方向间垂直
    eigenVectors.col(0) = eigenVectors.col(1).cross(eigenVectors.col(2));
    eigenVectors.col(1) = eigenVectors.col(2).cross(eigenVectors.col(0));

    // TODO： check
    Eigen::Vector3f temp = eigenVectors.col(0);
    eigenVectors.col(0) = eigenVectors.col(2);
    eigenVectors.col(1) = -eigenVectors.col(1);
    eigenVectors.col(2) = temp;

    Eigen::Matrix4f trans_mat = Eigen::Matrix4f::Identity();
    trans_mat.block<3, 3>(0, 0) = eigenVectors.transpose();                                   //R.
    trans_mat.block<3, 1>(0, 3) = -1.0f * (eigenVectors.transpose()) * (pcaCentroid.head(3)); //  -R*t

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transed(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*cloud, *cloud_transed, trans_mat);

    return std::tuple<pcl::PointCloud<pcl::PointXYZ>::Ptr, Eigen::Matrix4f>(cloud_transed, trans_mat);
}

bool CloudUtils::isBoundaryPoint(Eigen::Vector2f q_point, const float& max_diff_angle, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_neighbor_pca)
{
    // convert angle from degrees to radians
    float max_diff_angle_rad = max_diff_angle / 180.0 * M_PI;
    int num_points = cloud_neighbor_pca->points.size();


    // Compute the angles between each neighboring point and the query point itself
    std::vector<float> angles(num_points);
    float max_dif = FLT_MIN, dif;
    int cp = 0;


    for (int i = 0; i < num_points; i++)
    {
        if (!std::isfinite(cloud_neighbor_pca->points[i].x) ||
            !std::isfinite(cloud_neighbor_pca->points[i].y) ||
            !std::isfinite(cloud_neighbor_pca->points[i].z))
            continue;

        Eigen::Vector2f delta = cloud_neighbor_pca->points[i].getVector4fMap().head(2) - q_point;
        if (delta == Eigen::Vector2f::Zero())
            continue;

        angles[cp++] = std::atan2(delta[1], delta[0]); // the angles are fine between -PI and PI too
    }
    if (cp == 0)
        return (false);

    angles.resize(cp);
    std::sort(angles.begin(), angles.end());

    // Compute the maximal angle difference between two consecutive angles
    for (std::size_t i = 0; i < angles.size() - 1; ++i)
    {
        dif = angles[i + 1] - angles[i];
        if (max_dif < dif)
            max_dif = dif;
    }
    // Get the angle difference between the last and the first
    dif = 2 * static_cast<float>(M_PI) - angles[angles.size() - 1] + angles[0];
    if (max_dif < dif)
        max_dif = dif;

    // Check results
    return (max_dif > max_diff_angle_rad);
}

pcl::PointCloud<PointT>::Ptr CloudUtils::refineBoundary(const pcl::PointCloud<PointT>::Ptr &cloud, const pcl::PointCloud<PointT>::Ptr &cloud_bound, const int& num_neighbors)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_bound_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud, *cloud_xyz);
    pcl::copyPointCloud(*cloud_bound, *cloud_bound_xyz);

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud_xyz);

    int num_points_bound = cloud_bound->points.size();
    std::vector<bool> mask_refined_bound(num_points_bound);

    #pragma omp parallel for
    for (int i = 0; i < num_points_bound; i++)
    {
        pcl::Indices k_indices;
        std::vector<float> k_sqr_distances;

        auto point_query = cloud_bound_xyz->points[i];
        kdtree.nearestKSearch(point_query, num_neighbors, k_indices, k_sqr_distances);

        auto cloud_nearest_neighbor = extractPointsByIndices(cloud_xyz, k_indices);
        auto [cloud_nearest_neighbor_pca, trans] = calculatePCA(cloud_nearest_neighbor);

        Eigen::Vector2f point_query_trans = (trans * point_query.getVector3fMap().homogeneous()).head(2); // get xy position      

        bool boundary_flag = isBoundaryPoint(point_query_trans, 100, cloud_nearest_neighbor_pca);
        mask_refined_bound[i] = boundary_flag;
    }

    pcl::PointCloud<PointT>::Ptr cloud_bound_refine(new pcl::PointCloud<PointT>);
    for (int i = 0; i < num_points_bound; i++){
        if (mask_refined_bound[i]){
            cloud_bound_refine->points.push_back(cloud_bound->points[i]);
        }
         /* code */
    }

    int num_points_refine = cloud_bound_refine->points.size();
    INFO_MSG("Refined boundary points: %d.\n", num_points_refine );
    assert(num_points_refine != 0);

    return cloud_bound_refine;
}
