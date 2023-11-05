#include <iostream>
#include <vector>
#include <fstream>
#include <string>


#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>

#include <pcl/common/transforms.h>
#include <pcl/common/common.h>

#include <pcl/filters/uniform_sampling.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "CloudUtils.h"
#include "PointsUtils.h"

#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>

typedef pcl::PointXYZRGBNormal PointT;

int main(int argc, char** argv)
{

    std::string path_read_ply, path_save_features;
    if( argc-1 == 2){
        path_read_ply = argv[1];
        path_save_features = argv[2];
        std::cout << "The input source file path is " << path_read_ply << "\n"
                    <<"The output file path is " << path_save_features << "\n";
    }
    else{
        std::cout << "Please input 2 arguments with the order:\n"
                    << "[1] path_read_ply;\n"
                    << "[2] path_save_features;\n";
        return -1;
    }

    // load ply file
    auto cloud_source = CloudUtils::loadPLYFile(path_read_ply);

    
    // Uniform sampling
    std::cout << "Uniform sampling...\n";
    double voxel_size = 0.5;
    pcl::PointCloud<PointT>::Ptr cloud_sampled(new pcl::PointCloud<PointT>());
    pcl::UniformSampling<PointT> uniform_sampling;
    uniform_sampling.setInputCloud(cloud_source);
    uniform_sampling.setRadiusSearch(voxel_size);
    uniform_sampling.filter(*cloud_sampled);
    std::cout << "The number of sampled points is " << cloud_sampled->points.size() <<"\n";

    // Estimate normals
    std::cout << "Estimate normals...\n";
    double search_radius_normal = voxel_size * 5;
	if(cloud_sampled->points[0].normal_x == cloud_sampled->points[0].normal_y == cloud_sampled->points[0].normal_z == 0.0f){
		pcl::NormalEstimationOMP<PointT,PointT> normEst; 
		pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
		normEst.setInputCloud(cloud_sampled); 
		normEst.setRadiusSearch(search_radius_normal);  // _reg_args.data_args.sampling_args.uniform_sampling_leaf_size*3
		normEst.setNumberOfThreads(4);
		normEst.setSearchMethod(tree);
		normEst.compute(*cloud_sampled); 
	}


    // Calculate FPFH feature
    std::cout << "Calculate FPFH feature...\n";
    double search_radius_fpfh = voxel_size*10;
    pcl::FPFHEstimationOMP<PointT, PointT, pcl::FPFHSignature33> fest;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr object_features(new pcl::PointCloud<pcl::FPFHSignature33>());
    fest.setRadiusSearch(search_radius_fpfh);  
    fest.setInputCloud(cloud_sampled);
    fest.setInputNormals(cloud_sampled);
    fest.compute(*object_features);

    // Save FPFH features
    std::cout << "Save FPFH features...\n";
    FILE* fid = fopen(path_save_features.c_str(), "wb");
    int nV = cloud_sampled->size(), nDim = 33;
    fwrite(&nV, sizeof(int), 1, fid);
    fwrite(&nDim, sizeof(int), 1, fid);
    for (int v = 0; v < nV; v++) {
        const PointT &pt = cloud_sampled->points[v];
        float xyz[3] = {pt.x, pt.y, pt.z};
        fwrite(xyz, sizeof(float), 3, fid);
        const pcl::FPFHSignature33 &feature = object_features->points[v];
        fwrite(feature.histogram, sizeof(float), 33, fid);
    }
    fclose(fid);

    std::cout << "Done." << "\n";
    return 0;
}