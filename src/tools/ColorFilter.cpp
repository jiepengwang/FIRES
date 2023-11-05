#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>


#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>

#include <pcl/common/transforms.h>
#include <pcl/common/common.h>

#include <pcl/filters/uniform_sampling.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "CloudUtils.h"
#include "PointsUtils.h"
#include "PointsReg.h"

typedef pcl::PointXYZRGBNormal PointT;


int main(int argc, char** argv)
{

    std::string path_read_source, path_save_outliers, path_save_inliers;
    int color_threshold = 10;
    if( argc-1 == 4){
        path_read_source = argv[1];
        path_save_inliers = argv[2];
        path_save_outliers = argv[3];
        color_threshold = std::stoi(argv[4]);
        std::cout << "The input source file path is " << path_read_source << "\n"
                    <<"The input target file path is " << path_save_inliers << "\n"
                    <<"The output file path is " << path_save_outliers << "\n"
                    <<"The color_threshold is " << color_threshold << "\n";
    }
    else{
        std::cout << "Please input 3 arguments with the order:\n"
                    << "[1] path_read_source;\n"
                    << "[2] path_save_inliers;\n"
                    << "[3] path_save_outliers;\n"
                    << "[4] color_threshold;\n";
        return -1;
    }

    // load ply file
    auto cloud = CloudUtils::loadPLYFile(path_read_source);
    
    Eigen::Vector3i color_filter_kernel(color_threshold, 256, 256);  // Only check channel R
    auto [cloud_inliers, cloud_outliers] = CloudUtils::filterCloudByColor(cloud, color_filter_kernel);

    CloudUtils::savePLYFile(path_save_inliers, cloud_inliers);
    CloudUtils::savePLYFile(path_save_outliers, cloud_outliers);

    std::cout << "Done." << "\n";
    return 0;
}