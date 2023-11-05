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


int main(int argc, char** argv)
{
    std::string path_read, path_save, path_transform;
    if( argc-1 == 3){
        path_read = argv[1];
        path_transform = argv[2];
        path_save = argv[3];
        std::cout << "The input file path is " << path_read << "\n"
                    <<"The path_transform  is " << path_transform << "\n"
                    <<"The path_save is " << path_save << "\n";
    }
    else{
        std::cout << "Please input 3 arguments with the order:\n"
                    << "[1] path_read;\n"
                    << "[2] path_transform;\n"
                    << "[2] path_save;\n";
        return -1;
    }

    // load ply file
    auto cloud = CloudUtils::loadPLYFile(path_read);
    auto trans = CloudUtils::readTransformMatrix(path_transform);
    std::cout << "The transform matrix is \n" << trans << ".\n";
 
    pcl::PointCloud<PointT>::Ptr cloud_STR(new pcl::PointCloud<PointT>());
    pcl::transformPointCloudWithNormals(*cloud, *cloud_STR, trans);

    CloudUtils::savePLYFile(path_save, cloud_STR); 

    std::cout << "Done." << "\n";
    return 0;
}