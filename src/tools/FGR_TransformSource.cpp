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

typedef pcl::PointXYZRGBNormal PointT;


int main(int argc, char** argv)
{

    std::string path_read_source, path_read_target, path_read_source_trans, path_save_source;
    if( argc-1 == 4){
        path_read_source = argv[1];
        path_read_target = argv[2];
        path_read_source_trans = argv[3];
        path_save_source = argv[4];
        std::cout << "The input source file path is " << path_read_source << "\n"
                    <<"The input target file path is " << path_read_target << "\n"
                    <<"The path_read_source_trans is " << path_read_source_trans << "\n"
                    <<"The output file path is " << path_save_source << "\n";
    }
    else{
        std::cout << "Please input 3 arguments with the order:\n"
                    << "[1] path_read_source;\n"
                    << "[2] path_read_target;\n"
                    << "[3] path_read_source_trans;\n"
                    << "[4] path_save_source;\n";
        return -1;
    }

    // load ply file
    auto cloud_source = CloudUtils::loadPLYFile(path_read_source);
    auto cloud_target = CloudUtils::loadPLYFile(path_read_target);

    Eigen::Matrix4f trans;
    std::fstream fin(path_read_source_trans, std::ios::in);
    int tmp;
    fin >> tmp >> tmp >> tmp;
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++){
            fin >> trans(i,j);
        }  
    }
    fin.close();
    std::cout << "The transformation is \n" << trans << ".\n";
    
    pcl::transformPointCloudWithNormals(*cloud_source, *cloud_source, trans);
    // CloudUtils::visualizeClouds(cloud_source, cloud_target);

    CloudUtils::savePLYFile(path_save_source, cloud_source);
    std::cout << "Done." << "\n";
    return 0;
}