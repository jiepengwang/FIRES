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

    std::string path_read_source, path_read_target, path_save_source;
    if( argc-1 == 3){
        path_read_source = argv[1];
        path_read_target = argv[2];
        path_save_source = argv[3];
        std::cout << "The input source file path is " << path_read_source << "\n"
                    <<"The input target file path is " << path_read_target << "\n"
                    <<"The output file path is " << path_save_source << "\n";
    }
    else{
        std::cout << "Please input 3 arguments with the order:\n"
                    << "[1] path_read_source;\n"
                    << "[2] path_read_target;\n"
                    << "[3] path_save_source;\n";
        return -1;
    }

    // load ply file
    auto cloud_source = CloudUtils::loadPLYFile(path_read_source);
    auto cloud_target = CloudUtils::loadPLYFile(path_read_target);

    Points3D points_source = CloudUtils::convertCloudToPoints(cloud_source);
    Points3D points_target = CloudUtils::convertCloudToPoints(cloud_target);

    auto [mat_trans, scale_interval] = PointsUtils::estimateInitialTransformationByPCA(points_source, points_target);
    
    Points3D points_source_trans = points_source;
    CloudUtils::transformPoints3D(points_source_trans, mat_trans);

    // CloudUtils::visualizeClouds(points_source_trans, points_target);
    
    CloudUtils::savePLYFile(path_save_source, points_source_trans);

    std::fstream fwrite("./scale_FGR_PCAAlignScale.txt", std::ios::out);
    fwrite << scale_interval[2] << "\n";
    fwrite.close();

    CloudUtils::writeTransformMatrix("./trans_FGR_PCAAlignScale.txt", mat_trans);
    
    std::cout << "Done." << "\n";
    return 0;
}