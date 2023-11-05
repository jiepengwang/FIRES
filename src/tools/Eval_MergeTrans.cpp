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

    std::string path_cloud_read, path_cloud_save, path_trans_pacalignscale, path_trans_fgr, path_trans_sicp;
    if( argc-1 == 5){
        path_cloud_read = argv[1];
        path_cloud_save = argv[2];
        path_trans_pacalignscale = argv[3];
        path_trans_fgr = argv[4];
        path_trans_sicp = argv[5];
        
        std::cout << "The path_cloud_read is " << path_cloud_read << "\n"
                    <<"The path_cloud_save is " << path_cloud_save << "\n"
                    <<"The path_trans_pacalignscale is " << path_trans_pacalignscale << "\n"
                    <<"The path_trans_fgr is " << path_trans_fgr << "\n"
                    <<"The path_trans_sicp is " << path_trans_sicp << "\n";
    }
    else{
        std::cout << "Please input 3 arguments with the order:\n"
                    << "[1] path_cloud_read;\n"
                    << "[2] path_cloud_save;\n"
                    << "[3] path_trans_pacalignscale;\n"
                    << "[4] path_trans_fgr;\n"
                    << "[5] path_trans_sicp;\n";
        return -1;
    }

    // load ply file
    auto cloud = CloudUtils::loadPLYFile(path_cloud_read);
    Eigen::Matrix4f trans_pacalignscale = CloudUtils::readTransformMatrix(path_trans_pacalignscale);
    Eigen::Matrix4f trans_fgr = CloudUtils::readFGRTransformMatrix(path_trans_fgr);
    Eigen::Matrix4f trans_sicp = CloudUtils::readTransformMatrix(path_trans_sicp);

    Eigen::Matrix4f trans_merge = trans_sicp*trans_fgr*trans_pacalignscale;


    pcl::transformPointCloudWithNormals(*cloud, *cloud, trans_merge);

    CloudUtils::savePLYFile(path_cloud_save, cloud);
    std::cout << "Done." << "\n";
    return 0;
}