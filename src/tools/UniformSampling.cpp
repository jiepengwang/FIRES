#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <math.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <fstream>
#include <string>

#include <pcl/filters/crop_box.h>


typedef pcl::PointXYZRGBNormal PointT;

void visualizePointCloud(const pcl::PointCloud<PointT>::Ptr cloud_filtered2)
{
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer());
    viewer->addPointCloud<PointT>(cloud_filtered2, "cloud");

    pcl::PointXYZ origin(0,0,0);
    pcl::PointXYZ axisX(1,0,0), axisY(0,1,0),axisZ(0,0,1);
    viewer->addArrow(axisX, origin, 1.0, 0.0, 0.0, false, "arrow_x");
    viewer->addArrow(axisY, origin, 0.0, 1.0, 0.0, false, "arrow_y");
    viewer->addArrow(axisZ, origin, 0.0, 0.0, 1.0, false, "arrow_z");

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(50);
    }
}

int main(int argc, char** argv)
{
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>()),
                                            cloud_filter_piece(new pcl::PointCloud<PointT>());
    
    float voxel_leaf;
    std::string file_path_read, file_path_save;
    if( argc-1 == 3){
        voxel_leaf = std::atof(argv[1]);
        file_path_read = argv[2];
        file_path_save = argv[3];
        std::cout << "The size of voxel leaf is " << voxel_leaf << "\n"
                    <<"The input file path is " << file_path_read << "\n"
                    <<"The output file path is " << file_path_save << "\n";
    }
    else{
        std::cout << "Please input 3 arguments with the order:\n"
                    << "[1] float voxel leaf;\n"
                    << "[2] Input file path;\n"
                    << "[3] Output file path;\n";
        return -1;
    }
       
    pcl::io::loadPLYFile(file_path_read, *cloud);

    PointT minPt, maxPT;
    pcl::getMinMax3D(*cloud, minPt, maxPT);
    auto box = maxPT.getVector3fMap() - minPt.getVector3fMap();
    std::cout << "Box size is " << box.transpose() << "\n";
  
    pcl::PointCloud<PointT>::Ptr cloud_sampled(new pcl::PointCloud<PointT>());

    pcl::VoxelGrid<PointT> sor;
    sor.setInputCloud (cloud);
    sor.setLeafSize (voxel_leaf, voxel_leaf, voxel_leaf);
    sor.filter(*cloud_sampled);

    std::cout << "The total number of points is " << cloud->points.size()
                << "The number of sampled points is " << cloud_sampled->points.size() << std::endl;

    pcl::getMinMax3D(*cloud_sampled, minPt, maxPT);
    auto box2 = maxPT.getVector3fMap() - minPt.getVector3fMap();
    std::cout << "After sampling, box size is " << box2.transpose() << "\n";

    pcl::io::savePLYFileBinary(file_path_save, *cloud_sampled);
    return 0;
}

