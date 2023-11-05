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
#include "PiecesMatching.h"
#include "IOUtils.h"
#include "LogUtils.h"

typedef pcl::PointXYZRGBNormal PointT;

int main(int argc, char** argv)
{
    std::string path_file_cloud;
    float image_width = -1;
    if( argc-1 == 2){
        path_file_cloud = argv[1];
        image_width = std::stof(argv[2]);
        std::cout << "The path_file_cloud is " << path_file_cloud << "\n"
                    <<"The image_width is " << image_width << "\n";
    }
    else{
        std::cout << "Please input 2 arguments with the order:\n"
                    << "[1] path_file_cloud;\n"
                    << "[2] image_width;\n";
        return -1;
    }
    OPEN_LOG("log_preprocess_ExtractPieces");
    // load ply file
    auto cloud = CloudUtils::loadPLYFile(path_file_cloud);
    auto [cloud_pca, trans] = CloudUtils::calculatePCA(cloud);
    
    auto box = CloudUtils::getCloudBoundingBox(cloud_pca);
    float scale = (float)image_width / box.maxCoeff() * 0.5 ; // 0.5: Because pieces only occupy half of image width
    std::cout << "The scale is " << scale << "\n";
    auto cloud_pca_scale = CloudUtils::scaleCloud(cloud_pca, scale);
    // CloudUtils::visualizeCloud(cloud_pca_scale, 3);
    
    START_TIMER();
    // Segement and extrance pieces
    float search_radius = 10;
    auto [mask_cloud, num_pieces] = PiecesMatching::segmentPiecesWithRegionGrowing(cloud_pca_scale, 5);
    auto cloud_pieces = PiecesMatching::extractPiecesByID(cloud, mask_cloud, num_pieces);
    INFO_MSG("End [ExtractPieces] %s.", END_TIMER());

    // Save segmented pieces
    auto [parent_path, stem, ext] =  IOUtils::getFileParentPathStemExt(path_file_cloud);
    std::string path_dir_pieces_save = parent_path + "/" + stem + "_pieces";
    PiecesMatching::saveExtractedPieces(cloud_pieces, path_dir_pieces_save);
    
    std::cout << "Done." << "\n";
    CLOSE_LOG();
    return 0;
}