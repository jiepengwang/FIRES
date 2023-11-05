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

#include <pcl/kdtree/kdtree_flann.h>

std::vector<int> pclKnnUsingKDTree(const pcl::PointCloud<PointT>::Ptr& cloud, const PointT& point,
                                        const int& num_neighbors)
{
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud (cloud);

    int K = num_neighbors;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);

    if (kdtree.nearestKSearch (point, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 ){
        return pointIdxNKNSearch;
    }
    else{
        std::cout << "Fail to find nearest neighbors.\n";
        exit(-1);
    }
}
int main(int argc, char** argv)
{
    std::string path_file_cloud;
    float image_width = -1;
    int nInfoLevel;
    if( argc-1 >= 3){
        path_file_cloud = argv[1];
        image_width = std::stof(argv[2]);
        nInfoLevel = std::stoi(argv[3]);
        if (argc-1 >= 4){
            OPTREG::bCoarseReconstruction = std::stoi(argv[4]); // 0, false; 1, true;
            INFO_MSG("Use coarse reconstruction.\n");
        }
        
        std::cout << "The path_file_cloud is " << path_file_cloud << "\n"
                    <<"The image_width is " << image_width << "\n"
                    <<"The nInfoLevel is " << nInfoLevel << "\n";
    }
    else{
        std::cout << "Please input 3 arguments with the order:\n"
                    << "[1] path_file_cloud;\n"
                    << "[2] image_width;\n"
                    << "[3] nInfoLevel;\n"
                    << "[4] [Optional] use coarse reconstruction;\n";
        return -1;
    }
    
    OPEN_LOG("log_preprocess_ExtractPieces_NoSample");

    OPTREG::nMinPiecePoints = 6000;
    int num_rand_sample = 30000;
    int nMaxPixelsDiff = 15;

    if (OPTREG::bCoarseReconstruction){
        OPTREG::nMinPiecePoints = 3000;
        num_rand_sample = 10000;
        nMaxPixelsDiff = 10;
    }

    OPTREG::nInfoLevel = nInfoLevel;
    // load ply file
    auto cloud = CloudUtils::loadPLYFile(path_file_cloud);

    auto cloud_rand_sample = CloudUtils::randomSampling(cloud, num_rand_sample);
    auto [cloud_rand_sample_pca, trans] = CloudUtils::calculatePCA(cloud_rand_sample);


    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rand_sample_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud,*cloud_xyz);
    pcl::copyPointCloud(*cloud_rand_sample,*cloud_rand_sample_xyz);

    
    // scale cloud to half of image resolution
    auto box = CloudUtils::getCloudBoundingBox(cloud_rand_sample_pca);
    float scale = (float)image_width / box.maxCoeff() * 0.5 ; // 0.5: Because pieces only occupy half of image width
    std::cout << "The scale is " << scale << "\n";

    
    // Segement cloud_rand_sample
    float search_radius = float(nMaxPixelsDiff)/scale;
    START_TIMER();
    auto [mask_cloud_rand_sample, num_pieces] = PiecesMatching::segmentPiecesWithRegionGrowing(cloud_rand_sample, search_radius);
    INFO_MSG("Segment sampled cloud (%d points): %s.\n", num_rand_sample, END_TIMER());

    // Segment cloud
    int num_points = cloud->points.size();
    std::vector<int> mask_cloud(num_points, -1);

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud_rand_sample_xyz);

    UPDATE_TIMER();
    #pragma omp parallel for
    for(int i=0; i<num_points; i++){
        auto point = cloud_xyz->points[i];

        int K = 1;
        std::vector<int> pointIdxNKNSearch(K);
        std::vector<float> pointNKNSquaredDistance(K);
        kdtree.nearestKSearch(point, K, pointIdxNKNSearch, pointNKNSquaredDistance);

        int point_index_search = pointIdxNKNSearch[0];
        mask_cloud[i] = mask_cloud_rand_sample[point_index_search];
    }
    INFO_MSG("Segment raw cloud (%d points): %s.\n", num_points, END_TIMER());

    
    auto cloud_pieces = PiecesMatching::extractPiecesByID(cloud, mask_cloud, num_pieces);

    // Save segmented pieces
    auto [parent_path, stem, ext] =  IOUtils::getFileParentPathStemExt(path_file_cloud);
    std::string path_dir_pieces_save = parent_path + "/" + stem + "_pieces";
    PiecesMatching::saveExtractedPieces(cloud_pieces, path_dir_pieces_save);
    
    INFO_MSG("End [ExtractPieces] %s.", END_TIMER());

    std::cout << "Done." << "\n";
    CLOSE_LOG();
    return 0;
}