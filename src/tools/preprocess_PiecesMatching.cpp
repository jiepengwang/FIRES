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
    std::string path_dir_top, path_dir_bottom;
    int nInfoLevel;
    if( argc-1 >= 3){
        path_dir_top = argv[1];
        path_dir_bottom = argv[2];
        nInfoLevel = std::stoi(argv[3]);
        std::cout << "The path_file_cloud is " << path_dir_top << "\n"
                    <<"The path_dir_bottom is " << path_dir_bottom << "\n"
                    <<"The nInfoLevel is " << nInfoLevel << "\n";
    }
    else{
        std::cout << "Please input 3 arguments with the order:\n"
                    << "[1] path_dir_top;\n"
                    << "[2] path_dir_bottom;\n"
                    << "[3] nInfoLevel;\n";
        return -1;
    }
    OPEN_LOG("log_preprocess_PiecesMatching");

    OPTREG::nInfoLevel = nInfoLevel;
    OPTREG::fAlphaShapeRadius = 0.005;

    auto clouds_top = CloudUtils::loadPLYFilesInDirectory(path_dir_top);
    auto clouds_bottom = CloudUtils::loadPLYFilesInDirectory(path_dir_bottom);

    // Add Uniform sampling
    {
        INFO_MSG("Start Uniform sAMPLING");
        START_TIMER();
        int num_sides = clouds_top.size();
        for (int i = 0; i < num_sides; i++){
            float fVoxel = CloudUtils::estimateVoxelSize(clouds_top[i], OPTREG::fNormVoxel, 20000);
            clouds_top[i] = CloudUtils::uniformSampling(clouds_top[i], fVoxel);
        }

        int num_sides_bottom = clouds_bottom.size();
        for (int i = 0; i < num_sides_bottom; i++){
            float fVoxel = CloudUtils::estimateVoxelSize(clouds_bottom[i], OPTREG::fNormVoxel, 20000);
            clouds_bottom[i] = CloudUtils::uniformSampling(clouds_bottom[i], fVoxel);
        }

        if( num_sides_bottom != num_sides )
        {
            std::ofstream flog(path_dir_top + "/../log_run_batch.txt", std::ios::app);
            flog << "Segmetented pieces are not equal. Top: "<< num_sides << "; Bottom: " << num_sides_bottom << ".\n";
            flog.close();
        }
        INFO_MSG("Sample pieces: (%s).\n\n\n", END_TIMER());
    }
    
    START_TIMER();
    std::string path_matches = path_dir_top + "/../pieces/matches_top2bottom.txt";
    std::cout << "Path matches: " << path_matches << "\n";
    auto [pairs_best_match, transs_matching_source, transs_matching_target]  = PiecesMatching::findMatchesInRawPieces(clouds_top, clouds_bottom, path_matches);
    INFO_MSG("End [PiecesMatching] %s.", END_TIMER());

    std::string path_dir_matches_save = path_dir_top + "/../pieces";
    PiecesMatching::saveMatchedPieces(clouds_top, clouds_bottom, pairs_best_match, path_dir_matches_save);
   

    // Save transforms of matches
    for (int i = 0; i < pairs_best_match.size(); i++){
        int index_source = pairs_best_match[i].first;
        int index_target = pairs_best_match[i].second;
        
        std::string path_trans_source = path_dir_matches_save + "/piece_" + std::to_string(i) + "/trans_top.txt";
        std::string path_trans_target = path_dir_matches_save + "/piece_" + std::to_string(i) + "/trans_bottom.txt";
        CloudUtils::writeTransformMatrix(path_trans_source.c_str(), transs_matching_source[i]);
        CloudUtils::writeTransformMatrix(path_trans_target.c_str(), transs_matching_target[i]);

        if (OPTREG::nInfoLevel >= DEBUG_LEVEL){
            auto cloud_trans_top = CloudUtils::transformCloud(clouds_top[index_source], transs_matching_source[i]);
            auto cloud_trans_bottom = CloudUtils::transformCloud(clouds_bottom[index_target], transs_matching_target[i]);
            CloudUtils::visualizeClouds(cloud_trans_top, cloud_trans_bottom);
        }
    }
    CLOSE_LOG();
    return 0;
}