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
    std::string path_dir_pieces, path_dir_GT, path_corres = "./frag_gt_corres.txt";
    int nInfoLevel;
    if( argc-1 >= 3){
        path_dir_pieces = argv[1];
        path_dir_GT = argv[2];
        nInfoLevel = std::stoi(argv[3]);
         if( argc-1 >= 4) path_corres = argv[4];
        std::cout << "The path_file_cloud is " << path_dir_pieces << "\n"
                    <<"The path_dir_GT is " << path_dir_GT << "\n"
                    <<"The nInfoLevel is " << nInfoLevel << "\n"
                    <<"The path_corres is " << path_corres << "\n";
    }
    else{
        std::cout << "Please input 3 arguments with the order:\n"
                    << "[1] path_dir_pieces;\n"
                    << "[2] path_dir_GT;\n"
                    << "[3] nInfoLevel;\n"
                    << "[4] path_corres;\n";
        return -1;
    }
    OPEN_LOG("log_preprocess_PiecesMatching");

    OPTREG::nInfoLevel = nInfoLevel;
    OPTREG::fAlphaShapeRadius = 0.005;

    int num_clouds = 100;
    CloudsVec clouds_top;
    for(int i = 0; i<num_clouds; i++){
        std::string path_top_temp = path_dir_pieces + "/piece_" + std::to_string(i) + "/top.ply";
        if(IOUtils::checkPathExistence(path_top_temp)){
            auto cloud_temp = CloudUtils::loadPLYFile(path_top_temp);
            INFO_MSG("[%d] Cloud Path: %s.\n", i, path_top_temp.c_str());
            clouds_top.push_back(cloud_temp);
        }
        else{
            break;
        }
    }

    auto vec_path_gts = IOUtils::findFilesWithExtInDir(path_dir_GT, ".ply");
    int num_gts = vec_path_gts.size();

    CloudsVec clouds_gt;
    for(int i = 0; i<num_gts; i++){
        std::string path_gt_temp = vec_path_gts[i];
        if(IOUtils::checkPathExistence(path_gt_temp)){
            auto cloud_temp = CloudUtils::loadPLYFile(path_gt_temp);
            INFO_MSG("[%d] GT cloud Path: %s.\n", i, path_gt_temp.c_str());
            clouds_gt.push_back(cloud_temp);
        }
        else{
            break;
        }
    }
    ASSERT(clouds_gt.size() == num_gts);
    INFO_MSG("Fragments: %d; GT: %d.\n", clouds_top.size(), clouds_gt.size());

    // Add Uniform sampling
    {
        START_TIMER();
        int num_sides = clouds_top.size();
        for (int i = 0; i < num_sides; i++){
            float fVoxel = CloudUtils::estimateVoxelSize(clouds_top[i], OPTREG::fNormVoxel, 20000);
            clouds_top[i] = CloudUtils::uniformSampling(clouds_top[i], fVoxel);
        }

        for (int i = 0; i < num_sides; i++){
            float fVoxel = CloudUtils::estimateVoxelSize(clouds_gt[i], OPTREG::fNormVoxel, 20000);
            clouds_gt[i] = CloudUtils::uniformSampling(clouds_gt[i], fVoxel);
        }
        INFO_MSG("Sample pieces: (%s).\n", END_TIMER());
    }
    
    START_TIMER();
    auto [pairs_best_match, transs_matching_source, transs_matching_target]  = PiecesMatching::findMatchesInRawPieces(clouds_top, clouds_gt);
    INFO_MSG("End [PiecesMatching] %s.\n", END_TIMER());

    // Save transforms of matches
    std::ofstream fWriteCorres(path_corres,std::ios::out);
    for (int i = 0; i < pairs_best_match.size(); i++){
        int index_source = pairs_best_match[i].first;
        int index_target = pairs_best_match[i].second + 1;
        INFO_MSG("[%d] Matches: %d, %d, {%s}.\n", i, index_source, index_target, vec_path_gts[index_target-1].c_str());
        fWriteCorres << index_source << " " << vec_path_gts[index_target-1] << "\n";
        fWriteCorres.flush();

    } 
    fWriteCorres.close();
    
    CLOSE_LOG();
    return 0;
}