#include "PiecesMatching.h"
#include <tuple>

#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "AlphaShape.h"
#include "CloudUtils.h"
#include "IOUtils.h"
#include "LogUtils.h"

#include <boost/filesystem.hpp>
#include <fstream>


// Segment different pieces
pcl::PointCloud<PointT>::Ptr PiecesMatching::extractPieceID(const pcl::PointCloud<PointT>::Ptr& cloud, const std::vector<int>& mask_cloud, const int& piece_id)
{
    pcl::PointCloud<PointT>::Ptr cloud_piece(new pcl::PointCloud<PointT>());
    int num_points_piece = std::count(mask_cloud.begin(), mask_cloud.end(), -1);
    cloud_piece->points.reserve(num_points_piece);

    int num_points = cloud->points.size();
    for (int i = 0; i < num_points; i++){
        if (mask_cloud[i] == piece_id)
            cloud_piece->points.push_back(cloud->points[i]);
    }

    VERBOSE_MSG("Segmented piece [%d]: %d points.\n", piece_id, cloud_piece->points.size());
    return cloud_piece;
}

bool PiecesMatching::segmentPieceIDByRegionGrowing(const pcl::PointCloud<PointT>::Ptr& cloud, std::vector<int>& mask_cloud, 
                                                        const int& piece_id, const float& search_radius)
{
    int num_points = cloud->points.size();
    int num_unseg = std::count(mask_cloud.begin(), mask_cloud.end(), -1);

    // Find and add the first seed
    std::vector<int> indices_seed;
    indices_seed.reserve(num_points);

    auto iter_mask = std::find(mask_cloud.begin(), mask_cloud.end(), -1); //std::experimental::randint(0, num_points);
    int index_seed = iter_mask - mask_cloud.begin();
    if (iter_mask == mask_cloud.end()){
        std::cout << "Fail to find a seed point in mask_cloud.\n";
        return false;
    }
    indices_seed.push_back(index_seed);

    //pcl knn radius search
    pcl::KdTreeFLANN<PointT> tree_cloud;
    tree_cloud.setInputCloud(cloud);

    std::vector<int> radius_search_indices;
    std::vector<float> radius_search_dists;
    radius_search_indices.reserve(100);
    radius_search_dists.reserve(100);
    // float search_radius = 6; // 2 pixels
    do{
        int tmp_index = indices_seed.back();
        indices_seed.pop_back();

        if (mask_cloud[tmp_index]!=-1){  // The mask of this point has been updated;
            continue;
        }

        PointT tmp_point = cloud->points[tmp_index];
        tree_cloud.radiusSearch(tmp_point, search_radius, radius_search_indices, radius_search_dists);
        if(!radius_search_indices.empty())
            mask_cloud[tmp_index] = piece_id;
            indices_seed.insert(indices_seed.end(), radius_search_indices.begin(), radius_search_indices.end());
        
        // std::cout << "The number of seeds now is " << indices_seed.size() << "\n";
    } while (!indices_seed.empty());

    int num_unseg2 = std::count(mask_cloud.begin(), mask_cloud.end(), -1);
    if (num_unseg2 < num_unseg){
        VERBOSE_MSG("Segmented points by region growing: %d with ID [%d].\n", (num_unseg - num_unseg2), piece_id);
        return true;
    }
    else
        return false;
}


bool PiecesMatching::saveExtractedPieces(const std::vector<pcl::PointCloud<PointT>::Ptr>& cloud_pieces, const std::string& path_dir_save)
{
    int num_pieces = cloud_pieces.size();
    for (int i = 0; i < num_pieces; i++){
        std::string path_save_i = path_dir_save+ "/pieceseg_"+std::to_string(i)+".ply";
        CloudUtils::savePLYFile(path_save_i, cloud_pieces[i]);

        if (OPTREG::nInfoLevel>=5){
            CloudUtils::visualizeCloud(cloud_pieces[i]);
        }   
    }
    return true;
}

std::vector<pcl::PointCloud<PointT>::Ptr> PiecesMatching::extractPiecesByID(const pcl::PointCloud<PointT>::Ptr& cloud, const std::vector<int>& mask_cloud, const int& num_pieces)
{
    std::vector<pcl::PointCloud<PointT>::Ptr> cloud_pieces;
    int num_points_pieces = 0;
    for (int i = 1; i < num_pieces+1; i++)  // Because the ID of pieces begins from 1
    {
        int piece_id = i;
        auto cloud_piece_id = extractPieceID(cloud, mask_cloud, piece_id);
        
        int num_points_piece_id = cloud_piece_id->points.size();
        if(num_points_piece_id>OPTREG::nMinPiecePoints){
            cloud_pieces.push_back(cloud_piece_id);
            num_points_pieces += num_points_piece_id;
            // CloudUtils::visualizeCloud(cloud_piece_id, 3);
            std::cout << "There are " << cloud_piece_id->points.size() 
                << " points in piece " << piece_id << ".\n";
        }
    }

    INFO_MSG("Points in raw cloud: %d; Points in extraced pieces: %d; Points removed: %d.\n",
                    cloud->points.size(), num_points_pieces, (cloud->points.size() - num_points_pieces));
    return cloud_pieces;
}


std::tuple<std::vector<int>, int> PiecesMatching::segmentPiecesWithRegionGrowing(const pcl::PointCloud<PointT>::Ptr& cloud, const float& search_radius)
{

    int num_points = cloud->points.size();
    std::vector<int> mask_cloud(num_points, -1);

    //TODO:: VoxelGrid sampling to improve effciency

    // Region growing
    int piece_id = 0;
    while (std::find(mask_cloud.begin(), mask_cloud.end(), -1) != mask_cloud.end()){
        piece_id++;
        segmentPieceIDByRegionGrowing(cloud, mask_cloud, piece_id, search_radius);
    }

    return {mask_cloud, piece_id};
}   



// Pieces Matching
std::tuple<std::vector<std::pair<int, int>>, std::vector<Eigen::Matrix4f>>
PiecesMatching::findCostOfBestMatchUsingAlphaShape(const std::vector<std::pair<Points2D, Eigen::VectorXf>>& contour_angless_source,
                                                                        const std::vector<std::pair<Points2D, Eigen::VectorXf>>& contour_angless_target,
                                                                        const std::string& path_matches,
                                                                        const bool& b_only_one_side)
{
    // Pairs of best match: source, target
    int num_pieces_source = contour_angless_source.size();
    int num_pieces_target = contour_angless_target.size();
    INFO_MSG("Contour matching: Source, %d; Target, %d.\n", num_pieces_source, num_pieces_target);
    
    // 1. Iterate all possibilities
    Eigen::Matrix<Eigen::VectorXf, Eigen::Dynamic,Eigen::Dynamic> mat_indices(num_pieces_source, num_pieces_target);
    Eigen::MatrixXf mat_mincost(num_pieces_source, num_pieces_target);
    for (size_t i = 0; i < num_pieces_source; i++){
        auto angles_source = contour_angless_source[i].second;

        for (int j = 0; j < num_pieces_target; j++){
            auto angles_target = contour_angless_target[j].second;
            auto [indices, min_cost] = AlphaShape::findBestMatch(angles_source, angles_target);

            mat_indices(i, j) = indices;
            mat_mincost(i, j) = min_cost;
            std::cout << min_cost<< "\n";
            // std::cout << indices.transpose()<< "\n\n";
        }
    }
    INFO_MSG("The mat_mincost of pieces is:\n%s.\n",LogUtils::toString(mat_mincost).c_str());

    // 2. Find the optimal matches
    Eigen::MatrixXf mask_mat_mincost(num_pieces_source, num_pieces_target);
    mask_mat_mincost.setZero();

    Eigen::VectorXf::Index index_minvalue_col;
    int sum_index = 0;
    std::vector<std::pair<int, int>> pairs_best_match;
    for (int i = 0; i < mat_mincost.rows(); i++){
        /* code */
        auto min_value = mat_mincost.row(i).minCoeff(&index_minvalue_col);
        sum_index += index_minvalue_col;

        pairs_best_match.push_back(std::pair<int, int>(i, index_minvalue_col));
        INFO_MSG("The matches is %d-%d.\n", i, index_minvalue_col);
    }
    if(path_matches != "empty" && boost::filesystem::exists(path_matches)){
        std::cout << "Use board matches.******************************************************************\n\n";
        std::ifstream fmatches(path_matches);
        int id_top, id_bottom;
        if (fmatches.is_open()){
           for (int i = 0; i < mat_mincost.rows(); i++){
               fmatches >> id_top >> id_bottom;
               pairs_best_match[i] = std::pair<int, int>(id_top, id_bottom);
               std::cout << "Matches: " << id_top << "-" << id_bottom << "\n"; 
           }
        }
        
    }
    std::cout << "Use board matches.2******************************************************************\n\n";

    // 3. Guarantee that the matches are one-to-one between source and target 
    int sum_index_gt = (0 + num_pieces_target-1) * num_pieces_target / 2;
    if (sum_index != sum_index_gt ){
        std::cerr << "There are repetitive matches. \n";
        // exit(-1);
    } 

    // 4. Calculate the contour transformation for the optimal matches
    std::vector<Eigen::Matrix4f> transs(num_pieces_source);

    int num_contour_points =  contour_angless_source[0].first.size();
    Points2Df points_source_sorted(num_contour_points);
    for (size_t i = 0; i < pairs_best_match.size(); i++)
    {
        int index_source = pairs_best_match[i].first;
        int index_target = pairs_best_match[i].second;

        auto indices_source = mat_indices(index_source, index_target);
        auto points_source = contour_angless_source[index_source].first;
        // Sort source points
        for (int j = 0; j < num_contour_points; j++){
           points_source_sorted[j] = points_source[indices_source[j]];
        }

        Eigen::Matrix4f trans = AlphaShape::calcualteTransformation(points_source_sorted, contour_angless_target[index_target].first);
        transs[i] = trans;
    }
    

    return {pairs_best_match, transs};  
}


std::vector<std::pair<Points2D, Eigen::VectorXf>> PiecesMatching::get2DContourAndDirectedAnglesOfClouds(const std::vector<pcl::PointCloud<PointT>::Ptr>& clouds)
{
    std::vector<std::pair<Points2D, Eigen::VectorXf>> contour_angless;
    contour_angless.resize(clouds.size());

   
	int num_sample = 200;
    float fNormVoxel =  4 * OPTREG::fNormVoxel;
    for (int i = 0; i < clouds.size(); i++){
        //TODO: Add UniformSampling to improve effieciency
        
        float fVoxel = CloudUtils::estimateVoxelSize(clouds[i], fNormVoxel, 20000);
        auto cloud_unisample = CloudUtils::uniformSampling(clouds[i], fVoxel);

        double alpha_value = OPTREG::fAlphaShapeRadius; //2 * fVoxel;
        DEBUG_MSG("Cloud voxel size: %f; Alpha value: %f.\n", fVoxel, alpha_value);

        auto points2d = CloudUtils::convertCloudToPoints2D(cloud_unisample);

        auto [contour_source, angles_source] = AlphaShape::getContourAngleSignature(points2d, alpha_value, num_sample);
        contour_angless[i] = std::pair<Points2D, Eigen::VectorXf>(contour_source, angles_source);
    }

    return contour_angless;
}


void PiecesMatching::saveMatchedPieces(const std::vector<pcl::PointCloud<PointT>::Ptr>& clouds_pieces_source, 
                                    const std::vector<pcl::PointCloud<PointT>::Ptr>& clouds_pieces_target,
                                    const std::vector<std::pair<int, int>>& pairs,
                                    const std::string& path_dir_save)
{
    int num_pairs = pairs.size();

    std::string path_file_source, path_file_target;
    for (int i = 0; i < num_pairs; i++){
        int index_source = pairs[i].first;
        int index_target = pairs[i].second;

        path_file_source = path_dir_save + "/piece_" + std::to_string(i) + "/top.ply";
        path_file_target = path_dir_save + "/piece_" + std::to_string(i) + "/bottom.ply";

        CloudUtils::savePLYFile(path_file_source, clouds_pieces_source[index_source]);
        CloudUtils::savePLYFile(path_file_target, clouds_pieces_target[index_target]);
    }

    return;
}


std::vector<pcl::PointCloud<PointT>::Ptr> PiecesMatching::normalizeCloudsAfterPCA(const std::vector<pcl::PointCloud<PointT>::Ptr>& clouds)
{
    int num_clouds = clouds.size();
    std::vector<pcl::PointCloud<PointT>::Ptr> clouds_pca_norm;
    for (int i = 0; i < num_clouds; i++){
        auto [cloud_pca, trans] =  CloudUtils::calculatePCA(clouds[i]);
        CloudUtils::normalizeCloud(cloud_pca);
        clouds_pca_norm.push_back(cloud_pca);
    }

    return clouds_pca_norm;
}

std::tuple<std::vector<pcl::PointCloud<PointT>::Ptr>,  
            std::vector<Eigen::Matrix4f>,
            std::vector<float>>
PiecesMatching::normalizeCloudsAfterPCA2(const std::vector<pcl::PointCloud<PointT>::Ptr>& clouds, const std::string& side)
{
    int num_clouds = clouds.size();
    std::vector<pcl::PointCloud<PointT>::Ptr> clouds_pca_norm(num_clouds);
    std::vector<Eigen::Matrix4f> transs(num_clouds);
    std::vector<float> scales(num_clouds);
    for (int i = 0; i < num_clouds; i++){
        auto [cloud_pca, trans] =  CloudUtils::calculatePCA(clouds[i]);
        auto scale = CloudUtils::normalizeCloud(cloud_pca, true); // use short edge in XY plane

        if(side=="top" || side=="bottom"){
            Eigen::MatrixXf data = cloud_pca->getMatrixXfMap().colwise().normalized(); 
            Eigen::MatrixXf points_normal = data.block(4,0,3, data.cols());

            Eigen::Vector3f normal_m = points_normal.rowwise().mean();

            Eigen::Matrix4f mat_flip = Eigen::Matrix4f::Identity();
            mat_flip(1,1) = -1;
            mat_flip(2,2) = -1;


            Eigen::Vector3f p1 = normal_m;
            Eigen::Vector3f p2;
            if(side=="top"){
                p2 = Eigen::Vector3f(0,0,1);
            }
            if(side=="bottom"){
                p2 = Eigen::Vector3f(0,0,-1);
            }
            
            float norminator = p1.dot(p2);
            float denominator = p1.norm() * p2.norm();
            float den_p2 = p2.norm();
            float angle = acos(norminator/(denominator+1e-5))/3.1415926 * 180;
            std::cout << "angle is " << angle << "\n";

            if (angle>90){
                std::cout << side << " " << i << " Flip this point cloud...\n";
                pcl::transformPointCloudWithNormals(*cloud_pca, *cloud_pca, mat_flip);
                trans = mat_flip*trans;
            }
        }
        else{
            ASSERT(false);
        }

        clouds_pca_norm[i] = cloud_pca;
        transs[i] = trans;
        scales[i] = scale;
    }

    return {clouds_pca_norm, transs, scales};
}


Eigen::Matrix4f getScaleMatrix(const float& scale){
    Eigen::Matrix4f mat_scale = Eigen::Vector4f(scale, scale, scale, 1).asDiagonal();
    return mat_scale;
}

std::tuple<std::vector<std::pair<int, int>> , std::vector<Eigen::Matrix4f>, std::vector<Eigen::Matrix4f> >
PiecesMatching::findMatchesInRawPieces(const std::vector<pcl::shared_ptr<pcl::PointCloud<PointT>>>& clouds_source, 
                                        const std::vector<pcl::shared_ptr<pcl::PointCloud<PointT>>>& clouds_target,
                                        const std::string& path_matches,
                                        const bool& b_only_one_side)
{
    START_TIMER();
    INFO_MSG("Cloud: Source, %d; Target, %d.\n", clouds_source.size(), clouds_target.size());

    auto [clouds_pca_norm_source, transs_pca_source, scales_source] = normalizeCloudsAfterPCA2(clouds_source, "top");
    auto [clouds_pca_norm_target, transs_pca_target, scales_target] = normalizeCloudsAfterPCA2(clouds_target, "bottom");
    DEBUG_MSG("[normalizeCloudsAfterPCA] Time elapsed: %s.\n", END_TIMER());

    if(OPTREG::nInfoLevel >= VISUAL_LEVEL){
        // Visualization of boundaries: Verbose level
        int n_cloud = clouds_pca_norm_source.size();
        for(int i = 0; i<n_cloud; i++){
            DEBUG_MSG("Visualize source contour: %d.\n", i);
            CloudUtils::savePLYFile("./test" + std::to_string(i) + "_top.ply", clouds_pca_norm_source[i]);
            CloudUtils::savePLYFile("./test" + std::to_string(i) + "_bottom.ply", clouds_pca_norm_target[i]);
        }
    }

    UPDATE_TIMER();
    auto contour_angless_source = get2DContourAndDirectedAnglesOfClouds(clouds_pca_norm_source);
    auto contour_angless_target = get2DContourAndDirectedAnglesOfClouds(clouds_pca_norm_target);
    DEBUG_MSG("[get2DContourAndDirectedAnglesOfClouds] Time elapsed: %s.\n", END_TIMER());

    if(OPTREG::nInfoLevel >= VISUAL_LEVEL){
        // Visualization of boundaries: Verbose level
        for(int i = 0; i<contour_angless_source.size(); i++){
            DEBUG_MSG("Visualize source contour: %d.\n", i);
            CloudUtils::visualizeCloud(contour_angless_source[i].first, 5);
            DEBUG_MSG("Visualize target contour: %d.\n", i);
            CloudUtils::visualizeCloud(contour_angless_target[i].first, 5);
        }
    }

    UPDATE_TIMER();
    auto [pairs_best_match, transs_source] = findCostOfBestMatchUsingAlphaShape(contour_angless_source, contour_angless_target, path_matches);
    DEBUG_MSG("[findCostOfBestMatchUsingAlphaShape] Time elapsed: %s.\n", END_TIMER());
    // Transform clouds: source and target
    
    int num_pairs = pairs_best_match.size();
    std::vector<Eigen::Matrix4f> transs_matching_source(num_pairs), 
                                    transs_matching_target(num_pairs);

    std::ofstream fwriteS_target("./pieces/scales_alphashape_target.txt");
    std::ofstream fwriteS_source("./pieces/scales_alphashape_source.txt");
    for (int i = 0; i < num_pairs; i++){
        int ind_source = pairs_best_match[i].first;
        int ind_target = pairs_best_match[i].second;

        if(OPTREG::nInfoLevel>=DEBUG_LEVEL){
            AlphaShape::savePointsToFile("./contour_source_" + std::to_string(i) + ".txt",
                contour_angless_source[ind_source].first);

            AlphaShape::savePointsToFile("./contour_target_" + std::to_string(i) + ".txt",
                contour_angless_target[ind_target].first);
        }

        Eigen::Matrix4f trans_matching_source = transs_source[i] * getScaleMatrix(scales_source[ind_source]) * transs_pca_source[ind_source];
        Eigen::Matrix4f trans_matching_target = getScaleMatrix(scales_target[ind_target]) * transs_pca_target[ind_target];
        transs_matching_source[i] = trans_matching_source;
        transs_matching_target[i] = trans_matching_target;

        fwriteS_target << i << " " << scales_target[ind_target] << "\n";
        fwriteS_source << i << " " << scales_source[ind_source] << "\n";

    }
    fwriteS_target.close();
    fwriteS_source.close();
    
    return {pairs_best_match, transs_matching_source, transs_matching_target};
}


