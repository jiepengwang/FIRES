#pragma once
#include <Eigen/Core>
#include <vector>
#include <tuple>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

typedef std::vector<Eigen::Vector2f> Points2D;
typedef pcl::PointXYZRGBNormal  PointT;

namespace PiecesMatching
{
    // Segment the raw cloud to different pieces: top and bottom
    std::tuple<std::vector<int>, int> segmentPiecesWithRegionGrowing(const pcl::PointCloud<PointT>::Ptr& cloud, const float& search_radius = 6);
    bool segmentPieceIDByRegionGrowing(const pcl::PointCloud<PointT>::Ptr& cloud, std::vector<int>& mask_cloud, 
                                        const int& piece_id, const float& search_radius);
    
    pcl::PointCloud<PointT>::Ptr extractPieceID(const pcl::PointCloud<PointT>::Ptr& cloud, const std::vector<int>& mask_cloud, const int& piece_id);
    std::vector<pcl::PointCloud<PointT>::Ptr> extractPiecesByID(const pcl::PointCloud<PointT>::Ptr& cloud, const std::vector<int>& mask_cloud, const int& num_pieces);
    bool saveExtractedPieces(const std::vector<pcl::PointCloud<PointT>::Ptr>& cloud_pieces, const std::string& path_dir_save);
    


    // Find matches of pieces in two sides;
    std::tuple<std::vector<std::pair<int, int>>, std::vector<Eigen::Matrix4f>>
    findCostOfBestMatchUsingAlphaShape(const std::vector<std::pair<Points2D, Eigen::VectorXf>>& contour_angless_source,
                                        const std::vector<std::pair<Points2D, Eigen::VectorXf>>& contour_angless_target,
                                        const std::string& path_matches = "empty",
                                        const bool& b_only_one_side = false);

    std::vector<std::pair<Points2D, Eigen::VectorXf>> get2DContourAndDirectedAnglesOfClouds(const std::vector<pcl::PointCloud<PointT>::Ptr>& clouds);

    void saveMatchedPieces(const std::vector<pcl::PointCloud<PointT>::Ptr>& clouds_pieces_source, 
                                        const std::vector<pcl::PointCloud<PointT>::Ptr>& clouds_pieces_target,
                                        const std::vector<std::pair<int, int>>& pairs,
                                        const std::string& path_dir_save);

    std::vector<pcl::PointCloud<PointT>::Ptr> normalizeCloudsAfterPCA(const std::vector<pcl::PointCloud<PointT>::Ptr>& clouds);
    std::tuple<std::vector<pcl::PointCloud<PointT>::Ptr>,  
                    std::vector<Eigen::Matrix4f>,
                    std::vector<float>>
    normalizeCloudsAfterPCA2(const std::vector<pcl::PointCloud<PointT>::Ptr>& clouds, const std::string& side);

    
    

    std::tuple<std::vector<std::pair<int, int>> , std::vector<Eigen::Matrix4f>, std::vector<Eigen::Matrix4f> >
    findMatchesInRawPieces(const std::vector<pcl::shared_ptr<pcl::PointCloud<PointT>>>& clouds_source, 
                            const std::vector<pcl::shared_ptr<pcl::PointCloud<PointT>>>& clouds_target,
                            const std::string& path_matches = "empty",
                            const bool& b_only_one_side = false);


    void test();
}