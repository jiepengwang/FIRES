#pragma once
/*
some common functions when using pcl for point clouds
*/
#include <tuple>
#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


typedef pcl::PointXYZRGBNormal PointT;
typedef std::vector<Eigen::Vector3f> Points3Df;
typedef std::vector<Eigen::Vector2f> Points2Df;
typedef std::vector<pcl::PointCloud<PointT>::Ptr> CloudsVec;




// #define EIGEN_MAX_STATIC_ALIGN_BYTES 0

namespace CloudUtils
{
    // IO: load and save
    pcl::PointCloud<PointT>::Ptr loadPLYFile(const std::string& file_path);

    void loadPLYFile(const std::string& file_path, pcl::PointCloud<PointT>::Ptr& cloud);
    std::vector<pcl::PointCloud<PointT>::Ptr> loadPLYFiles(const std::vector<std::string>& path_files);
    std::vector<pcl::PointCloud<PointT>::Ptr> loadPLYFilesInDirectory(const std::string& path_dir);

    void removeNaNPoints(pcl::PointCloud<PointT>::Ptr cloud);

    void savePLYFile(const std::string& file_path, const pcl::PointCloud<PointT>::Ptr& cloud);
    void savePLYFile(const std::string& file_path, const Points3Df& points);


    // PCA
    std::tuple<pcl::PointCloud<PointT>::Ptr, Eigen::Matrix4f>  
            calculatePCA(const pcl::PointCloud<PointT>::Ptr& cloud);
    std::tuple<pcl::PointCloud<PointT>::Ptr, pcl::PointCloud<PointT>::Ptr> 
            alignCloudsPCA(const pcl::PointCloud<PointT>::Ptr& cloud_source, 
                                const pcl::PointCloud<PointT>::Ptr& cloud_target);
    float alignCloudsScale(const pcl::PointCloud<PointT>::Ptr& cloud_source, 
                            const pcl::PointCloud<PointT>::Ptr& cloud_target);
    Eigen::Matrix4f alignCloudsAlphaShape(pcl::PointCloud<PointT>::Ptr& cloud_source, 
                                                pcl::PointCloud<PointT>::Ptr& cloud_target);
    std::tuple<pcl::PointCloud<PointT>::Ptr, pcl::PointCloud<PointT>::Ptr>  
            alignClouds(const pcl::PointCloud<PointT>::Ptr& cloud_source, 
                            const pcl::PointCloud<PointT>::Ptr& cloud_target);
    void alignClouds(const std::string& path_reg, const std::string& path_gt);
    

    // Transform cloud
    float normalizeCloud(const pcl::PointCloud<PointT>::Ptr& cloud, const bool& bShortEdge = false);
    pcl::PointCloud<PointT>::Ptr scaleCloud(const pcl::PointCloud<PointT>::Ptr& cloud, 
                                                const float& scale);
    

    // Sampling
    pcl::PointCloud<PointT>::Ptr uniformSampling(const pcl::PointCloud<PointT>::Ptr& cloud, const float& voxel_size);
    pcl::PointCloud<PointT>::Ptr randomSampling(const pcl::PointCloud<PointT>::Ptr& cloud, const int& num_sample);
    float estimateVoxelSize(const pcl::PointCloud<PointT>::Ptr& cloud, const float& fNormVoxel, const int& num_rand = 30000);
    pcl::PointCloud<PointT>::Ptr statisticalOutlierRemoval(const pcl::PointCloud<PointT>::Ptr& cloud, const int& num_neighbor, const float& std_dev);
    
    void transformPoints3D(Points3Df& points, const Eigen::Matrix4f& trans);
    void transformPoints2D(Points2Df& points, const Eigen::Matrix4f& trans);

    void transformCloud(const pcl::PointCloud<PointT>::Ptr& cloud, 
                          const pcl::PointCloud<PointT>::Ptr& cloud_trans, 
                          const Eigen::Matrix4f& trans);
    pcl::PointCloud<PointT>::Ptr transformCloud(const pcl::PointCloud<PointT>::Ptr& cloud, const Eigen::Matrix4f& trans);



    // tools
    Eigen::Vector3f getCloudBoundingBox(const pcl::PointCloud<PointT>::Ptr& cloud);
    Eigen::Vector3f getCloudBoundingBox(const Points3Df& points);

    pcl::PointCloud<PointT>::Ptr convertPointsToCloud(const Points3Df& points);
    Points3Df convertCloudToPoints(const pcl::PointCloud<PointT>::Ptr& cloud);
    Points2Df convertCloudToPoints2D(const pcl::PointCloud<PointT>::Ptr& cloud);
    Points3Df convertPoints2DToHomo3D(const Points2Df& points2d);

    void convertCloudToPointsAndNormals3D(pcl::PointCloud<PointT>::Ptr cloud, Points3Df& points, Points3Df& normals);


    //File name
    std::string extendFileNameInPath(const std::string& path, const std::string& str_expand);


   
    std::tuple<pcl::PointCloud<PointT>::Ptr,pcl::PointCloud<PointT>::Ptr>
    filterCloudByColor(const pcl::PointCloud<PointT>::Ptr& cloud, const Eigen::Vector3i& color_filter_kernel);

    pcl::PointCloud<PointT>::Ptr filterCloudByRadiusOutlierRemoval(const pcl::PointCloud<PointT>::Ptr& cloud, const double& search_radius, const int& num_neighbor);

    
    // Visualization
    void visualizeCloud(pcl::PointCloud<PointT>::Ptr cloud, const double& pointsize = 1.0);
    void visualizeClouds(pcl::PointCloud<PointT>::Ptr cloud1, 
                            pcl::PointCloud<PointT>::Ptr cloud2, const double& pointsize_cloud2 = 1.0);
    

    void visualizeCorrespondences(const pcl::PointCloud<PointT>::Ptr cloud_source, const pcl::PointCloud<PointT>::Ptr cloud_target, 
                                        const std::vector<std::pair<int, int>>& matches, const double& matches_sample_rate = 1.0);
    void sortVector3D(const Points3Df& vec,  Points3Df& sorted_vec,  Eigen::VectorXi& ind);
    
    // Eigen data structure
    void visualizeCloud(const Points3Df& points, const double& pointsize = 1.0);
    void visualizeCloud(const Points2Df& points2d, const double& pointsize = 1.0);
    void visualizeClouds(const Points3Df& points1, const Points3Df& points2, const double& pointsize_points2 = 1.0);
    void visualizeClouds(const Points2Df& points1, const Points2Df& points2, const double& pointsize_points2 = 1.0);

    void visualizeClouds(const pcl::PointCloud<PointT>::Ptr cloud, const Points3Df& points3d, const double& pointsize_points3d = 1.0);

    void visualizeCorrespondences(const Points3Df& points_source, const Points3Df& points_target, 
                                        const std::vector<std::pair<int, int>>& matches, const double& matches_sample_rate = 1.0);
    
    void writeTransformMatrix(const char* filepath, const Eigen::Matrix4f& transtemp);
    Eigen::Matrix4f readTransformMatrix(const std::string& filepath);
    Eigen::Matrix4f readFGRTransformMatrix(const std::string& filepath);

    
    pcl::PointCloud<PointT>::Ptr refineBoundary(const pcl::PointCloud<PointT>::Ptr &cloud, const pcl::PointCloud<PointT>::Ptr &cloud_bound, const int& num_neighbors);
    pcl::PointCloud<pcl::PointXYZ>::Ptr extractPointsByIndices(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const std::vector<int> &indices);
    std::tuple<pcl::PointCloud<pcl::PointXYZ>::Ptr, Eigen::Matrix4f> calculatePCA(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
    bool isBoundaryPoint(Eigen::Vector2f q_point, const float& max_diff_angle, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_neighbor_pca);
    pcl::PointCloud<PointT>::Ptr refineBoundary(const pcl::PointCloud<PointT>::Ptr &cloud, const pcl::PointCloud<PointT>::Ptr &cloud_bound, const int& num_neighbors);
             
};
