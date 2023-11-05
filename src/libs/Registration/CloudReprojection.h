#pragma once   
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#undef max
#undef min
#endif

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>

#include <vector>
#include <map>
#include <tuple>

#include "PointsReg.h"
#include "arguments.h"

#include <opencv2/opencv.hpp>

#define DEBUG 1

typedef pcl::PointXYZRGBNormal PointT;
//typedef Eigen::Matrix<uint8_t,Eigen::Dynamic,Eigen::Dynamic>;

struct CameraInfo{
    std::string image_path;
    Eigen::Matrix3f K; // the intrinsic camera parameters (3x3)
    Eigen::Matrix3f R; // rotation (3x3) and
    Eigen::Vector3f C; // translation (3,1), the extrinsic camera parameters
};

struct ReprojData{
    std::map<std::string, CameraInfo> cameras;
    std::map<std::string, Eigen::MatrixXi> masks;
    std::map<std::string, std::vector<Eigen::Vector2i> > mask_contours;

    pcl::PointCloud<PointT>::Ptr cloud;
    pcl::PointCloud<PointT>::Ptr cloud_piece;
    pcl::PointCloud<PointT>::Ptr cloud_pclbound;
    pcl::PointCloud<PointT>::Ptr cloud_pclbound_refine;
};

class CloudReprojection : public PointsReg
{
private: 
    int _width, _height;
    float _scale_k = 1.0;
    

    ReprojData _data_source, _data_target;

public:
    CloudReprojection(/* args */);
    ~CloudReprojection();

    CloudReprojection(const int& height, const int& width):
    _height(height),
    _width(width){}

    CloudReprojection(const ReprojArgs& reproj_args, 
                        const pcl::PointCloud<PointT>::Ptr& cloud_source, 
                        const pcl::PointCloud<PointT>::Ptr& cloud_target, 
                        const int& height, const int& width):
    _reproj_args(reproj_args),
    _height(height),
    _width(width)
    {
        std::cout << "[CloudReprojection] Initialization...\n";

        // Target cloud
        _data_target.cloud = cloud_target;
        std::cout << "The number of _cloud_target points is " 
                    << _data_target.cloud->points.size() << ".\n";
        segmentCloudPiece(_data_target, _reproj_args.masks_path_target, _reproj_args.cameras_path_target);     

        std::cout << "Finish to initialize CloudReprojection.\n";
    }
    
    // data
    ReprojArgs _reproj_args;
    float _scale_cloud = 1.0;
    // Cloud-related
    void segmentCloudPiece(ReprojData& data, const std::string& masks_path, 
                                                const std::string& cameras_path);
    
    pcl::PointCloud<PointT>::Ptr reprojectCloud(const pcl::PointCloud<PointT>::Ptr& cloud, 
                                                    const CameraInfo& camera);
    Eigen::Vector2f reprojectPoint(const Eigen::Vector3f& point, 
                                                                const CameraInfo& camera);
    pcl::PointCloud<PointT>::Ptr normalizeCloud(const pcl::PointCloud<PointT>::Ptr& cloud);  
    
    // Extract boundary

    Eigen::MatrixXf generateDepthMap(const pcl::PointCloud<PointT>::Ptr &cloud_image);
    pcl::PointCloud<PointT>::Ptr extractCoarseBoundaryByMasks(const pcl::PointCloud<PointT>::Ptr &cloud,
                                                          const std::map<std::string, std::vector<Eigen::Vector2i>> &mask_contours,
                                                          const std::map<std::string, CameraInfo> &cameras);
                                                          
    void visualizeCloud(const pcl::PointCloud<PointT>::Ptr& cloud, const int point_size = 1);
    void visualizeCloud(const pcl::PointCloud<PointT>::Ptr& cloud1, 
                            const pcl::PointCloud<PointT>::Ptr& cloud2,const int point_size = 1);
    void visualizeCamCloud(const pcl::PointCloud<PointT>::Ptr& cloud, const int point_size = 1);
    void visualizeCamCloud(const Points2Di& contour, const int point_size = 3);
    void visualizeCamCloud(const pcl::PointCloud<PointT>::Ptr& cloud, 
                                const Points2Di& contour, const int point_size = 3);

    // Camera-related
    std::map<std::string, CameraInfo> readCamerasInfo(const std::string& filepath);
    Eigen::Matrix4f composeCameraScaleK(const double& scale, const Eigen::Matrix3f& K);
    Eigen::Matrix4f composeCameraRC(const Eigen::Matrix3f& R, const Eigen::Vector3f& C);
    
    // Image-related
    std::map<std::string, Eigen::MatrixXi> loadImageMasks(const std::string& masks_folder, const int& step_sample = 2, const bool& bSample = false);
    Eigen::MatrixXi loadImageMask(const std::string& pngfilepath);
    
    // Refine boundary
    std::vector<Eigen::Vector2i> extractMaskContour(const Eigen::MatrixXi& mask);
    void extractMaskContours(const std::map<std::string, Eigen::MatrixXi>& masks,
                                std::map<std::string, std::vector<Eigen::Vector2i>>& contours);


    // Version 1: one by one
    std::vector<int> filterPoints(const pcl::PointCloud<PointT>::Ptr& cloud, const Eigen::MatrixXi& mask);
    pcl::PointCloud<PointT>::Ptr filterBackgroundPoints(const pcl::PointCloud<PointT>::Ptr& cloud, 
                                                            const std::map<std::string, Eigen::MatrixXi>& masks,
                                                            const std::map<std::string, CameraInfo>& cameras);
       
    
    // Image to Cloud

    pcl::PointCloud<PointT>::Ptr transformCloudFromImageToWorld(const pcl::PointCloud<PointT>::Ptr& cloud_image, 
                                                        const CameraInfo& cam_info,
                                                        const Eigen::Vector3f& n, const float& d);


    bool segmentPiecesWithRegionGrowing(const std::string& path_file);

    

    std::vector<pcl::PointCloud<PointT>::Ptr> loadPLYFilesInDirectory(const std::vector<std::string>& path_files);

};