#pragma once
#include <vector> 
#include <tuple>
#include <Eigen/Core>
#include <flann/flann.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "AlphaShape.h"
#include "ICPReg.h"
#include "arguments.h"


typedef pcl::PointXYZRGBNormal PointT;

struct RegClouds{
    pcl::PointCloud<PointT>::Ptr piece;
    pcl::PointCloud<PointT>::Ptr boundary;
    pcl::PointCloud<PointT>::Ptr contour;  // points on the overlap area
    // pcl::PointCloud<PointT>::Ptr mvs_ouput;

    Eigen::Matrix4f trans_global;       // Global registration: PCA + AlphaShape
    Eigen::Matrix4f trans_scale_source; // Align scale of two clouds
    Eigen::Matrix4f trans_local;        // Local refinement: ICP
};


class PCReg
{
private:
    ICPRegArgs _reg_args;    

    RegClouds _clouds_source, _clouds_target;
    RegPoints3Ds _points3Ds_source, _points3Ds_target;


public:

    PCReg(const ICPRegArgs& reg_args, const int& a = 5){
        _reg_args = reg_args;
    }
    ~PCReg(){}

    std::string _path_clouds;   // path of cloud and boundary
    pcl::PointCloud<PointT>::Ptr cloud_reg;
    bool bUseScale = true;

    // Point cloud registration
    void registerUsingICPWithBoundaryConstraint();  


    // Load file and calculate FPFH feature
    
    void loadClouds();
    void loadRegClouds(RegClouds& clouds, const std::string& path_dir, const std::string& side);

    void alignClouds();
    void scaleClouds(RegClouds& clouds, const float& scale);
    
    
    // for matching
    template<class T>
	void buildKDTree(const std::vector<T>& data, KDTree* tree);
	template<class T>
	void searchKDTree(KDTree* tree, const T& input,
		                std::vector<int>& indices, std::vector<float>& dists, int nn);

    // Data type conversion and save
    Points3D convertCloudToPoints3D(const pcl::PointCloud<PointT>::Ptr& cloud);
    void convertCloudToPoints3D(pcl::PointCloud<PointT>::Ptr cloud, Points3D& points);
    void convertCloudToNormals3D(pcl::PointCloud<PointT>::Ptr cloud, Points3D& normals);
    void convertCloudToPointsAndNormals3D(pcl::PointCloud<PointT>::Ptr cloud, Points3D& points, Points3D& normals);
    void convertCloudsToPoints3Ds(RegClouds& clouds, RegPoints3Ds & points3Ds);
    
   
    // points extraction
    std::vector<Eigen::Vector2f> extract2DContourByAlphaShape(pcl::PointCloud<PointT>::Ptr& cloud);
};

