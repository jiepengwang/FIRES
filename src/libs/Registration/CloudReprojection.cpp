#include "CloudReprojection.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <pcl/io/ply_io.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d_omp.h>

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

#include "CloudUtils.h"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#include <stdlib.h>
#else
#include <experimental/random>
#endif

#include <set>

#include "IOUtils.h"
#include "AlphaShape.h"
#include "PiecesMatching.h"
#include "LogUtils.h"


CloudReprojection::CloudReprojection()
{
}

CloudReprojection::~CloudReprojection()
{
}


Eigen::MatrixXi increaseMaskArea(const Eigen::MatrixXi& mask, const int& diff_alpha)
{
    int num_rows = mask.rows();
    int num_cols = mask.cols();
    Eigen::MatrixXi mask_ = mask;
    // mask_ = mask;

    // Points2Df points2d;   increased points
    int num_inliers = 0;
    auto comp = [](const Eigen::Vector2f& c1, const Eigen::Vector2f& c2){
        return c1[0] < c2[0] || (c1[0] == c2[0] && c1[1] < c2[1]);};
    std::set<Eigen::Vector2f, decltype(comp)> points2d(comp);
    for(int i = 1; i < num_rows-1; i++)
    {
        for (int j = 1; j < num_cols-1; j++)
        {     
            if(mask(i, j)>diff_alpha){
                num_inliers += 1;
                if(mask(i-1, j-1)<diff_alpha){
                    mask_(i-1, j-1) = 255;
                    points2d.insert(Eigen::Vector2f(i-1, j-1));
                }

                if(mask(i-1, j+1)<diff_alpha){
                    mask_(i-1, j+1) = 255;
                    points2d.insert(Eigen::Vector2f(i-1, j+1));
                }

                if(mask(i+1, j-1)<diff_alpha){
                    mask_(i+1, j-1) = 255;
                    points2d.insert(Eigen::Vector2f(i+1, j-1));
                }

                if(mask(i+1, j+1)<diff_alpha){
                    mask_(i,j) = 255;
                    points2d.insert(Eigen::Vector2f(i+1, j+1));
                }
            }// Points on piece
        }
        
    }
    
    std::cout << "Points2d_increase.size() " << points2d.size() << "\n";
    // CloudUtils::visualizeCloud(points2d, 7);
    std::cout << "num_inliers: " << num_inliers << "\n";
    return mask_;
}



Eigen::MatrixXi increaseMaskArea(const Eigen::MatrixXi& mask, const int& iter_num, const int& diff_alpha)
{
    int num_rows = mask.rows();
    int num_cols = mask.cols();
    Eigen::MatrixXi mask_ = mask;
    // mask_ = mask;

    for (int i = 0; i < iter_num; i++)
    {
        Eigen::MatrixXi tmp_mat = mask_;
        mask_ = increaseMaskArea(tmp_mat, diff_alpha);
    }
    
    // CloudUtils::visualizeCloud(points2d_increase, 7);
    return mask_;
}


int countMaskInliers(const Eigen::MatrixXi& mask, const int& thres_alpha)
{
    int num_rows = mask.rows();
    int num_cols = mask.cols();

    int num_inliers = 0;
    for(int i = 1; i < num_rows-1; i++){
        for (int j = 1; j < num_cols-1; j++) {     
            if(mask(i, j) > thres_alpha)
            {
                num_inliers += 1;
            }
        }
    }
    return num_inliers;
}

void increseMasksAreaParallel(std::map<std::string, Eigen::MatrixXi>& map_masks, const int& iter_num, const int& thres_alpha)
{

    // #pragma omp parallel for 
    for(std::map<std::string, Eigen::MatrixXi>::iterator iter_mask = std::begin(map_masks); 
            iter_mask != std::end(map_masks); 
            iter_mask++)
    {
        auto mask = iter_mask->second;
        int num_inliers_1 = countMaskInliers(mask, thres_alpha);
        mask = increaseMaskArea(mask, iter_num, thres_alpha);  // iter_num:3,  thres_alpha:80
        int num_inliers_2 = countMaskInliers(iter_mask->second, thres_alpha);
        
        std::cout << "For Mask " << iter_mask->first << ", inliers before and after increaing area: "
                    << num_inliers_1 << ", " << num_inliers_2 << ".\n";
    }
    return;
}


void CloudReprojection::segmentCloudPiece(ReprojData& data, 
                                            const std::string& masks_path, 
                                            const std::string& cameras_path)
{
    data.masks = loadImageMasks(masks_path);
    // increseMasksAreaParallel(data.masks, 3, 80);
    data.cameras = readCamerasInfo(cameras_path);
    data.cloud_piece = filterBackgroundPoints(data.cloud, data.masks, data.cameras);
    CloudUtils::savePLYFile("./data/seg.ply", data.cloud_piece);
}


pcl::PointCloud<PointT>::Ptr CloudReprojection::reprojectCloud(const pcl::PointCloud<PointT>::Ptr& cloud, 
                                                                const CameraInfo& camera)
{
    // Transfrorm points from world coordinates to camera coordinates
    pcl::PointCloud<PointT>::Ptr cloud_cam(new pcl::PointCloud<PointT>()),
                                    cloud_image(new pcl::PointCloud<PointT>());

    auto trans_world2cam = composeCameraRC(camera.R, camera.C);
    pcl::transformPointCloud(*cloud, *cloud_cam, trans_world2cam);
    
    // Normalize Z and transform cloud to image plane
    auto cloud_cam_norm = normalizeCloud(cloud_cam);

  
    //visualizeCloud(cloud_cam_norm);
    auto trans_scaleK = composeCameraScaleK(_scale_k, camera.K);
    pcl::transformPointCloud(*cloud_cam_norm, *cloud_image, trans_scaleK);

    // Update depth
    for (int i = 0; i < cloud_image->points.size(); i++){
        cloud_image->points[i].z = cloud_cam->points[i].z;
    }
    
    return cloud_image;
}

Eigen::Vector2f CloudReprojection::reprojectPoint(const Eigen::Vector3f& point, 
                                                                const CameraInfo& camera)
{
    // Transfrorm points from world coordinates to camera coordinates   
    auto trans_world2cam = composeCameraRC(camera.R, camera.C);
    
    Eigen::Matrix4f trans_K = Eigen::Matrix4f::Identity();
    trans_K.block<3,3>(0,0) = camera.K;

    auto trans_reproj_image = trans_K * trans_world2cam;

    Eigen::Vector4f point_reproj = trans_reproj_image * point.homogeneous();
    point_reproj = point_reproj/point_reproj[2];

    Eigen::Vector2f point_reproj_2d = point_reproj.head(2);
    return point_reproj_2d;
}

std::map<std::string, CameraInfo> CloudReprojection::readCamerasInfo(const std::string& filepath)
{
    // Read cameras info
    FILE* frp = fopen(filepath.c_str(), "r");
    if (frp == NULL){
        std::cerr << "Fail to open file " << filepath << ".\n";
        exit(-1);
    }
    
    // Read the number of cameras
    int num_images, image_path_size;
    fread(&(num_images), sizeof(int), 1, frp);
    
    std::vector<char> image_path;
    std::map<std::string, CameraInfo> map_cameras;
    CameraInfo cam_info;
    Eigen::Matrix3d K,R;
    Eigen::Vector3d C;
    for (int i = 0; i < num_images; i++){
		fread(&(image_path_size), sizeof(int), 1, frp);
        image_path.clear();
        image_path.resize(image_path_size);
		
        fread(&(image_path[0]), sizeof(char), image_path_size, frp);
        fread(&(K), sizeof(Eigen::Matrix3d), 1, frp);
		fread(&(R), sizeof(Eigen::Matrix3d), 1, frp);
		fread(&(C), sizeof(Eigen::Vector3d), 1, frp);

        std::string tmppath(image_path.begin(), image_path.end());
        cam_info.image_path = tmppath;
        if (K(2,0) != 0.0){
            if (OPTREG::nInfoLevel>5 && i==0)
                std::cout << "Transpose between cv::Mat and Eigen::Matrix.\n";
            cam_info.K = K.cast<float>().transpose();
            cam_info.R = R.cast<float>().transpose();
        }
        else{
            cam_info.K = K.cast<float>();
            cam_info.R = R.cast<float>();
        } // Transpose between cv::Mat and Eigen::Matrix

        cam_info.C = C.cast<float>();
        if (OPTREG::nInfoLevel>6){
            std::cout << cam_info.image_path << "\n";
            std::cout << "cam_info.K\n" << cam_info.K << "\n";
            std::cout << "cam_info.R\n"<< cam_info.R << "\n";
            std::cout << "cam_info.C\n"<< cam_info.C.transpose() << "\n\n";
        }
        
        boost::filesystem::path bst_path(cam_info.image_path);
        std::string filename = bst_path.stem().string();
        map_cameras.insert(std::pair<std::string, CameraInfo>(filename, cam_info));
	}
    fclose(frp);

    std::cout << "Finish to read file " << filepath << " with " << map_cameras.size() << " camera info.\n";
    return map_cameras;
}

Eigen::Matrix4f CloudReprojection::composeCameraRC(const Eigen::Matrix3f& R, const Eigen::Vector3f& C)
{
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    trans.block<3,3>(0,0) = R;
    trans.block<3,1>(0,3) = R*(-C);

    return trans;
}

Eigen::Matrix4f CloudReprojection::composeCameraScaleK(const double& scale, const Eigen::Matrix3f& K)
{
    Eigen::Matrix4f trans_K = Eigen::Matrix4f::Identity(), trans_scale = Eigen::Matrix4f::Identity();
    trans_K.block<3,3>(0,0) = K;
    trans_scale(0,0) = trans_scale(1,1) = scale;

    auto trans_scaleK = trans_scale * trans_K;
    return trans_scaleK;
}

pcl::PointCloud<PointT>::Ptr CloudReprojection::normalizeCloud(const pcl::PointCloud<PointT>::Ptr& cloud)
{
    int num_points = cloud->points.size();

    pcl::PointCloud<PointT>::Ptr cloud_norm(new pcl::PointCloud<PointT>());
    pcl::copyPointCloud(*cloud,*cloud_norm);
    for (int i = 0; i < num_points; i++){
        // std::cout <<  "Points " << i << " is " << cloud->points[i]. getVector3fMap().transpose() << "\n";
        auto tmp_z = cloud_norm->points[i].z;
        cloud_norm->points[i].x /= tmp_z;
        cloud_norm->points[i].y /= tmp_z;
        cloud_norm->points[i].z  = 1.0;
        if (i==0){
            // VERBOSE_MSG("Depth sample: %d.\n", tmp_z);
        }       
    }

    // visualizeCloud(cloud_norm);
    return cloud_norm;
}


Eigen::MatrixXf CloudReprojection::generateDepthMap(const pcl::PointCloud<PointT>::Ptr &cloud_image)
{
    Eigen::MatrixXf depthmap(_height, _width);
    depthmap.setZero();
    for (int i = 0; i < cloud_image->points.size(); i++){
        auto point = cloud_image->points[i].getVector3fMap();
        int ind_x = point[0], ind_y = point[1];
        VERBOSE_MSG2("Point: %f, %f. %f.\n", point[0], point[1], point[2]);
        VERBOSE_MSG2("Point index: %d, %d.\n", ind_x, ind_y);

        if (depthmap(ind_y, ind_x) == 0){
            depthmap(ind_y, ind_x) = point[2];
        }
        else{
            depthmap(ind_y, ind_x) = std::min(depthmap(ind_y, ind_x), point[2]);
        }
    }
    
    return depthmap;
}

pcl::PointCloud<PointT>::Ptr CloudReprojection::extractCoarseBoundaryByMasks(const pcl::PointCloud<PointT>::Ptr &cloud,
                                                          const std::map<std::string, std::vector<Eigen::Vector2i>> &mask_contours,
                                                          const std::map<std::string, CameraInfo> &cameras)
{
    // Two important paratmeters
    float max_depth_ratio = 0.001;
    float max_square_dist = OPTREG::max_dist_non_boundary;  // max dist to boundary

    std::vector<int> knn_indices;
    std::vector<float> knn_dists;

    std::vector<bool> mask_cloud_trans(cloud->points.size(), true);
    // Iterate all views
    for (auto &mask_contour : mask_contours){
        VERBOSE_MSG("Reproject boundary points to view: %s.\n ", mask_contour.first.c_str());

        KDTree contour_tree(flann::KDTreeSingleIndexParams(15));
        buildKDTree(mask_contour.second, &contour_tree);

        // Find corresponding camerainfo for the mask contour
        auto iter_camerainfo = cameras.find(mask_contour.first);
        if (iter_camerainfo == cameras.end()){
            std::cerr << "Fail to find camera " << mask_contour.first << ".\n";
            continue;
        }

        // Reproject cloud_bound to view plane and iterate all points
        auto cloud_trans = reprojectCloud(cloud, iter_camerainfo->second);
        if (OPTREG::nInfoLevel >= VISUAL_LEVEL){
            visualizeCamCloud(cloud_trans, mask_contour.second, 5);
        }

        auto depthmap = generateDepthMap(cloud_trans);

        // Iterate all points
        int count = 0;
        for (size_t i = 0; i < cloud_trans->points.size(); i++){
            if (mask_cloud_trans[i] == false){
                continue;
            }
            auto point = cloud_trans->points[i].getVector3fMap();
            float depth_cloud = point[2];
            Eigen::Vector2f pointt(point[1], point[0]); // swap x and y to image coordinates

            // Check visibility
            float depth_in_depthmap = depthmap(int(point[1]), int(point[0]));
            float depth_ratio = std::abs(depth_in_depthmap - depth_cloud) / depth_in_depthmap;

            if (depth_in_depthmap <= 0 || depth_cloud <= 0){
                continue; // no depth
            }

            if (depth_ratio > max_depth_ratio){
                // VERBOSE_MSG("Point is Not visible.\n");
                continue;
            }

            // Find nearest points on mask_contour
            searchKDTree(&contour_tree, pointt, knn_indices, knn_dists, 1);
            if (knn_dists[0] > max_square_dist){
                VERBOSE_MSG2("depth_cloud and depth_in_depthmap at (%d, %d): %f,  %f.\n", int(point[1]), int(point[0]), depth_cloud, depth_in_depthmap);
                count++;
                mask_cloud_trans[i] = false;
            }
        } // Iterate all points in cloud_bound

        if (OPTREG::nInfoLevel >= VISUAL_LEVEL){
            // extract visible points and do visualization
            pcl::PointCloud<PointT>::Ptr cloud_temp(new pcl::PointCloud<PointT>());
            for (int i = 0; i < mask_cloud_trans.size(); i++){
                if (mask_cloud_trans[i] == true){
                    cloud_temp->points.push_back(cloud->points[i]);
                }
            }
            INFO_MSG("Msks [%s]. Temporary boundary points: %d.\n", mask_contour.first.c_str(), cloud_temp->points.size());
            CloudUtils::visualizeCloud(cloud_temp); 
        }
    } // Iterate all views

    // Extract cloud
    pcl::PointCloud<PointT>::Ptr cloud_bound_coarse(new pcl::PointCloud<PointT>());
    for (int i = 0; i < mask_cloud_trans.size(); i++){
        if (mask_cloud_trans[i] == true){
            cloud_bound_coarse->points.push_back(cloud->points[i]);
        }
    }

    INFO_MSG("Coarse boundary points: %d.\n.", cloud_bound_coarse->points.size());
    return cloud_bound_coarse;
}

void CloudReprojection::visualizeCloud(const pcl::PointCloud<PointT>::Ptr& cloud, const int point_size)
{ 
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer());
    viewer->addPointCloud<PointT>(cloud, "cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, "cloud");
    
    pcl::PointXYZ origin(0,0,0);
    pcl::PointXYZ axisX(1,0,0), axisY(0,1,0),axisZ(0,0,1);
    viewer->addArrow(axisX, origin, 1.0, 0.0, 0.0, false, "arrow_x");
    viewer->addArrow(axisY, origin, 0.0, 1.0, 0.0, false, "arrow_y");
    viewer->addArrow(axisZ, origin, 0.0, 0.0, 1.0, false, "arrow_z");

    while (!viewer->wasStopped()){
        viewer->spinOnce(50);
    }
}

void CloudReprojection::visualizeCloud(const pcl::PointCloud<PointT>::Ptr& cloud1, 
                                            const pcl::PointCloud<PointT>::Ptr& cloud2, const int point_size)
{ 
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer());
    viewer->addPointCloud<PointT>(cloud1, "cloud1");

    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud2_color(cloud2, 255,0,0);
    viewer->addPointCloud(cloud2, cloud2_color, "cloud2");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, "cloud2");
       
    pcl::PointXYZ origin(0,0,0);
    pcl::PointXYZ axisX(1,0,0), axisY(0,1,0),axisZ(0,0,1);
    viewer->addArrow(axisX, origin, 1.0, 0.0, 0.0, false, "arrow_x");
    viewer->addArrow(axisY, origin, 0.0, 1.0, 0.0, false, "arrow_y");
    viewer->addArrow(axisZ, origin, 0.0, 0.0, 1.0, false, "arrow_z");

    while (!viewer->wasStopped()){
        viewer->spinOnce(50);
    }
}

void CloudReprojection::visualizeCamCloud(const pcl::PointCloud<PointT>::Ptr& cloud, const int point_size)
{ 
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer());
    viewer->addPointCloud<PointT>(cloud, "cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, "cloud");
    
    int height = _height, width = _width;
    pcl::PointXYZ point1(0,0,1), point2(width, 0, 1), point3(width,height,1), point4(0, height, 1);
    viewer->addLine<pcl::PointXYZ>(point1, point2, 255,0,0, "lineID1");
    viewer->addLine<pcl::PointXYZ>(point2, point3, 255,0,0, "lineID2");
    viewer->addLine<pcl::PointXYZ>(point3, point4, 255,0,0, "lineID3");
    viewer->addLine<pcl::PointXYZ>(point4, point1, 255,0,0, "lineID4");

    pcl::PointXYZ origin(0,0,0);
    pcl::PointXYZ axisX(100,0,0), axisY(0,100,0),axisZ(0,0,100);
    viewer->addArrow(axisX, origin, 1.0, 0.0, 0.0, false, "arrow_x");
    viewer->addArrow(axisY, origin, 0.0, 1.0, 0.0, false, "arrow_y");
    viewer->addArrow(axisZ, origin, 0.0, 0.0, 1.0, false, "arrow_z");

    while (!viewer->wasStopped()){
        viewer->spinOnce(50);
    }
}

void CloudReprojection::visualizeCamCloud(const Points2Di& contour, const int point_size)
{ 
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer());

    // Add contour
    pcl::PointCloud<PointT>::Ptr cloud_imcontour(new pcl::PointCloud<PointT>());
    for (size_t i = 0; i < contour.size(); i++){
        float tmp_x = contour[i][0], tmp_y =  contour[i][1];
        PointT point;
        point.y = tmp_x;
        point.x = tmp_y;
        point.z = 1.0;
        cloud_imcontour->push_back(point);
    }
    // std::cout << "cloud_imcontour size is " << cloud_imcontour->points.size() << "\n";
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_imcontour_color(cloud_imcontour, 187,255,255);
    viewer->addPointCloud(cloud_imcontour, cloud_imcontour_color, "cloud_imcontour");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, "cloud_imcontour");
    
    int height = _height, width = _width;
    pcl::PointXYZ point1(0,0,1), point2(width, 0, 1), point3(width,height,1), point4(0, height, 1);
    viewer->addLine<pcl::PointXYZ>(point1, point2, 255,0,0, "lineID1");
    viewer->addLine<pcl::PointXYZ>(point2, point3, 255,0,0, "lineID2");
    viewer->addLine<pcl::PointXYZ>(point3, point4, 255,0,0, "lineID3");
    viewer->addLine<pcl::PointXYZ>(point4, point1, 255,0,0, "lineID4");

    pcl::PointXYZ origin(0,0,0);
    pcl::PointXYZ axisX(100,0,0), axisY(0,100,0),axisZ(0,0,100);
    viewer->addArrow(axisX, origin, 1.0, 0.0, 0.0, false, "arrow_x");
    viewer->addArrow(axisY, origin, 0.0, 1.0, 0.0, false, "arrow_y");
    viewer->addArrow(axisZ, origin, 0.0, 0.0, 1.0, false, "arrow_z");

    while (!viewer->wasStopped()){
        viewer->spinOnce(50);
    }
}

void CloudReprojection::visualizeCamCloud(const pcl::PointCloud<PointT>::Ptr& cloud, const Points2Di& contour, const int point_size)
{ 
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer());
    viewer->addPointCloud<PointT>(cloud, "cloud");

    // Add contour
    pcl::PointCloud<PointT>::Ptr cloud_imcontour(new pcl::PointCloud<PointT>());
    for (size_t i = 0; i < contour.size(); i++){
        float tmp_x = contour[i][0], tmp_y =  contour[i][1];
        PointT point;
        point.y = tmp_x;
        point.x = tmp_y;
        point.z = 1.0;
        cloud_imcontour->push_back(point);
    }
    std::cout << "cloud_imcontour size is " << cloud_imcontour->points.size() << "\n";
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_imcontour_color(cloud_imcontour, 187,255,255);
    viewer->addPointCloud(cloud_imcontour, cloud_imcontour_color, "cloud_imcontour");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, "cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, "cloud_imcontour");
    
    int height = _height, width = _width;
    pcl::PointXYZ point1(0,0,1), point2(width, 0, 1), point3(width,height,1), point4(0, height, 1);
    viewer->addLine<pcl::PointXYZ>(point1, point2, 255,0,0, "lineID1");
    viewer->addLine<pcl::PointXYZ>(point2, point3, 255,0,0, "lineID2");
    viewer->addLine<pcl::PointXYZ>(point3, point4, 255,0,0, "lineID3");
    viewer->addLine<pcl::PointXYZ>(point4, point1, 255,0,0, "lineID4");

    pcl::PointXYZ origin(0,0,0);
    pcl::PointXYZ axisX(100,0,0), axisY(0,100,0),axisZ(0,0,100);
    viewer->addArrow(axisX, origin, 1.0, 0.0, 0.0, false, "arrow_x");
    viewer->addArrow(axisY, origin, 0.0, 1.0, 0.0, false, "arrow_y");
    viewer->addArrow(axisZ, origin, 0.0, 0.0, 1.0, false, "arrow_z");

    while (!viewer->wasStopped()){
        viewer->spinOnce(50);
    }
}


std::vector<int> CloudReprojection::filterPoints(const pcl::PointCloud<PointT>::Ptr& cloud, const Eigen::MatrixXi& mask)
{
    // Filter points in one view
    int num_points = cloud->points.size();
    int num_rows = mask.rows();
    int num_cols = mask.cols();
    int diff_alpha = 20;  //125
    
    //TODO: Scale cloud_image
    _scale_cloud = (float)mask.cols()/(float)_width;
    auto cloud_scale = CloudUtils::scaleCloud(cloud, _scale_cloud);
    // visualizeCamCloud(cloud_scale);

    // Get indices
    std::vector<int> index_inliers;
    bool b_half_mask = true;
    try{
        for(int i = 0; i < num_points; i++){
            auto point = cloud_scale->points[i];
            int index_y = (int)(round(point.x));
            int index_x = (int)(round(point.y));
            if(index_x >= 0 && index_y >= 0){
                if(index_x < num_rows && index_y < num_cols){
                    if(mask(index_x, index_y) > diff_alpha){
                        index_inliers.push_back(i);
                    }        
                }// Index inside image 
            }// Index lager than 0
        } // Points iteration
    }
    catch(const std::exception& e){
        std::cerr << e.what() << '\n';
    }

    return index_inliers;
}


pcl::PointCloud<PointT>::Ptr CloudReprojection::filterBackgroundPoints(const pcl::PointCloud<PointT>::Ptr& cloud, const std::map<std::string, Eigen::MatrixXi>& masks,
                                                const std::map<std::string, CameraInfo>& cameras)
{
    pcl::PointCloud<PointT>::Ptr cloud_piece(new pcl::PointCloud<PointT>());
    pcl::copyPointCloud(*cloud, *cloud_piece);
    
    CloudUtils::visualizeCloud(cloud_piece, 5);
    for (auto& mask : masks){
        auto iter_camerainfo = cameras.find(mask.first);
        if (iter_camerainfo == cameras.end()){
            std::cerr << "Fail to find camera " << mask.first << "\n";
            continue;
        }

        auto cloud_image = reprojectCloud(cloud_piece, iter_camerainfo->second);
        auto piece_indices = filterPoints(cloud_image, mask.second);
        
        // Extract points
        pcl::PointCloud<PointT>::Ptr cloud_inliers(new pcl::PointCloud<PointT>()),
                                        cloud_outliers(new pcl::PointCloud<PointT>());
        for (auto& elem : piece_indices){
            auto point = cloud_piece->points[elem];
            cloud_inliers->points.push_back(point);
        }
        // CloudUtils::visualizeCloud(cloud_inliers, 3);
        
        cloud_piece.swap(cloud_inliers);
        cloud_inliers->clear();
        std::cout <<"The number of points in cloud_piece now is " << cloud_piece->points.size() 
                    << " using image " << mask.first << ".\n";
        if(cloud_piece->points.size() == 0){
            std::cout << "There are no points in filtered cloud.\n" 
                        << "Image resolution may be wrong.\n";
            exit(-1);
        }

    }
    std::cout <<"\n[Cloud boundary] The extracted number of points in cloud_piece is " << cloud_piece->points.size() << ".\n";
    CloudUtils::visualizeCloud(cloud_piece, 3);
    return cloud_piece;
}
// Image related 

std::map<std::string, Eigen::MatrixXi>  CloudReprojection::loadImageMasks(const std::string& masks_folder, const int& step_sample, const bool& bSample)
{
    auto path_masks = IOUtils::getFileListInDirectory(masks_folder);
    std::map<std::string, Eigen::MatrixXi> map_masks;

    INFO_MSG("Step when loading masks: %d.\n", step_sample);
    
    START_TIMER();
    for (int i = 0; i < path_masks.size(); (i = i + step_sample)){
        std::string path_mask = path_masks[i];
        auto [ppath, stem, ext] = IOUtils::getFileParentPathStemExt(path_mask);
        if(ext == ".png"){
            auto mask = loadImageMask(path_mask);
            // auto mask_increase = increaseMaskArea(mask, 4, 80);
            map_masks.insert(std::pair<std::string, Eigen::MatrixXi>(stem, mask));
        }
    }
    
    ASSERT(map_masks.size() != 0);
    INFO_MSG("Masks loaded: %d.(%s)\n.", map_masks.size(), END_TIMER());
    return map_masks;
}


Eigen::MatrixXi CloudReprojection::loadImageMask(const std::string& pngfilepath)
{
    cv::Mat img, alpha_channel, scale_al;
    img = cv::imread(pngfilepath, cv::IMREAD_UNCHANGED);
    cv::extractChannel(img, alpha_channel, img.channels()-1);
    
    // Sample mask
    float scale = float(alpha_channel.cols)/ float(_width);
    cv::Size size_raw = alpha_channel.size();
    cv::Size size_sample(size_raw.width/scale, size_raw.height/scale);
    cv::resize(alpha_channel, alpha_channel, size_sample, cv::INTER_LINEAR);

    Eigen::MatrixXi mask;
    cv::cv2eigen(alpha_channel, mask);
    VERBOSE_MSG("Scale of mask [%s]: %.03f. Mask size: %d*%d.\n", pngfilepath.c_str(), 
                                    scale, mask.cols(), mask.rows());

    return mask;
}

std::vector<Eigen::Vector2i> CloudReprojection::extractMaskContour(const Eigen::MatrixXi& mask)
{
    int num_rows = mask.rows();
    int num_cols = mask.cols();
    int diff_alpha = 60;

    int num_pixels = _width * 4;
    std::vector<Eigen::Vector2i> points_contour;
    points_contour.reserve(num_pixels);
    START_TIMER();
    try{
        for(int i = 1; i < num_rows-1; i++){
            for (int j = 1; j < num_cols-1; j++){
                if(mask(i, j)>diff_alpha){
                    if(mask(i-1, j-1)<diff_alpha  || 
                        mask(i-1, j+1)<diff_alpha ||
                        mask(i-1, j+1)<diff_alpha ||
                        mask(i+1, j+1)<diff_alpha)
                    {
                        points_contour.push_back(Eigen::Vector2i(i,j));
                    }
                }// Points on piece
            }// Iteration on colums
        } // Iteration on rows
    }
    catch(const std::exception& e){
        std::cerr << e.what() << '\n' ;
    }

    VERBOSE_MSG("The number of pixels on contour: %d. (%s)\n", points_contour.size(), END_TIMER());

    return points_contour;
}



void CloudReprojection::extractMaskContours( const std::map<std::string, Eigen::MatrixXi>& masks, std::map<std::string, std::vector<Eigen::Vector2i>>& contours)
{
    contours.clear();
    START_TIMER();
    for (auto & mask : masks){
        auto contour = extractMaskContour(mask.second);
        // visualizeCamCloud(contour);
        contours.insert(std::pair<std::string, std::vector<Eigen::Vector2i>>(mask.first, contour));
    }
    INFO_MSG("Finish extract mask contours.(%s)\n", END_TIMER());;
}



// Image to cloud
pcl::PointCloud<PointT>::Ptr CloudReprojection::transformCloudFromImageToWorld(const pcl::PointCloud<PointT>::Ptr& cloud_image, 
                                                const CameraInfo& cam_info,
                                                const Eigen::Vector3f& n, const float& d)
{
    pcl::PointCloud<PointT>::Ptr cloud_norm(new pcl::PointCloud<PointT>());
    
    // transform points from image coordinates to cam coordinates
    Eigen::Matrix4f K_homo_inv = Eigen::Matrix4f::Identity();
    Eigen::Matrix2f K_R = cam_info.K.block<2,2>(0,0);
    Eigen::Vector2f K_t = cam_info.K.block<2,1>(0,2);
    K_homo_inv.block<2,2>(0,0) = K_R.inverse();
    K_homo_inv.block<2,1>(0,3) = -K_R.inverse()*K_t;
    pcl::transformPointCloudWithNormals(*cloud_image, *cloud_norm, K_homo_inv);

    // transform points from cam coordinates to world coordinates
    int num_points = cloud_norm->points.size(); 
    float s_num = n.transpose()*cam_info.C + d;
    auto s_den = n.transpose()*cam_info.R.transpose();
    
    // Get reference of points data
    auto points_xyzrgbn = cloud_norm->getMatrixXfMap();
    
    Eigen::MatrixXf points_xyz = points_xyzrgbn.block(0,0,3, points_xyzrgbn.cols());
    Eigen::VectorXf scales = -s_num * (s_den*points_xyz).cwiseInverse().transpose();
    Eigen::MatrixXf points_rot_scale = cam_info.R.transpose()*points_xyz* scales.asDiagonal();
    Eigen::MatrixXf points_xyz_world = (points_rot_scale).colwise()+cam_info.C;

    // update points in cloud
    points_xyzrgbn.block(0,0,3, points_xyzrgbn.cols()) = points_xyz_world;
    
    visualizeCloud(cloud_norm);
    return cloud_norm;
}