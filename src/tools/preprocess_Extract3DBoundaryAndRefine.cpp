#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/io.h>
#include <pcl/features/boundary.h>

#include "CloudUtils.h"
#include "PointsUtils.h"
#include "PiecesMatching.h"
#include "IOUtils.h"
#include "CloudReprojection.h"
#include "LogUtils.h"

typedef pcl::PointXYZRGBNormal PointT;
typedef std::map<std::string, CameraInfo> CamInfos;
typedef std::map<std::string, Eigen::Vector3f> CamCenters;

CamCenters getCamCenters(const CamInfos& cameras)
{
    // Explicit camera centers in world coordinates
    int num_cams = cameras.size();
    CamCenters cam_centers;
    for(auto& cam : cameras){
        cam_centers.insert(std::make_pair(cam.first, cam.second.C));
        DEBUG_MSG("Cam center: %s.\n", TO_CSTR(cam.second.C.transpose()));
    }
    
    INFO_MSG("Cameras number: %d.\n", cam_centers.size());
    return cam_centers;
}

std::string getNearestVisibleCamCenter(const PointT& point_query, const CamCenters& cam_centers)
{
    int num_cams = cam_centers.size();
    const Eigen::Vector3f normal = point_query.getNormalVector3fMap();
    const Eigen::Vector3f position = point_query.getVector3fMap();

    Eigen::VectorXf dists(num_cams);
    std::vector<std::string> names(num_cams);
    int i = 0;
    for (auto& cam_center_t: cam_centers){

        Eigen::Vector3f ray_dir = cam_center_t.second - position;

        //Check visibility
        if(ray_dir.dot(normal)<=0){
            dists[i] = INFINITY;
            names[i]=cam_center_t.first;
            i++;
            continue;
        }
        
        //Calculate dist
        float dist = ray_dir.cross(normal).norm();
        dists[i] = dist;   
        names[i]=cam_center_t.first;
        i++;   
    }

    // Find nearest cam
    Eigen::Index index_min;
    float dist_min = dists.minCoeff(&index_min);

    return names[index_min];
}

float  findNearestMaskContourPointDist(const Eigen::Vector3f& point_query, const CameraInfo& cam_info, const Points2Di& mask_contour)
{
    CloudReprojection reproj;
    auto point_query_reproj = reproj.reprojectPoint(point_query, cam_info);


    KDTree tree_contour(flann::KDTreeSingleIndexParams(15));
    reproj.buildKDTree(mask_contour, &tree_contour);

    std::vector<int> knn_indices;
    std::vector<float> knn_dists;
    reproj.searchKDTree(&tree_contour, point_query_reproj, knn_indices, knn_dists, 1);

    return knn_dists[0];
}

 pcl::PointCloud<PointT>::Ptr refinePCLBoundaryPoints(const ReprojData& data, const CamCenters& cam_centers, const float& max_dist_pixel)
{
    std::vector<int> indices_bound_refine;
    int num_points = data.cloud_pclbound->points.size();
    for (int i = 0; i < num_points; i++){
        auto point = data.cloud_pclbound->points[i];
        auto cam_nn_name = getNearestVisibleCamCenter(point, cam_centers);

        auto cam_info = data.cameras.find(cam_nn_name)->second;
        auto cam_nn_mask = data.mask_contours.find(cam_nn_name)->second;

        
        float dist = findNearestMaskContourPointDist(point.getArray3fMap(), cam_info, cam_nn_mask);
        if(dist<max_dist_pixel){
            // Refined boundary point
            indices_bound_refine.push_back(i);
        }
    }
    
    INFO_MSG("Refined boundary points: %d.\n", indices_bound_refine.size());
    //Extract refined boundary points
    pcl::PointCloud<PointT>::Ptr cloud_bound_refine(new pcl::PointCloud<PointT>);

    for (int i = 0; i < indices_bound_refine.size(); i++){
        int ind_point = indices_bound_refine[i];
        cloud_bound_refine->points.push_back(data.cloud_pclbound->points[ind_point]);
    }
    // CloudUtils::visualizeClouds(data.cloud_pclbound, cloud_bound_refine, 3);
    return cloud_bound_refine;       
}

pcl::PointCloud<PointT>::Ptr smoothCloudNormal(const pcl::PointCloud<PointT>::Ptr& cloud, const bool& bSmoothNormal = true, const bool& bSmoothGeo = false)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<PointT>::Ptr cloud_smooth(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*cloud, *cloud_xyz);
    pcl::copyPointCloud(*cloud, *cloud_smooth);

    
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud_xyz);

    // Neighbors within radius search
    int num_neighbors = 36;
    #pragma omp parallel for 
    for (int i = 0; i < cloud->points.size(); i++)
    {
        std::vector<int> indices_knn;
        std::vector<float> dists_knn;

        auto point_xyz = cloud_xyz->points[i];
        if( kdtree.nearestKSearch(point_xyz, num_neighbors, indices_knn, dists_knn) > 0 ){
            //Smooth cloud normal
            Eigen::Vector3f normal_ave(0), geo_ave(0);
            for (int j = 0; j < num_neighbors; j++){
                int index_cloud_point = indices_knn[j];
                normal_ave += cloud->points[index_cloud_point].getNormalVector3fMap();
            }
            normal_ave /= float(num_neighbors);
            auto normal_origin = cloud->points[i].getNormalVector3fMap().transpose();
            VERBOSE_MSG2("Origin normal: %s; smoothed normal, %s.\n", TO_CSTR(normal_origin), TO_CSTR(normal_ave.transpose()));

            cloud_smooth->points[i].getNormalVector3fMap() = normal_ave;
        }
}

    return cloud_smooth;
}


int main(int argc, char** argv)
{
    std::string path_dir_clouds, 
                str_file_search;   // Top or bottom
    int nInfoLevel;
    float image_width = -1, image_height = -1;
    std::string bCoarse = "False";
    if( argc-1 >= 5){
        path_dir_clouds = argv[1];
        str_file_search = argv[2]; 
        image_width = std::stof(argv[3]);
        image_height = std::stof(argv[4]);
        nInfoLevel = std::stoi(argv[5]);
        if(argc-1 >= 6){
            std::cout << "Coarse reconstruction: using 6 pixels distance to extract boundaries.\n\n\n";
            bCoarse = argv[6];
        }
        
        std::cout << "The path_file_cloud is " << path_dir_clouds << "\n"    // registration dir
                    <<"The str_file_search is " << str_file_search << "\n"
                    <<"The image_width is " << image_width << "\n"
                    <<"The image_height is " << image_height << "\n"
                    <<"The nInfoLevel is " << nInfoLevel << "\n"
                    <<"bCoarse " << bCoarse << "\n";
    }
    else{
        std::cout << "Please input 5 arguments with the order:\n"
                    << "[1] path_dir_clouds;\n"
                    << "[2] str_file_search;\n"
                    << "[3] image_width;\n"
                    << "[4] image_height;\n"
                    << "[5] nInfoLevel;\n"
                    << "[6] (optional) bCoarse (Default: \"Faulse\")\n";
        return -1;
    }

    OPTREG::nInfoLevel = nInfoLevel;
    OPTREG::fMaxSqDist = 1.5;
    OPTREG::nMinViews = 6;

    int nStepLoadMasks = 8; //2 for load sig setting; 12 for sa settting;
    int nBoundDistPixels = 900; //9 for sig setting, 900 for sa setting;
    int num_neighbors_refine_bound = 1000;


    if (bCoarse == "True"){
        OPTREG::max_dist_non_boundary = 225;  // 6 pixels for coarse reconstruction; 3 pixels for fine reconstruction
        num_neighbors_refine_bound = 300;
    }
    else{
        INFO_MSG("Use fine reconstruction.\n");
        OPTREG::max_dist_non_boundary = nBoundDistPixels;  // 6 pixels for coarse reconstruction; 3 pixels for fine reconstruction
    }
    INFO_MSG("\n\n\n\n\n\n************* Max distance(pixels^2) to extract boundary: %f. *************\n\n\n\n\n\n", OPTREG::max_dist_non_boundary);
   

    // OPTREG::fNormSearchRadius = 1.0 / 100.0; 
    // load ply file
    auto path_files = IOUtils::findFilesInDirectories(path_dir_clouds + "/pieces", str_file_search);
    INFO_MSG("Find %d files.\n", path_files.size());
    auto clouds = CloudUtils::loadPLYFiles(path_files);

    // Extract boundaries
    CloudReprojection reproj(image_height, image_width);
    ReprojData data_;

    // Extract amd Refine boundaries
    if (boost::algorithm::contains(str_file_search, "top")){
        str_file_search = "top";
    }
    else{
        str_file_search = "bottom";
    }
    OPEN_LOG("log_preprocess_Extract3DBoundaryAndRefine");

    std::string path_dir_masks = path_dir_clouds + "/"+str_file_search+"_masks";
    std::string path_file_cameras = path_dir_clouds + "/"+str_file_search+"_cameras/cameras_krc.bin";
    
    bool bMasks = IOUtils::checkPathExistence(path_dir_masks);
    bool bSampleMasks = true;
    if (bMasks){
        data_.masks = reproj.loadImageMasks(path_dir_masks, nStepLoadMasks); 
        data_.cameras = reproj.readCamerasInfo(path_file_cameras);
        reproj.extractMaskContours(data_.masks, data_.mask_contours);
        
        reproj._scale_cloud = (float)data_.masks.begin()->second.cols()/(float)image_width;  
    }
    else{
        INFO_MSG("Masks is not existent.\n");
        exit(-1);
    }
    
        
    std::vector<ReprojData> datas(clouds.size());
    if (clouds.size() > 1){
        OPTREG::nMinViews = 3;
    }
    #pragma omp parallel for  
    for (int i = 0; i < clouds.size(); i++){
        try{
            auto [parent_path, stem, ext] = IOUtils::getFileParentPathStemExt(path_files[i]);
            START_TIMER();
            ReprojData data = data_;
            data.cloud_piece = clouds[i];

            data.cloud_pclbound = reproj.extractCoarseBoundaryByMasks(data.cloud_piece, data.mask_contours, data.cameras);

            // save pclbound
            std::string path_file_pclbound = parent_path + "/" + stem + "_pclbound" + ext;
            INFO_MSG("Save file: %s.\n", path_file_pclbound.c_str());
            if(data.cloud_pclbound->points.size()==0){
                INFO_MSG("Use original point cloud to replace the bound.\n");
                data.cloud_pclbound = data.cloud_piece;

            }
            CloudUtils::savePLYFile(path_file_pclbound, data.cloud_pclbound);

            data.cloud_pclbound_refine = CloudUtils::refineBoundary(data.cloud_piece, data.cloud_pclbound, num_neighbors_refine_bound);
            data.cloud_pclbound_refine = CloudUtils::statisticalOutlierRemoval(data.cloud_pclbound_refine, 20, 1.0);
            
            std::string path_file_pclbound_refine = parent_path + "/" + stem + "_pclbound_refine" + ext;
            CloudUtils::savePLYFile(path_file_pclbound_refine, data.cloud_pclbound_refine);

            INFO_MSG("PCL boundary: %d; Refined boundary: %d.\n",  data.cloud_pclbound->points.size(), data.cloud_pclbound_refine->points.size()); 
            INFO_MSG("End of Extract3DBoundary [%d] %s.\n\n\n", i, END_TIMER());
        }
        catch(int e){
            std::cout << "error type: " << e << "\n";
            INFO_MSG("Fail Extract3DBoundary [%d].\n\n\n", i);

        }
    }

    std::cout << "Done." << "\n";
    CLOSE_LOG();
    return 0;
}