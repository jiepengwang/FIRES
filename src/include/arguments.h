#ifndef __ARGUMENTS_H__
#define __ARGUMENTS_H__

#include <iostream>
#include <string>
#include <exception>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
 
struct DataArgs{
    struct PathArgs{
        std::string path_read_config_file = "config_pcreg.json";
        std::string path_read_cloud_source;
        std::string path_read_cloud_target;
        std::string path_save_trans;
    }path_args;
    
    float dist;  // filter remote points in cloud

    struct SamplingArgs{
        bool uniform_sampling_flag;
        double uniform_sampling_leaf_size;
    }sampling_args;

    struct TransformArgs{
        Eigen::Vector3d scale;
        Eigen::AngleAxisd rot;
        Eigen::Vector3d translate;  
    }transform_args;
};


struct CloudPaths{
    std::string rawcloud_source, rawcloud_target,
                    piece_source, piece_target,
                    pclbound_source, pclbound_target,
                    pclbound_refine_source, pclbound_refine_target,
                    contour_source, contour_target; 

};

struct FGRArgs{
    bool flag_fgr;
    
    struct FPFHArgs{
        bool    visualization_flag;
        bool    uniform_sampling_flag;
        double  uniform_sampling_leaf_size;
        double  fpfh_search_radius;
        double  fpfh_thread_num;        
    }fpfh_args;

    struct CorresArgs{
        int     corres_knn;
        double  corre_dist;
        double  contour_dist;
    }corres_args;
};


struct ICPArgs{
    bool    b_icp, visualize_icp;
    int     icp_maxIter;
    double  trans_step;
    double  icp_corres_nnradius;
    double  icp_maxCorresDist, icp_transEpsilon, icp_euclideanEpsilon;

};



struct ICPRegArgs{
    std::string path_config_file = "config_icpreg.json";
    
    DataArgs data_args;
    CloudPaths paths;

    bool flag_pclbound;
    
    int     id_type_icp;

    int     num_iter_icp;
    int     num_iter_gauss = 10;
    float   dist_trim = 0.3,
            angle_trim;

    double  contour_search_radius; // Extract points whose projection on xy is near to 2D contour points
    double  corres_search_radius;  // Extract points near corrrespondence points

    struct Boundary{
        double  search_radius;  // Extract boundary
        double  search_angle;

        double  refine_radius;  // Refine boundary
        int     refine_neighbors; 
    }boundary;
    
    struct Visualization{
        struct CorresSampleRate{
            double contour;
            double boundary;
        }corres_sample_rate;
        
    }visualization;
    
};

struct ReprojArgs{
    std::string cameras_path_source, cameras_path_target, 
                masks_path_source, masks_path_target;

    std::string path_dir_mvs;
    float       scale_k;
    float       search_radius, search_angle;
    float       max_contour_dist = 4;
    int         min_views_num = 2;

    int         image_width, image_height;
};



struct PreprocessArgs
{
    std::string path_rawcloud_source, path_rawcloud_target,
                    path_pieceseg_source, path_pieceseg_target,
                    path_pclbound_source, path_pclbound_target,
                    path_pclbound_refine_source, path_pclbound_refine_target,
                    path_piececontour_source, path_piececontour_target; 
    
    ReprojArgs reproj_args;
};


struct RegArgs{
    DataArgs data_args;
    FGRArgs fgr_args;
    ICPArgs icp_args;

    ICPRegArgs icpreg_args;

    ReprojArgs reproj_args;
};


inline void parseArguments(RegArgs& reg_args, int argc, char** argv)
{
	// parse config file
    std::cout << "Start to parse config file: " << reg_args.data_args.path_args.path_read_config_file << '\n';
	boost::property_tree::ptree args_tree;  // read and parse config file
	try{
		boost::property_tree::read_json(reg_args.data_args.path_args.path_read_config_file, args_tree);

		// Input file and output file path
		reg_args.data_args.path_args.path_read_cloud_source = args_tree.get<std::string>("data_args.path_args.path_read_cloud_source");
		reg_args.data_args.path_args.path_read_cloud_target = args_tree.get<std::string>("data_args.path_args.path_read_cloud_target");
        reg_args.data_args.path_args.path_save_trans = args_tree.get<std::string>("data_args.path_args.path_save_trans");
		reg_args.data_args.dist = args_tree.get<float>("data_args.dist");
		
        std::cout << "Path_read_cloud_source: " << reg_args.data_args.path_args.path_read_cloud_source << '\n';
        std::cout << "Path_read_cloud_target: " << reg_args.data_args.path_args.path_read_cloud_target << '\n';
        std::cout << "Path_save_trans: " << reg_args.data_args.path_args.path_save_trans << "\n\n";

		// Sampling arguments
		reg_args.data_args.sampling_args.uniform_sampling_flag = args_tree.get<bool>("data_args.sampling_args.uniform_sampling_flag");
		reg_args.data_args.sampling_args.uniform_sampling_leaf_size = args_tree.get<double>("data_args.sampling_args.uniform_sampling_leaf_size");
        
        // Transformation arguments        
        {   
            // Scale
            int i = 0;
            for (boost::property_tree::ptree::value_type & value : args_tree.get_child("data_args.transform_args.scale")){
                reg_args.data_args.transform_args.scale[i] = value.second.get_value<double>(); 
                i++;
            }
            
            // Rot
            i = 0;
            reg_args.data_args.transform_args.rot.angle() = args_tree.get<double>("data_args.transform_args.rot.angle")/180.0*M_PI;
            for (boost::property_tree::ptree::value_type & value : args_tree.get_child("data_args.transform_args.rot.axis")){
                reg_args.data_args.transform_args.rot.axis()[i] = value.second.get_value<double>();
                i++;
            }

            // Translate
            i = 0;
            for (boost::property_tree::ptree::value_type & value : args_tree.get_child("data_args.transform_args.translate")){
                reg_args.data_args.transform_args.translate[i] = value.second.get_value<double>();
                i++;
            }
        }
        
        // FGR arguments
		{   
            reg_args.fgr_args.flag_fgr = args_tree.get<bool>("fgr_args.flag_fgr");
            reg_args.fgr_args.fpfh_args.uniform_sampling_flag = args_tree.get<bool>("fgr_args.fpfh_args.uniform_sampling_flag");
            reg_args.fgr_args.fpfh_args.uniform_sampling_leaf_size = args_tree.get<double>("fgr_args.fpfh_args.uniform_sampling_leaf_size");
            reg_args.fgr_args.fpfh_args.fpfh_search_radius = args_tree.get<double>("fgr_args.fpfh_args.fpfh_search_radius");
            reg_args.fgr_args.fpfh_args.fpfh_thread_num = args_tree.get<int>("fgr_args.fpfh_args.fpfh_thread_num");
            
            reg_args.fgr_args.corres_args.corres_knn = args_tree.get<int>("fgr_args.corres_args.corres_knn");
            reg_args.fgr_args.corres_args.corre_dist = args_tree.get<double>("fgr_args.corres_args.corre_dist");
            reg_args.fgr_args.corres_args.contour_dist = args_tree.get<double>("fgr_args.corres_args.contour_dist");
        }
		
        // ICPRegArgs
        {   
            reg_args.icpreg_args.contour_search_radius  = args_tree.get<double>("icpreg_args.contour_search_radius");
            reg_args.icpreg_args.corres_search_radius   = args_tree.get<double>("icpreg_args.corres_search_radius");
            
            reg_args.icpreg_args.boundary.search_radius = args_tree.get<double>("icpreg_args.boundary.search_radius");
            reg_args.icpreg_args.boundary.search_angle  = args_tree.get<double>("icpreg_args.boundary.search_angle");
            reg_args.icpreg_args.boundary.refine_radius  = args_tree.get<double>("icpreg_args.boundary.refine_radius");
            reg_args.icpreg_args.boundary.refine_neighbors  = args_tree.get<int>("icpreg_args.boundary.refine_neighbors");

            reg_args.icpreg_args.dist_trim = args_tree.get<double>("icpreg_args.dist_trim");
            reg_args.icpreg_args.num_iter_icp = args_tree.get<int>("icpreg_args.num_iter_icp");
            reg_args.icpreg_args.num_iter_gauss = args_tree.get<int>("icpreg_args.num_iter_gauss");

            // parse visualization args
            {
                reg_args.icpreg_args.visualization.corres_sample_rate.boundary = args_tree.get<double>("icpreg_args.visualization.corres_sample_rate.boundary");
                reg_args.icpreg_args.visualization.corres_sample_rate.contour = args_tree.get<double>("icpreg_args.visualization.corres_sample_rate.contour");
            }

        }
		
        // Icp-related arguments
        {   
            reg_args.icp_args.b_icp = args_tree.get<bool>("icp_args.b_icp");
            reg_args.icp_args.visualize_icp = args_tree.get<bool>("icp_args.visualize_icp");
            reg_args.icp_args.icp_maxCorresDist = args_tree.get<double>("icp_args.maxCorresDist");
            reg_args.icp_args.icp_maxIter = args_tree.get<double>("icp_args.maxIter");
            reg_args.icp_args.icp_corres_nnradius = args_tree.get<double>("icp_args.corres_nnradius");  
            reg_args.icp_args.icp_transEpsilon = args_tree.get<double>("icp_args.transEpsilon");
            reg_args.icp_args.icp_euclideanEpsilon = args_tree.get<double>("icp_args.euclideanEpsilon"); 
        }

        // Reproj args
        {
            reg_args.reproj_args.cameras_path_source = args_tree.get<std::string>("reproj_args.cameras_path_source");
            reg_args.reproj_args.cameras_path_target = args_tree.get<std::string>("reproj_args.cameras_path_target");

            reg_args.reproj_args.masks_path_source = args_tree.get<std::string>("reproj_args.masks_path_source");
            reg_args.reproj_args.masks_path_target = args_tree.get<std::string>("reproj_args.masks_path_target");

            reg_args.reproj_args.min_views_num = args_tree.get<int>("reproj_args.min_views_num");
            reg_args.reproj_args.max_contour_dist = args_tree.get<float>("reproj_args.max_contour_dist");
        }
	}
	catch(const std::exception& e){
		std::cerr << e.what() << '\n';
		std::cout << "Error in reading config file " << reg_args.data_args.path_args.path_read_config_file << '\n';
		return;
	}
 }




inline void parseICPRegArgs(ICPRegArgs& reg_args, int argc, char** argv)
{
	// parse config file
    std::cout << "Start to parse config file: " << reg_args.path_config_file << '\n';
	boost::property_tree::ptree args_tree;  // read and parse config file
	try{
		boost::property_tree::read_json(reg_args.path_config_file, args_tree);

		// Input file and output file path
		reg_args.paths.piece_source = args_tree.get<std::string>("paths.piece_source");
		reg_args.paths.piece_target = args_tree.get<std::string>("paths.piece_target");
        
        reg_args.paths.pclbound_source = args_tree.get<std::string>("paths.pclbound_source");
		reg_args.paths.pclbound_target = args_tree.get<std::string>("paths.pclbound_target");
		
        reg_args.paths.pclbound_refine_source = args_tree.get<std::string>("paths.pclbound_refine_source");
		reg_args.paths.pclbound_refine_target = args_tree.get<std::string>("paths.pclbound_refine_target");

        reg_args.paths.contour_source = args_tree.get<std::string>("paths.contour_source");
		reg_args.paths.contour_target = args_tree.get<std::string>("paths.contour_target");

		// Sampling arguments
		reg_args.data_args.sampling_args.uniform_sampling_flag = args_tree.get<bool>("data_args.sampling_args.uniform_sampling_flag");
		reg_args.data_args.sampling_args.uniform_sampling_leaf_size = args_tree.get<double>("data_args.sampling_args.uniform_sampling_leaf_size");
        
        // Transformation arguments        
        {   
            // Scale
            int i = 0;
            for (boost::property_tree::ptree::value_type & value : args_tree.get_child("data_args.transform_args.scale")){
                reg_args.data_args.transform_args.scale[i] = value.second.get_value<double>(); 
                i++;
            }
            
            // Rot
            i = 0;
            reg_args.data_args.transform_args.rot.angle() = args_tree.get<double>("data_args.transform_args.rot.angle")/180.0*M_PI;
            for (boost::property_tree::ptree::value_type & value : args_tree.get_child("data_args.transform_args.rot.axis")){
                reg_args.data_args.transform_args.rot.axis()[i] = value.second.get_value<double>();
                i++;
            }

            // Translate
            i = 0;
            for (boost::property_tree::ptree::value_type & value : args_tree.get_child("data_args.transform_args.translate")){
                reg_args.data_args.transform_args.translate[i] = value.second.get_value<double>();
                i++;
            }
        }
        

        // ICPRegArgs
        {   
            reg_args.flag_pclbound  = args_tree.get<bool>("icpreg_args.flag_pclbound");
            reg_args.contour_search_radius  = args_tree.get<double>("icpreg_args.contour_search_radius");
            reg_args.corres_search_radius   = args_tree.get<double>("icpreg_args.corres_search_radius");
            
            reg_args.boundary.search_radius = args_tree.get<double>("icpreg_args.boundary.search_radius");
            reg_args.boundary.search_angle  = args_tree.get<double>("icpreg_args.boundary.search_angle");
            reg_args.boundary.refine_radius  = args_tree.get<double>("icpreg_args.boundary.refine_radius");
            reg_args.boundary.refine_neighbors  = args_tree.get<int>("icpreg_args.boundary.refine_neighbors");

            reg_args.dist_trim = args_tree.get<float>("icpreg_args.dist_trim");
            reg_args.angle_trim = args_tree.get<float>("icpreg_args.angle_trim");
            reg_args.num_iter_icp = args_tree.get<int>("icpreg_args.num_iter_icp");
            reg_args.num_iter_gauss = args_tree.get<int>("icpreg_args.num_iter_gauss");

            reg_args.id_type_icp = args_tree.get<int>("icpreg_args.id_type_icp");

            // parse visualization args
            {
                reg_args.visualization.corres_sample_rate.boundary = args_tree.get<double>("icpreg_args.visualization.corres_sample_rate.boundary");
                reg_args.visualization.corres_sample_rate.contour = args_tree.get<double>("icpreg_args.visualization.corres_sample_rate.contour");
            }

        }

	}
	catch(const std::exception& e){
		std::cerr << e.what() << '\n';
		std::cout << "Error in reading config file " << reg_args.data_args.path_args.path_read_config_file << '\n';
		return;
	}
 }

#endif
