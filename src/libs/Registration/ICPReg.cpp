#include "ICPReg.h"
#include <pcl/common/common.h>
#include "PointsUtils.h"
#include "CloudUtils.h"
#include "ScaleICP.h"
#include "IOUtils.h"
#include "LogUtils.h"

#include <omp.h>
#include <boost/date_time/posix_time/posix_time.hpp>  
#include <boost/date_time/gregorian/gregorian.hpp>

ICPReg::ICPReg(/* args */)
{
}

ICPReg::~ICPReg()
{
}



std::tuple<Eigen::MatrixXf, float> ICPReg::registerPoints3D_scaleicp(const bool bUseScale)
{
    /*
    Register source to target
    */
    int num_iter_icp = OPTREG::nICPIter;
    // bool b_scaling = true;
    ScaleICP regfuncs;
    float relative_rmse = 0.0005;

    Eigen::Vector3f interval_scaling  = Eigen::Vector3f::Identity();

    interval_scaling[2] = 1.0;
    interval_scaling[0] = 0.8;

    // Global and local trans and scale
    Eigen::Matrix4f trans_global = Eigen::Matrix4f::Identity(), trans;
    float scale_global = interval_scaling[2], scale;
    float error_last = -1.0;

    Pairs corres_bound, corres_bound_source, corres_bound_target;
    Pairs corres_contour, corres, corres_contour_sym;
    
    // The number of corres
    int num_corres;
    
    enum types_sicp{bound_sicp, bound_belt_sicp, bound_sicp_belt_s, bound_icp_belt_s};
    int id_types_sicp = OPTREG::nICPType;

    float relative_rmse_ = 1.0;
    std::string dir_rmse_s = _path_clouds + "/reg/RMSE_scale.txt";
    if(!IOUtils::checkParentPathExistence(dir_rmse_s)){
        std::cout << "Fail to create directory " << dir_rmse_s << ".\n";
        exit(-1); 
    }
    
    std::fstream fwrite(dir_rmse_s, std::ios::out);
    fwrite << "Step RMSE Scale\n";
    for (int i = 0; i < num_iter_icp; i++)
    {
        std::cout << "\n\nStep " << i << "...\n";
        
        
        switch(id_types_sicp)
        {
            case 8:
            {
                // point2plane.
                std::cout <<"(8) Point2plane: Bound SICP.\n";
                float dist_trim_sicp = 0.01;
                regfuncs.findICPCorrespondences(_data_source.boundary, _data_target.piece, corres_bound_source);
                regfuncs.findICPCorrespondences(_data_target.boundary, _data_source.piece, corres_bound_target);

                auto [mat_source1, mat_target1, mat_target1_normal] =
                    regfuncs.getCorresMatsWithNormal(_data_source.boundary, _data_target.piece, _data_target.piece_normal, corres_bound_source);
                auto [mat_target2, mat_source2, mat_target2_normal] =
                    regfuncs.getCorresMatsWithNormal(_data_target.boundary, _data_source.piece, _data_source.piece_normal, corres_bound_target);

                num_corres = _data_source.boundary.size() + _data_target.boundary.size();
                Eigen::MatrixXf mat_source(mat_source1.rows(), num_corres),
                                mat_target(mat_source1.rows(), num_corres),
                                mat_target_normal(mat_source1.rows(), num_corres);
                std::cout << "The number of corres is " << num_corres << ".\n";

                mat_source << mat_source1, mat_source2;
                mat_target << mat_target1, mat_target2;
                mat_target_normal << mat_target1_normal, mat_target2_normal;

                std::tie(trans, scale) =
                    regfuncs.updateSRTPoint2Plane(mat_source, mat_target, mat_target_normal, bUseScale, interval_scaling);

                break;
            }

            default:
                break;
        }

        CloudUtils::transformPoints3D(_data_source.piece, trans);
        CloudUtils::transformPoints3D(_data_source.boundary, trans);

        
        // Update glocal transformation and scale
        trans_global = trans * trans_global;
        scale_global *= scale;
        interval_scaling[2] = scale_global;

        VERBOSE_MSG("The transformation is \n%s.\n", LogUtils::toString(trans).c_str());
        std::cout << "The global scale is " << scale_global << ".\n";

        // Check whether ICP is converged.
        float err_rmse_source_bound = PointsReg::calculateRMSE(_data_source.boundary, _data_target.piece, Eigen::Matrix4f::Identity());
        float err_rmse_target_bound = PointsReg::calculateRMSE(_data_target.boundary, _data_source.piece, Eigen::Matrix4f::Identity());

        float err_rmse_num = err_rmse_source_bound*err_rmse_source_bound*_data_source.boundary.size() + 
                                err_rmse_target_bound*err_rmse_target_bound*_data_target.boundary.size();
        float err_rmse_den = _data_source.boundary.size() + _data_target.boundary.size();
        float err_rmse = std::sqrt(err_rmse_num / err_rmse_den);
        fwrite << i << " " << err_rmse << " " << scale_global << "\n";

        if (i==0){
            error_last = err_rmse;
        }
        else{
            relative_rmse_ = std::abs(1- err_rmse/error_last);
            error_last = err_rmse;
            
            std::cout << "The RMSE is " << err_rmse << "\n";
            std::cout << "The ratio of RMSE is " << relative_rmse_ << "\n";
            if (relative_rmse_ < relative_rmse){
                std::cout << "The algorithm has converged at step " << i << ".\n";
                // CloudUtils::visualizeClouds(_data_source.boundary, _data_target.piece);
                break;
            }
        }
        
        // Visualize regisration result every N step
        if (OPTREG::nInfoLevel >=5 && i%5 == 0){
            CloudUtils::visualizeClouds(_data_source.boundary, _data_target.piece);
        }
    }
    fwrite.close();

    return {trans_global, scale_global};
}