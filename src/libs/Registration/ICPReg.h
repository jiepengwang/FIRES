#pragma once
#include <vector>
#include <tuple>

#include <Eigen/Core>
#include <flann/flann.hpp>

#include "PointsReg.h"
#include "arguments.h"

struct RegPoints3Ds{ 
    Points3D piece;
    Points3D piece_normal;
    Points3D boundary;
    Points3D contour;  // points on the overlap area
    Points3D mvs_output;  // points on the overlap area
    Points3D contour_normal;
};


class ICPReg : public PointsReg
{
private:
    Points3D _points_source, _points_target,     // all points
                _points_nncontour_source, _points_nncontour_target, // points near contour
                _points_boundary_source, _points_boundary_target;
    Points2D _contour_source, _contour_target;  // contour points

    Points3D _points_source_cpy;

    std::vector<std::pair<int, int>> _corres, _corres_contour, 
                                        _corres_boundary_source, _corres_boundary_target;  // <target, source>
    Eigen::Matrix4f _trans = Eigen::Matrix4f::Identity();

    ICPRegArgs _args_icp;

    float _scale = 1.0;

    bool _flag_update_trim = false;


    // structure of data
    RegPoints3Ds _data_source, _data_target,
                _data_source_cpy, _data_target_cpy;

public:
    ICPReg(ICPRegArgs args_icp, Points3D points_source, Points3D points_target, 
                Points3D points_nncontour_source, Points3D points_nncontour_target, 
                Points3D points_boundary_source, Points3D points_boundary_target):
        _args_icp(args_icp),
        _points_source(points_source),
        _points_target(points_target),
        _points_nncontour_source(points_nncontour_source),
        _points_nncontour_target(points_nncontour_target),
        _points_boundary_source(points_boundary_source),
        _points_boundary_target(points_boundary_target)
    {
        std::cout << "\n\n[ICPReg Initialization]\n";
        std::cout << "The number of points in _points_source is " << _points_source.size() << ".\n";
        std::cout << "The number of points in _points_target is " << _points_target.size() << ".\n";
        std::cout << "The number of points in _points_nncontour_source is " << _points_nncontour_source.size() << ".\n";
        std::cout << "The number of points in _points_nncontour_target is " << _points_nncontour_target.size() << ".\n";
        std::cout << "The number of points in _points_boundary_source is " << _points_boundary_source.size() << ".\n";
        std::cout << "The number of points in _points_boundary_target is " << _points_boundary_target.size() << ".\n\n";

        checkNaNsInPoints3D(_points_source);
        checkNaNsInPoints3D(_points_target);

        checkNaNsInPoints3D(_points_nncontour_source);
        checkNaNsInPoints3D(_points_nncontour_target);

        checkNaNsInPoints3D(_points_boundary_source);
        checkNaNsInPoints3D(_points_boundary_target);

    }
    ICPReg();
    ~ICPReg();

    ICPReg(ICPRegArgs args_icp, RegPoints3Ds& points3Ds_source, RegPoints3Ds& points3Ds_target) :
        _args_icp(args_icp),
        _data_source(points3Ds_source),
        _data_target(points3Ds_target){
            _data_source_cpy = _data_source;
            _data_target_cpy = _data_target;
        }
        
    ICPReg(RegPoints3Ds& points3Ds_source, RegPoints3Ds& points3Ds_target) :
        _data_source(points3Ds_source),
        _data_target(points3Ds_target){
            _data_source_cpy = _data_source;
            _data_target_cpy = _data_target;
        }

    std::string _path_clouds;
    std::tuple<Eigen::MatrixXf, float> registerPoints3D_scaleicp(const bool bUseScale);

};