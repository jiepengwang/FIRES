#pragma once
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Alpha_shape_2.h>
#include <CGAL/Alpha_shape_vertex_base_2.h>
#include <CGAL/Alpha_shape_face_base_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/algorithm.h>
#include <CGAL/assertions.h>
#include <fstream>
#include <iostream>
#include <list>
#include <vector>
#include <Eigen/Core>

#include <tuple>

typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
typedef K::FT                                                FT;
typedef K::Point_2                                           Point;
typedef K::Segment_2                                         Segment;
typedef CGAL::Alpha_shape_vertex_base_2<K>                   Vb;
typedef CGAL::Alpha_shape_face_base_2<K>                     Fb;
typedef CGAL::Triangulation_data_structure_2<Vb,Fb>          Tds;
typedef CGAL::Delaunay_triangulation_2<K,Tds>                Triangulation_2;
typedef CGAL::Alpha_shape_2<Triangulation_2>                 Alpha_shape_2;
typedef Alpha_shape_2::Alpha_shape_edges_iterator            Alpha_shape_edges_iterator;

class AlphaShape
{
private:
    /* data */
public:
    AlphaShape(/* args */);
    ~AlphaShape();

    // static std::list<Point> readPointsFromFile(const std::string& filename);
    static int savePointsToFile(const std::string& filename, const std::vector<Eigen::Vector2f>& points);
    
    static std::vector<Eigen::Vector2f> extractAlphaShape(const std::list<Point>& points, double alpha_value);
    static std::vector<Eigen::Vector2f> sortContourPoints(const std::vector<Eigen::Vector2f>& points);
    static std::vector<Eigen::Vector2f> sampleContourPoints(const std::vector<Eigen::Vector2f>& points, int num_sampled);
    static Eigen::VectorXf calculateDirectedAngles(const std::vector<Eigen::Vector2f>& points);
    static Eigen::VectorXf calculateCumulativeSum(Eigen::VectorXf& angles);
    
    static std::tuple<Eigen::VectorXf, float> 
    findBestMatch(Eigen::VectorXf& angles_source, Eigen::VectorXf& angles_target);
    
    static Eigen::Matrix4f calcualteTransformation(const std::vector<Eigen::Vector2f>& points_source, const std::vector<Eigen::Vector2f>& points_target);
    
    // Filter
    static Eigen::VectorXf addWeightedMeanFilter(const Eigen::VectorXf& angles, const Eigen::Vector3f& filter_kernel);


    static std::tuple< std::vector<Eigen::Vector2f>, Eigen::VectorXf > 
    getContourAngleSignature(const std::vector<Eigen::Vector2f>& points, double alpha_value, int num_sampled);
    
    static std::tuple<std::vector<Eigen::Vector2f>,std::vector<Eigen::Vector2f>,Eigen::Matrix4f>
    registerContour(const std::vector<Eigen::Vector2f>& points_source, 
                                                    const std::vector<Eigen::Vector2f>& points_target,
                                                    const double alpha_value, const int num_sampled);

    void test();
};

