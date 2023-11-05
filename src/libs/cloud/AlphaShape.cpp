#include "AlphaShape.h"
#include <math.h>

#include <Eigen/SVD>
#include <Eigen/Dense>
#include "LogUtils.h"
#include "CloudUtils.h"

AlphaShape::AlphaShape(/* args */)
{
}

AlphaShape::~AlphaShape()
{
}


std::vector<Eigen::Vector2f> AlphaShape::extractAlphaShape(const std::list<Point>& points, double alpha_value)
{
    Alpha_shape_2 A(points.begin(), points.end(),
                    FT(10000),
                    Alpha_shape_2::GENERAL);
    A.set_alpha(alpha_value);

    std::vector<Eigen::Vector2f> points_eigen;
    Eigen::Vector2f tmp_point;
    Alpha_shape_edges_iterator it = A.alpha_shape_edges_begin(),
                                end = A.alpha_shape_edges_end();
    for( ; it!=end; ++it){
        auto seg_tmp = A.segment(*it);
        Eigen::Vector2f tmp_source= {seg_tmp.source().x(), seg_tmp.source().y()};
        points_eigen.push_back(tmp_source);

        Eigen::Vector2f tmp_target= {seg_tmp.target().x(), seg_tmp.target().y()};
        points_eigen.push_back(tmp_target);  
    }
    //savePointsToFile("./data/contour_unsorted.txt", points_eigen);
    VERBOSE_MSG("[AlphaShape] The number of extracted points of alpha contour is %d.\n", points_eigen.size()); 
    VERBOSE_MSG("Optimal alpha: %s.\n", LogUtils::toString(*A.find_optimal_alpha(1)).c_str()); 

    return points_eigen;
}

std::vector<Eigen::Vector2f> AlphaShape::sortContourPoints(const std::vector<Eigen::Vector2f>& points)
{
    auto comp = [](const Eigen::Vector2f& c1, const Eigen::Vector2f& c2){
        return c1[0] < c2[0] || (c1[0] == c2[0] && c1[1] < c2[1]);};

    std::map<Eigen::Vector2f,Eigen::Vector2f,decltype(comp)> points_pair(comp);
    int num_points = points.size()/2;
    for (size_t i = 0; i < num_points; i++){
        points_pair[points[2*i]] =  points[2*i+1];
    }

    // int i = 0; 
    int num_pairs = points_pair.size();
    int count = 0;
    auto iter_begin = points_pair.begin(), next = points_pair.begin();
    std::vector<Eigen::Vector2f> points_sorted;
    do{
        auto tmp = next->second;
        points_sorted.push_back(tmp);

        next = points_pair.find(tmp);
        if(count > num_pairs){
            INFO_MSG("Fail to find a continuous contour curve.\n");
            break;
        }
        // ASSERT(count <= num_pairs);
        count++;
    } while (next != iter_begin);
    
    VERBOSE_MSG("[sortContourPoints] The number of points on alpha contour is %d.\n", points_sorted.size());
    return points_sorted;
}

 std::vector<Eigen::Vector2f> AlphaShape::sampleContourPoints(const std::vector<Eigen::Vector2f>& points, int num_sampled)
 {
    std::vector<float> dists, dists_sum, dists_sum_norm;
    int num_points = points.size();

    INFO_MSG("Sorted contour points: %d.\n", num_points);
    // Calculate distance between neighboring points
    for (int i = 0; i < num_points; i++){
        int ind1 = i;
        int ind2 = i+1;
        if (ind2 == num_points){
            ind2 = 0;
        } 

        // Save dists between two points 
        float temp = (points[ind2]-points[ind1]).norm()+ 1e-6;
        assert(temp>0);
        dists.push_back(temp);

        // Save accumulated sum
        if(i==0){
            dists_sum.push_back(temp);
        }
        else{
            dists_sum.push_back(dists_sum[i-1]+ dists[i]);
        }
    }

    // Normalize sum
    for (int  i = 0; i < num_points; i++){
        dists_sum_norm.push_back(dists_sum[i]/dists_sum[num_points-1]);
    }
    
    // Sample equidistant points
    std::vector<Eigen::Vector2f> points_sampled;
    points_sampled.push_back(points[0]);

    float step = 1.0/num_sampled;
    DEBUG_MSG("Print value t:\n"); 
    float t = 0.0; 
    for(int i = 1; i < num_sampled; i++){
        float pos = step*i;
        int ind_seg = std::lower_bound(dists_sum_norm.begin(), dists_sum_norm.end(), pos) - dists_sum_norm.begin();
        ASSERT(ind_seg >=0 && ind_seg<num_sampled);

        int ind_seg_point_begin = ind_seg;
        int ind_seg_point_end = ind_seg + 1;
        if(ind_seg_point_end == num_points){
            INFO_MSG("Sample a point at the final segment.\n");
            ind_seg_point_end = 0;
        }

        // INterpolate a point beteen two points
        if (ind_seg==0){
            t = pos/dists_sum_norm[i];
        }
        else{
            t = (pos - dists_sum_norm[ind_seg-1])/(dists_sum_norm[ind_seg] - dists_sum_norm[ind_seg-1]);
        }
        ASSERT(t>=0 && t<=1);
        Eigen::Vector2f interp_point = (1-t) *points[ind_seg_point_begin] + t*points[ind_seg_point_end];
        points_sampled.push_back(interp_point);     

        if(OPTREG::nInfoLevel >= DEBUG_LEVEL){
            DEBUG_MSG("%f ", t);  
            std::cout << "Index seg: " << ind_seg <<"; i, " <<  i << ": Begin, " << points[ind_seg_point_begin].transpose() << ";     End, " <<  points[ind_seg_point_end].transpose() << ";     Inter, " << interp_point.transpose() << ".\n";
        }
    }
    DEBUG_MSG("\n");       
    VERBOSE_MSG("The number of sampled points on contour is %d.\n", points_sampled.size());
    return points_sampled; 
 }


Eigen::VectorXf AlphaShape::calculateDirectedAngles(const std::vector<Eigen::Vector2f>& points)
{
    int num_points = points.size();
    Eigen::VectorXf angles(num_points);

    for (int i = 0; i < num_points; i++){      
        auto index1 = (i == 0 ? num_points-1 : i-1), index2 = i;
        auto index3 = (i == num_points-1 ? 0 : i+1);

        auto p1 = points[index2] - points[index1];
        auto p2 = points[index3] - points[index2];

        float angle = atan2(p1[0]*p2[1]-p1[1]*p2[0], p1[0]*p2[0]+p1[1]*p2[1]) * 180 / M_PI; // degrees
        angles[i] = angle;
    }
    return angles;
}

Eigen::VectorXf AlphaShape::calculateCumulativeSum(Eigen::VectorXf& angles)
{
    int num_angles = angles.size();
    Eigen::VectorXf angels_sum(num_angles);
    for (int i = 0; i < num_angles; i++){
        if (i ==0){
            angels_sum[i] = angles[0];
        }
        else{
            angels_sum[i] =  angels_sum[i-1]+angles[i];
        }
    }
    if (angels_sum[num_angles-1] < 0){
        std::cout << "The sum of angles is smaller than 0, reversing number signs.\n";
        angles = -angles;
    }
    
    return angels_sum;
}

std::tuple<Eigen::VectorXf, float> AlphaShape::findBestMatch(Eigen::VectorXf& angles_source, Eigen::VectorXf& angles_target)
{
    int num_angles = angles_target.size();
    Eigen::VectorXf source_indices(num_angles);
   
    Eigen::VectorXf angles_source_sorted(num_angles);     // re-sort angles
    auto target_sum = calculateCumulativeSum(angles_target);
    // clockwise
    Eigen::VectorXf source_sum_clock(num_angles);   // L2 norm between source sum and target sum with diff sequence
    for ( int i = 0; i < num_angles; i++){
        for (int j = 0; j < num_angles; j ++){
            auto ind = ( i+j< num_angles ? i+j: i+j-num_angles);
            angles_source_sorted[j] = angles_source[ind];
        }

        Eigen::VectorXf angles_source_sorted_sum = calculateCumulativeSum(angles_source_sorted);
        float dist = (target_sum-angles_source_sorted_sum).squaredNorm();
        source_sum_clock[i] = dist;
    }


    Eigen::VectorXf::Index ind_clock, ind_anticlock;
    auto min_clock = source_sum_clock.minCoeff(&ind_clock);
    // auto min_anticlock = source_sum_anticlock.minCoeff(&ind_anticlock);

    float dist_best_match;
    if( true ){
        for (int j = 0; j < num_angles; j ++){
            auto ind = ( ind_clock+j< num_angles ? ind_clock+j: ind_clock+j-num_angles);
            source_indices[j] = ind;
        }
        dist_best_match = min_clock;
    }

    return {source_indices, dist_best_match};
 }

Eigen::Matrix4f AlphaShape::calcualteTransformation(const std::vector<Eigen::Vector2f>& points_source, const std::vector<Eigen::Vector2f>& points_target)
{
    // From source to target
    int num_points = points_source.size();
    int dim_points = points_source[0].size();

    Eigen::MatrixXf mat_source(2, num_points), mat_target(2, num_points);  // 2*N
    for (int i = 0; i < num_points; i++){
        mat_source.col(i) = points_source[i];
        mat_target.col(i) = points_target[i];
    }

    Eigen::Vector2f center_source = mat_source.rowwise().mean();
    Eigen::Vector2f center_target = mat_target.rowwise().mean();

    Eigen::MatrixXf mat_source_centered = mat_source.colwise() - center_source;
    Eigen::MatrixXf mat_target_centered = mat_target.colwise() - center_target;

    const Eigen::MatrixXf mat_cov = mat_source_centered * mat_target_centered.transpose();
    Eigen::JacobiSVD<Eigen::Matrix2f> svd( mat_cov, Eigen::ComputeFullV | Eigen::ComputeFullU );
    
    Eigen::Matrix2f mat_rot = svd.matrixV()*svd.matrixU().transpose();
    Eigen::Vector2f vec_translate = center_target - mat_rot*center_source;
    
    Eigen::Matrix4f trans_svd = Eigen::Matrix4f::Identity();
    trans_svd.block<2,2>(0,0) = mat_rot;
    trans_svd.block<2,1>(0,3) = vec_translate;
    if (mat_rot.determinant()<0){
        std::cout << "The determinant of Rotation Matrix of source 2D contour is smaller than 0.\n";
        trans_svd(2,2) = -1;
    }

    return trans_svd;
}


std::tuple<std::vector<Eigen::Vector2f>,std::vector<Eigen::Vector2f>,Eigen::Matrix4f>
AlphaShape::registerContour(const std::vector<Eigen::Vector2f>& points_source, 
                                                const std::vector<Eigen::Vector2f>& points_target,
                                                const double alpha_value, const int num_sampled)
{
    // Transfrom source to target
    auto source_points_angles = getContourAngleSignature(points_source, alpha_value, num_sampled);
    auto target_points_angles = getContourAngleSignature(points_target, alpha_value, num_sampled);
    auto [indices, min_cost] = findBestMatch(std::get<1>(source_points_angles), std::get<1>(target_points_angles));

    std::vector<Eigen::Vector2f> points_source_sorted;
    for (int i = 0; i < indices.size(); i++){
        points_source_sorted.push_back(std::get<0>(source_points_angles)[indices[i]]);
    }

    Eigen::Matrix4f trans = calcualteTransformation(points_source_sorted, std::get<0>(target_points_angles));
    std::get<0>(source_points_angles) = points_source_sorted;
    return std::tuple<std::vector<Eigen::Vector2f>,std::vector<Eigen::Vector2f>,Eigen::Matrix4f>(std::get<0>(source_points_angles), std::get<0>(target_points_angles), trans);
}


std::tuple< std::vector<Eigen::Vector2f>, Eigen::VectorXf > 
AlphaShape::getContourAngleSignature(const std::vector<Eigen::Vector2f>& points, double alpha_value, int num_sampled)
{
    std::list<Point> points_list;

    int num_points = points.size();
    for (int i = 0; i < num_points; i++){
        points_list.push_back(Point(points[i][0], points[i][1]));
    }

    Points2Df points_contour = extractAlphaShape(points_list, alpha_value); //0.01
	Points2Df points_sorted = sortContourPoints(points_contour);
	Points2Df points_sampled = sampleContourPoints(points_sorted, num_sampled);
	Eigen::VectorXf angles = calculateDirectedAngles(points_sampled);

    Eigen::VectorXf angles_weighted = addWeightedMeanFilter(angles, Eigen::Vector3f(0.3, 0.4, 0.3));  // 0.2, 0.6.0.2

   
    if(OPTREG::nInfoLevel>=DEBUG_LEVEL){
        // Save sorted points
        savePointsToFile("./data/contour_points_sorted.txt", points_sorted);
        savePointsToFile("./data/contour_points_sampled.txt", points_sampled);
        DEBUG_MSG("Visualize points and points_sorted, points and points_sampled.\n");
        CloudUtils::visualizeClouds(points, points_sorted, 8);
        CloudUtils::visualizeClouds(points, points_sampled, 8);
    }

    return std::tuple< std::vector<Eigen::Vector2f>, Eigen::VectorXf>(points_sampled, angles_weighted);
}

int AlphaShape::savePointsToFile(const std::string& filename, const std::vector<Eigen::Vector2f>& points)
{
    std::ofstream write_handle(filename, std::ios::out);
    int num_points = points.size();
    if ( write_handle.is_open()){
        write_handle << num_points << "\n";

        for (int i = 0; i < num_points; i++){
            write_handle << points[i][0] << " " <<  points[i][1] << "\n";
        }
    }
    else{
        std::cerr << "Fail to write points in file " << filename << ".\n";
        return -1;
    }

    std::cout << "Save " << num_points << " points in file " << filename << ".\n";
    write_handle.close();
    return 0;
}


Eigen::VectorXf AlphaShape::addWeightedMeanFilter(const Eigen::VectorXf& angles, const Eigen::Vector3f& filter_kernel)
{
    int num_angles = angles.size();
    Eigen::VectorXf angles_weighted(num_angles);

    for (int i = 0; i < num_angles; i++){
        int ind1 = (i!=0 ? i-1 : num_angles-1);
        int ind2 = i;
        int ind3 = ( i!=num_angles-1 ? i+1 : 0 );
        angles_weighted[i] = filter_kernel[0]*angles[ind1] + filter_kernel[1]*angles[ind2] + filter_kernel[2]*angles[ind3];
    }  

    return angles_weighted;
}