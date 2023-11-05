//
// Created by aska on 2020/11/20.
//
//#include <cnpy.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

#include "LogUtils.h"
#include "IOUtils.h"
#include <string>

void Calibrate(const std::vector<std::string>& vec_path_files, const std::string path_pose, 
				int thres_code, float board_width, bool cali_cam, int board_coord_method,
				std::string path_sfm_intrin)
{
	std::vector<std::vector<cv::Point3f>> obj_points;
	std::vector<int> board_ids;

	double scale = (board_width - board_width / 161 * 2) / 3975.0;
	for (int x = 0; x < 20; x++)
	{
		for (int y = 0; y < 20; y++) 
		{
			// v1
			if(board_coord_method==1){
			int a = x * 200;  // x-axis
			int b = 3975 - y * 200;  // y-axis

			obj_points.emplace_back();
			board_ids.emplace_back(x * 20 + y);

			auto &current_vec = obj_points.back();   // axis: bottom-left corner, x axis: left->right; y axis: bottom->top
			// clockwise from top-left
			current_vec.emplace_back(a * scale, b * scale, 0);
			current_vec.emplace_back((a + 175) * scale, b * scale, 0);
			current_vec.emplace_back((a + 175) * scale, (b - 175) * scale, 0);
			current_vec.emplace_back(a * scale, (b - 175) * scale, 0);
			}
			// v2
			if (board_coord_method == 2){
				int a = x * 200;
				int b = y * 200;

				obj_points.emplace_back();
				board_ids.emplace_back(x + 20 * y); // board_ids.emplace_back(x * 20 + y);
				//board_ids.emplace_back(20 * x +  y); // board_ids.emplace_back(x * 20 + y);
				auto& current_vec = obj_points.back();
				current_vec.emplace_back(a * scale, b * scale, 0);
				current_vec.emplace_back((a + 175) * scale, b * scale, 0);
				current_vec.emplace_back((a + 175) * scale, (b + 175) * scale, 0);
				current_vec.emplace_back(a * scale, (b + 175) * scale, 0);
			}
		}
	}

		
	cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_1000);
	cv::Ptr<cv::aruco::Board> board= cv::aruco::Board::create(obj_points, dictionary, board_ids); 


	std::vector<std::vector<cv::Point2f>> all_corners_cated;
	std::vector<int> all_ids_cated;
	std::vector<int> marker_count_per_frame;

	cv::Size img_size;
	int n = vec_path_files.size();
	std::vector<std::string> all_ids_valid;

	std::cout << "Precalibrated intrinsics by SfM: \n";
	float data[9] = { 4760.745904499341, 0, 1501.582972821507,
						 0, 4760.745904499341, 919.0317741559792,
						 0, 0, 1.0};

	cv::Mat cam_intrin = cv::Mat(3, 3, CV_32F, data);
	cam_intrin = cam_intrin * 2;
	cam_intrin.at<float>(2,2) = 1.0;
	if (path_sfm_intrin.size()>2) {
		std::fstream fintrin(path_sfm_intrin);
		if (!fintrin.is_open()) {
			std::cout << "Fail to open intrinsics file: " << path_sfm_intrin << "\n";
		}
		else {
			for (size_t m = 0; m < 3; m++){
				for (size_t n = 0; n < 3; n++){
					fintrin >> cam_intrin.at<float>(m, n);
				}
			}
		}
	}
	std::cout << "Cam intrin: " << cam_intrin << ".\n";  

	cv::Mat cam_distort = cv::Mat::zeros(cv::Size(5, 1), CV_32F);
	std::cout << "cam_distort: " << cam_distort << ".\n";
	std::vector<cv::Mat> rvecs, tvecs;

	for (int i = 0; i < n; i++)
	{
		cv::Mat img2 = cv::imread(vec_path_files[i]);
		cv::Mat img;
		cv::extractChannel(img2, img, 0);
		if (OPTREG::nInfoLevel >= VISUAL_LEVEL) {
			std::string path_debug = "./test";
			IOUtils::ensurePathExistence(path_debug);
			cv::imwrite(path_debug + "/channel_b" + std::to_string(i) + ".jpg", img);
		}

		cv::threshold(img, img, thres_code, 255, cv::ThresholdTypes::THRESH_BINARY);
		if(OPTREG::nInfoLevel>=VISUAL_LEVEL){
			std::string path_debug =  "./test";
			IOUtils::ensurePathExistence(path_debug);
			cv::imwrite(path_debug + "/" +std::to_string(i) + ".jpg", img);
		}

		img_size = img.size();
		
		 
		std::vector<int> marker_ids;
		std::vector<std::vector<cv::Point2f>> marker_corners, rejected_candidates;
		cv::aruco::detectMarkers(img, dictionary, marker_corners, marker_ids, parameters, rejected_candidates);
		cv::aruco::refineDetectedMarkers(img, board, marker_corners, marker_ids, rejected_candidates);

		std::cout  << i <<" " <<  marker_corners.size() << " " << img_size << "\n";
		if (marker_corners.size() == 0) {
			INFO_MSG("Skip  %d. No markers detected\n", i);
			continue;
		}

		// if at least one marker detected
		if (marker_ids.size() > 0) {
			cv::Vec3d rvec2, tvec2;
			int valid = cv::aruco::estimatePoseBoard(marker_corners, marker_ids, board, cam_intrin, cam_distort, rvec2, tvec2);
			if (cali_cam == false) {
				rvecs.emplace_back(rvec2);
				tvecs.emplace_back(tvec2);
			}

		cv::Mat imageCopy;
		img2.copyTo(imageCopy);
			cv::aruco::drawDetectedMarkers(imageCopy, marker_corners, marker_ids, cv::Scalar(0, 0, 255));
			cv::drawFrameAxes(imageCopy, cam_intrin, cam_distort, rvec2, tvec2, 10);
		cv::imwrite("./test/markers_" + std::to_string(i) + ".jpg", imageCopy);

		}

		if (OPTREG::nInfoLevel >= VERBOSE_LEVEL) {
			std::cout << "Code id and corner points:\n";
			for (int i = 0; i < 30; i++)
				std::cout << "Code id and corner points: " << board_ids[i] << ": " << obj_points[i] << ".\n";
		}

		boost::filesystem::path bpath(vec_path_files[i]);
		std::string stem = bpath.stem().string();
		all_ids_valid.emplace_back(stem);
		for (auto &corner : marker_corners)
		{
			all_corners_cated.emplace_back(corner);
		}
		for (auto &id : marker_ids)
		{
			all_ids_cated.emplace_back(id);
		}
		marker_count_per_frame.emplace_back(marker_ids.size());
	}

	if (cali_cam == true) {
	cv::Mat cameraMatrix, distCoeffs;
		int flags = cv::CALIB_USE_INTRINSIC_GUESS;
		double repError = cv::aruco::calibrateCameraAruco(all_corners_cated, all_ids_cated, marker_count_per_frame, board, img_size, cameraMatrix, distCoeffs, rvecs, tvecs);
		std::cout << "Calibrted camera intrinsics = " << cameraMatrix << "\n. Distortion " << distCoeffs << std::endl << std::endl;
	}

	int n_valid = all_ids_valid.size();
	VERBOSE_MSG("Calibrated images [%d], all images: %d.\n", n_valid, n);

	std::ofstream f_pose(path_pose, std::ios::out);
	for (int i = 0; i < n_valid; i++){
		f_pose << all_ids_valid[i] << " ";
		for (int j = 0; j < 3; j++){
			f_pose << rvecs[i].at<double>(j, 0) << " ";
		}
		for (int j = 0; j < 3; j++) {
			f_pose << tvecs[i].at<double>(j, 0) << " ";
		}
		f_pose << "\n";
	}
	f_pose.close();
	
	std::cout << "Done.\n";
}

int main(int argc, char **argv)
{
	std::string file_ext = ".tif";
	std::string path_pose = "./pose.txt";
	int thres_code = 160;
	int nInfoLevel = 4;
	float board_width = 300;
	bool cal_cam = false;
	int board_coord_method = 1;

	int num_arguments = 2;
	std::string dir_images = ".";
	std::string path_sfm_intrin = "";
	if (argc-1 >= 2){
		dir_images = argv[1];
		nInfoLevel = std::atoi(argv[2]);
		path_pose = argv[3];
		file_ext = argv[4];
		thres_code = std::atoi(argv[5]);
		board_width = std::atof(argv[6]);
		int flag_cali_cam = std::atof(argv[7]);
		board_coord_method = std::atoi(argv[8]);
		path_sfm_intrin = argv[9];
		

		if (flag_cali_cam==1){
			cal_cam = true;
			std::cout << "Mode: Calibarate cameras.\n\n";
		}
	}
	else{
		INFO_MSG("Please input %d arguments: dir_images, nInfoLevel.\n", num_arguments);
		exit(-1);
	}
	printf("Input arguments:\nDir:%s;\n nInfoLevel:%d.\n file extension: %s. board width: %f\n", dir_images.c_str(), nInfoLevel, file_ext.c_str(), board_width);
	OPTREG::nInfoLevel = nInfoLevel; 
	
	auto  vec_path_files = IOUtils::getFileListInDirectory(dir_images, file_ext);
	std::sort(vec_path_files.begin(), vec_path_files.end());
	for (int i = 0; i < vec_path_files.size();i++) {
		std::cout << vec_path_files[i] << "\n";
	}
	Calibrate(vec_path_files, path_pose, thres_code, board_width, cal_cam, board_coord_method, path_sfm_intrin);
}