#include <iostream>
#include "PCReg.h"
#include "LogUtils.h"
#include <omp.h>
#include "IOUtils.h"
#include "CloudUtils.h"


/**
 * @brief 
 * Load clouds: (1) top->source; bottom->target. and (2) scale top cloud by bb
 * Register source to target (top->bottom)
 * scale_icp.txt: scale_icp, scale_alignscale
 */
int main(int argc, char** argv)
{
	int num_pieces = -1;
	std::string dir_pieces;
	int nInfoLevel;
	bool bUseScale = true;
	if( argc-1 >= 3){
		dir_pieces = argv[1];
		num_pieces = std::atoi(argv[2]);
		bUseScale = (std::atoi(argv[3])>0);
		nInfoLevel = std::atoi(argv[4]);
		std::cout << "The path_file_cloud is " << dir_pieces << "\n"
					<< "The number of pieces is " << num_pieces << "\n"
					<< "Use scale ICP " << bUseScale << "\n"
					<< "The nInfoLevel is " << nInfoLevel << "\n";
    }
	else{
		std::cout << "Please intut 4 arguments: path_clouds, num_pieces, bUseScale(0 or 1),  and nInfoLevel.\n";
		return -1;
	}
	OPEN_LOG("log_ICPReg");

	// Parse arguments
	ICPRegArgs reg_args;
	parseICPRegArgs(reg_args, argc, argv);

	auto path_top_files = IOUtils::findFilesInDirectories(dir_pieces, "top");
	num_pieces = path_top_files.size();
	INFO_MSG("ICPReg pieces number: %d.\n", num_pieces);

	CloudsVec clouds_reg(num_pieces);
	OPTREG::nInfoLevel = nInfoLevel;
	
	#pragma omp parallel for
	for (int i = 0; i < num_pieces; i++)
	{
		try{
			// Perform registration
			START_TIMER();
			printf("Thread %2d\n", omp_get_thread_num());
			PCReg pcReg(reg_args, 5);
			pcReg.bUseScale = bUseScale;
			std::string path_clouds_ = dir_pieces + "/piece_" + std::to_string(i);
			pcReg._path_clouds = path_clouds_;
			pcReg.loadClouds();

			std::cout << "\n\nICP with boundary constraint: " << i << "\n";
			pcReg.registerUsingICPWithBoundaryConstraint();
			
			INFO_MSG("[%d] ICPReg End (%s).", i, END_TIMER());
        }
        catch(int e){
            std::cout << "error type: " << e << "\n";
            INFO_MSG("Fail to register piece [%d].\n\n\n", i);

        }
	}
	
	CLOSE_LOG();

    return 0;
}