#include "LogUtils.h"
#include <stdio.h>
#include <stdarg.h>


namespace OPTREG{

    FILE* pLogFile = fopen("./log.txt", "w");
    
    /** \brief information level
     * FATAL,   0;
     * WARN,    1;
     * ERROR,   2;
     * INFO,    3;
     * DEBUG,   4;
     * VERBOSE, 5;
     * VISUAL,  6;
     */
    int nInfoLevel = 3;

    /**
     * image width and heigh;
     */ 
    int image_width, image_height;

    /**
     * normalized voxel size to uniform sample raw cloud
     * Default: 1.0 / 600.0
     */
    float fNormVoxel =  1.0 / 500.0; 

    /**
     * minimum points in each segmented piece;
     * Default: 10000
     */ 
    int nMinPiecePoints = 10000;       


    /** 
     * \brief search radius for extract PCL boundary 
     * box[2]/fNormSearchRadius
    */
    float fNormSearchRadius = 1.0 / 100.0;

    /** 
     * \brief search angle for extract PCL boundary 
     * Default: 100 degrees
    */
    float fSearchAngle = 100.0;


    /**
     * minum views that can see a point, for pcl boundary refinement
     * Default: 6
     */
    float nMinViews = 6;

    /**
     * maximum square dist between a point and the nearest neighbor point in a mask contour, for pcl boundary refinement
     * Default: 4
     */
    float fMaxSqDist = 4;
    /**
     * minimum number of (refined) boundary points
     */
    int nMinBoundaryPoints = 100; 

    /**
     * Iteration numbers of ICP
     * Default: 100
     */ 
    int   nICPIter = 100;

    /**
     * ICP convergement condition
     * Default: 5e-4
     */ 
    int fICPConvergeThres = 5.0e-4;

    /**
     * ICP type (SICP, point2point, point2plane)
     * Default: 8 (point2plane)
     */ 
    int nICPType = 8;

    /**
     * Whether to use refined PCL boundary or not
     * Default: true
     */ 
    bool  bPCLBoundRefine = true;        


    float max_dist_non_boundary = 9;   // square distance

    bool  bCoarseReconstruction = false;

    float fAlphaShapeRadius = 0.01;   // Radius for extracting 2D contour
};


boost::posix_time::ptime LogUtils::getCurrentTime()
{
	boost::posix_time::ptime p6 = boost::posix_time::microsec_clock::local_time();
	//std::cout << prefix <<" Current time is " << p6 << ".\n";
    return p6;
}

boost::posix_time::time_duration LogUtils::getElapsedTime(const boost::posix_time::ptime& tStart)
{
	auto tEnd = getCurrentTime();
	//std::cout << "Time elapsed: " << tEnd - tStart << ".\n";
    //auto tElapsed = tEnd - tStart;
    return (tEnd - tStart);
}

int LogUtils::writeLog(const int& type_msg, const char *fmt, ...)
{
    if (type_msg <= OPTREG::nInfoLevel){
        char bufferLog[1024];
        va_list ap;
        int n=0;
        va_start(ap, fmt);
        n=vsnprintf(bufferLog, sizeof(bufferLog), fmt, ap); 
        va_end(ap);
        printf("%s",bufferLog);
        fflush(stdout);
        fprintf(OPTREG::pLogFile, "%s", bufferLog);  // write log to file
        fflush(stdout); 
        return n; 
    }// Control information print

}

bool LogUtils::openLog(const std::string& log_name)
{
    std::string log_name_date = getUniqueLogName(log_name);
    OPTREG::pLogFile = fopen(log_name_date.c_str(), "w");
}

bool LogUtils::closeLog()
{
    fclose(OPTREG::pLogFile);
    return true;
}


std::string LogUtils::getUniqueLogName(const std::string& prefix)
{
    time_t t = time(NULL);
    auto tStruct = localtime(&t);

    char buffer[256];
    snprintf(buffer, sizeof(buffer), "%04d-%02d-%02d_%02d-%02d-%02d",
        tStruct->tm_year + 1900,tStruct->tm_mon + 1, tStruct->tm_mday,
        tStruct->tm_hour, tStruct->tm_min,tStruct->tm_sec);
    std::string strDateTime = buffer;

    std::string logFileName = prefix + "_" + strDateTime + ".txt";

    return logFileName;
}