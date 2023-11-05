#pragma once
#include <string>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>

#include <assert.h>

namespace OPTREG{
    extern FILE* pLogFile;
    extern int nInfoLevel;

    extern int image_width, image_height;   // image details

    extern int nMinPiecePoints;             // 1. segment pieces;

    extern float fNormVoxel;                // 2. normalized voxel size to uniform sample raw cloud

    extern float fNormSearchRadius;     // 3.1 boundary extraction
    extern float fSearchAngle;
    extern float nMinViews;                 // 3.2 boundary refinment
    extern float fMaxSqDist;
    extern int   nMinBoundaryPoints;

    extern float max_dist_non_boundary;

    extern int   nICPIter;                 // ICP
    extern int   fICPConvergeThres;
    extern int   nICPType;
    extern bool  bPCLBoundRefine;

    extern bool  bCoarseReconstruction;       // for quick verification during capturing process   

    extern float fAlphaShapeRadius;          
};

namespace LogUtils{
    boost::posix_time::ptime getCurrentTime();
    boost::posix_time::time_duration getElapsedTime(const boost::posix_time::ptime& tStart);
    
    int writeLog(const int& type_msg, const char *fmt, ...);
    int writeLog(const char *fmt, ...);
    
    bool openLog(const std::string& log_name);
    bool closeLog();

    template <class T>
    std::string toString(const T& val);
    std::string getUniqueLogName(const std::string& prefix);

}

template <class T>
std::string LogUtils::toString(const T& val) 
{
    std::ostringstream os;
    os << val;
    return os.str();
}


//TODO: 
#define FATAL_MSG(fmt, ...)     LogUtils::writeLog(0, fmt, ##__VA_ARGS__)
#define WARN_MSG(fmt, ...)      LogUtils::writeLog(1, fmt, ##__VA_ARGS__)
#define ERROR_MSG(fmt, ...)     LogUtils::writeLog(2, fmt, ##__VA_ARGS__)
#define INFO_MSG(fmt, ...)      LogUtils::writeLog(3, fmt, ##__VA_ARGS__)
#define DEBUG_MSG(fmt, ...)     LogUtils::writeLog(4, fmt, ##__VA_ARGS__)
#define VERBOSE_MSG(fmt, ...)   LogUtils::writeLog(6, fmt, ##__VA_ARGS__)
#define VERBOSE_MSG2(fmt, ...)   LogUtils::writeLog(7, fmt, ##__VA_ARGS__)

// Data type conversion
#define TO_CSTR(exp)            LogUtils::toString(exp).c_str()


// Log level
#define  FATAL_LEVEL      0
#define  WARN_LEVEL       1
#define  ERROR_LEVEL      2
#define  INFO_LEVEL       3
#define  DEBUG_LEVEL      4
#define  VERBOSE_LEVEL    5
#define  VISUAL_LEVEL     6


//Timer
#define GET_CURRENT_TIME()  LogUtils::toString(LogUtils::getCurrentTime()).c_str()
#define START_TIMER()       auto tStart = LogUtils::getCurrentTime()
#define UPDATE_TIMER()      tStart = LogUtils::getCurrentTime()
#define END_TIMER()         LogUtils::toString(LogUtils::getElapsedTime(tStart)).c_str()

#define OPEN_LOG(log_name)  LogUtils::openLog(log_name)
#define CLOSE_LOG()  LogUtils::closeLog()

#define ASSERT(exp)    assert(exp)