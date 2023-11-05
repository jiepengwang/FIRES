cmake_minimum_required(VERSION 3.0)

set(CloudLib_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")

if(TARGET CloudLib)
    message("CloudLib is already defined.")

else()
    include_directories(${CloudLib_INCLUDE_DIR})
    aux_source_directory(. CloudLib_SRCS)

    add_library(CloudLib STATIC ${CloudLib_SRCS})
    target_link_libraries(CloudLib PUBLIC OpenMP::OpenMP_CXX ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  
                                        ${CGAL_LIBS}  ${VTK_LIBRARIES} ${FLANN_LIBRARIES})
endif()