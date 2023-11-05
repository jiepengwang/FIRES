cmake_minimum_required(VERSION 3.0)

set(RegLib_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")
include_directories(${CloudLib_INCLUDE_DIR})

if(TARGET RegLib)
    message("RegLib is already defined.")

else()
    include_directories(${RegLib_INCLUDE_DIR})
    aux_source_directory(. RegLib_SRCS)

    add_library(RegLib STATIC ${RegLib_SRCS})
    target_link_libraries(RegLib CloudLib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  
                                        ${CGAL_LIBS}  ${VTK_LIBRARIES}  ${FLANN_LIBRARIES} ${BOOST_LIBRARIES})
endif()