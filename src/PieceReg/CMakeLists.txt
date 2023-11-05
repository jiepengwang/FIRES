cmake_minimum_required( VERSION 3.0 )

include_directories(${CloudLib_INCLUDE_DIR})
include_directories(${RegLib_INCLUDE_DIR})

aux_source_directory(. SrcFiles)
add_executable(ICPReg ${SrcFiles})
target_link_libraries(ICPReg PUBLIC OpenMP::OpenMP_CXX CloudLib RegLib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  
                                        ${CGAL_LIBS}  ${VTK_LIBRARIES})