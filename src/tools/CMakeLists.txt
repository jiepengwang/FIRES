cmake_minimum_required(VERSION 3.0)

include_directories(${CloudLib_INCLUDE_DIR})
include_directories(${RegLib_INCLUDE_DIR})

add_executable(ColorFilter ColorFilter.cpp)
target_link_libraries(ColorFilter CloudLib RegLib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  ${CGAL_LIBS}  ${VTK_LIBRARIES})

add_executable(Eval_MergeTrans Eval_MergeTrans.cpp)
target_link_libraries(Eval_MergeTrans CloudLib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  ${CGAL_LIBS}  ${VTK_LIBRARIES})

add_executable(FGR_FPFH FGR_FPFH.cpp)
target_link_libraries(FGR_FPFH CloudLib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  ${CGAL_LIBS}  ${VTK_LIBRARIES})

add_executable(FGR_PCAAlignScale FGR_PCAAlignScale.cpp)
target_link_libraries(FGR_PCAAlignScale CloudLib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  ${CGAL_LIBS}  ${VTK_LIBRARIES})

add_executable(FGR_TransformSource FGR_TransformSource.cpp)
target_link_libraries(FGR_TransformSource CloudLib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  ${CGAL_LIBS}  ${VTK_LIBRARIES})

add_executable(postprocess_calibrateMVSScale postprocess_calibrateMVSScale.cpp)
target_link_libraries(postprocess_calibrateMVSScale CloudLib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  ${CGAL_LIBS}  ${VTK_LIBRARIES})

add_executable(preprocess_Extract3DBoundaryAndRefine preprocess_Extract3DBoundaryAndRefine.cpp)
target_link_libraries(preprocess_Extract3DBoundaryAndRefine CloudLib RegLib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  ${CGAL_LIBS}  ${VTK_LIBRARIES})

add_executable(preprocess_ExtractPieces_NoSample preprocess_ExtractPieces_NoSample.cpp)
target_link_libraries(preprocess_ExtractPieces_NoSample CloudLib RegLib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  ${CGAL_LIBS}  ${VTK_LIBRARIES})

add_executable(preprocess_ExtractPieces preprocess_ExtractPieces.cpp)
target_link_libraries(preprocess_ExtractPieces CloudLib RegLib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  ${CGAL_LIBS}  ${VTK_LIBRARIES})

add_executable(preprocess_GTMatching preprocess_GTMatching.cpp)
target_link_libraries(preprocess_GTMatching CloudLib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  ${CGAL_LIBS}  ${VTK_LIBRARIES})

add_executable(preprocess_PiecesMatching preprocess_PiecesMatching.cpp)
target_link_libraries(preprocess_PiecesMatching CloudLib RegLib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  ${CGAL_LIBS}  ${VTK_LIBRARIES})

add_executable(TransformCloud TransformCloud.cpp)
target_link_libraries(TransformCloud CloudLib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  ${CGAL_LIBS}  ${VTK_LIBRARIES})

add_executable(UniformSampling UniformSampling.cpp)
target_link_libraries(UniformSampling ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  ${CGAL_LIBS}  ${VTK_LIBRARIES})
