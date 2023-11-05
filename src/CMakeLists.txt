cmake_minimum_required( VERSION 3.0 )
set(ProjectName "PiecesRegistration" )
project(${ProjectName})
set(CMAKE_BUILD_TYPE Release)

if(MSVC)
	MESSAGE(STATUS "Windows system.")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
else()
	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CXX_COMPILER "g++")
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

# set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
set(CMAKE_C_FLAGS_Release "${CMAKE_C_FLAGS_Release} -O3")

# include PCL
find_package(PCL 1.10 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

if(MSVC)
	find_package(BOOST REQUIRED COMPONENTS regex asio system archive serialization filesystem)
	#find_path(BOOST_INCLUDE_DIR boost)
	include_directories(${BOOST_INCLUDE_DIR})
	link_directories(${BOOST_LIBRARY_DIRS})
	MESSAGE(STATUS "BOOST lib*** ${BOOST_LIBRARY_DIRS}. Include***: ${BOOST_INCLUDE_DIR}. Libs***: ${BOOST_LIBRARIES}")
	MESSAGE(STATUS "***BOOST Libs: ${BOOST_LIBRARIES}")
	set(BOOST_ALL_NO_LIB 1)
	add_compile_definitions(PCL_NO_PRECOMPILE)
	MESSAGE(STATUS "PCL libraries: ${PCL_LIBRARIES}")
	MESSAGE(STATUS "PCL libraries dir: ${PCL_LIBRARY_DIRS}")		
	MESSAGE(STATUS "PCL include dir: ${PCL_INCLUDE_DIRS}")		

endif()

# include OpenCV
FIND_PACKAGE(OpenCV ${SYSTEM_PACKAGE_REQUIRED})
if(OpenCV_FOUND)
	INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
	ADD_DEFINITIONS(${OpenCV_DEFINITIONS})
	SET(_USE_OPENCV TRUE)
	MESSAGE(STATUS "OpenCV ${OpenCV_VERSION} found (include: ${OpenCV_INCLUDE_DIRS})")
else()
	MESSAGE("-- Can't find OpenCV. Please specify OpenCV directory using OpenCV_DIR variable")
endif()


# include CGAL
find_package(CGAL REQUIRED)
include(${CGAL_USE_FILE})
include_directories(${CGAL_INCLUDE_DIRS})
MESSAGE(STATUS "CGAL ${CGAL_VERSION} found (include: ${CGAL_INCLUDE_DIRS})")

#include VTK
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

#Include openmp
FIND_PACKAGE(OpenMP)

add_subdirectory(libs)
add_subdirectory(PieceReg)
add_subdirectory(tools)
