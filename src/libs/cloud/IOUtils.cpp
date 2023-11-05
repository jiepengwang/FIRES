#include "IOUtils.h"

#include <iostream>
#include "LogUtils.h"

std::vector<std::string> IOUtils::getFileListInDirectory(const std::string& files_directory, const std::string& ext_file)
{
    boost::filesystem::path bp_files(files_directory);
    if(!boost::filesystem::is_directory(bp_files)){
        std::cout << files_directory << " is not a directory.\n";
        exit(-1);
    }

    std::vector<std::string> filelist;
    boost::filesystem::directory_iterator iter_dir(bp_files), iter_dir_end;
    for (;iter_dir != iter_dir_end; iter_dir++){
        auto path = iter_dir->path();
        if(boost::filesystem::is_regular_file(path)){
            if (ext_file.empty()){   
                filelist.push_back(path.string());
                DEBUG_MSG("Find file: %s.\n", path.string().c_str());
            }// Default: Don't check
            else{
                if (path.extension().string() == ext_file){
                    filelist.push_back(path.string());
                    DEBUG_MSG("Find file: %s.\n", path.string().c_str());
                }    
            }// check file extension
        }// check whether the path is a file
        else{
            std::cout << "Find directory: " << path.string() << ".\n";
        }
    }// Path iteration
    INFO_MSG("Found files: %d; Dir: %s.\n", filelist.size(), files_directory.c_str());

    return filelist;
}


std::tuple<std::string, std::string, std::string> IOUtils::getFileParentPathStemExt(const std::string& path_file)
{
    boost::filesystem::path bp_file(path_file);

    std::string parent_path = bp_file.parent_path().string(); 
    std::string stem = bp_file.stem().string();
    std::string ext = bp_file.extension().string();

    return {parent_path, stem, ext};
}

std::vector<std::string> IOUtils::findFilesInDirectories(const std::string& path_dir, const std::string& str_search)
{
    // File name contains str_search
    boost::filesystem::path bp_dir(path_dir);
    std::vector<std::string> path_files;
    for(auto& bp : boost::filesystem::recursive_directory_iterator(bp_dir)){
        bool b_contain = boost::algorithm::equals(bp.path().stem().string(), str_search);
        if(b_contain && boost::filesystem::is_regular_file(bp.path()) ){ 
            path_files.push_back(bp.path().string());
            DEBUG_MSG("Find file: %s.\n", bp.path().string().c_str());;
        }
    }

    INFO_MSG("Files number: %d.\n", path_files.size() );
    return path_files;
}

std::vector<std::string> IOUtils::findFilesWithExtInDir(const std::string& path_dir, const std::string& file_ext)
{
    // File name contains str_search
    boost::filesystem::path bp_dir(path_dir);
    std::vector<std::string> path_files;
    for(auto& bp : boost::filesystem::recursive_directory_iterator(bp_dir)){
        bool b_contain = boost::algorithm::equals(bp.path().extension().string(), file_ext);
        if(b_contain && boost::filesystem::is_regular_file(bp.path()) ){ 
            path_files.push_back(bp.path().string());
            DEBUG_MSG("Find file: %s.\n", bp.path().string().c_str());;
        }
    }
    
    INFO_MSG("Files number: %d.\n", path_files.size() );
    return path_files;
}

bool IOUtils::checkParentPathExistence(const std::string& dir)
{
    boost::filesystem::path bp_dir(dir);
    if(!boost::filesystem::exists(bp_dir.parent_path())){
		boost::filesystem::create_directories(bp_dir.parent_path());
		std::cout << "Create the directory: " <<  bp_dir.parent_path().string() << ".\n";
	}
    return true;
}


bool IOUtils::ensureParentPathExistence(const std::string& dir)
{
    boost::filesystem::path bp_dir(dir);
    if(!boost::filesystem::exists(bp_dir.parent_path())){
		boost::filesystem::create_directories(bp_dir.parent_path());
		std::cout << "Create the directory: " <<  bp_dir.parent_path().string() << ".\n";
	}
    return true;
}

bool IOUtils::ensurePathExistence(const std::string& dir)
{
    boost::filesystem::path bp_dir(dir);
    if(!boost::filesystem::exists(bp_dir)){
		boost::filesystem::create_directories(bp_dir);
		std::cout << "Create the directory: " <<  bp_dir.string() << ".\n";
	}
    return true;
}

bool IOUtils::checkPathExistence(const std::string& path)
{
    boost::filesystem::path bp(path);
    return boost::filesystem::exists(bp);   
}
