#pragma once
#include <vector>
#include <string>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

namespace IOUtils
{
    // File system
    std::vector<std::string> getFileListInDirectory(const std::string& files_directory, const std::string& ext_file = "");
    std::tuple<std::string, std::string, std::string> getFileParentPathStemExt(const std::string& path_file);
    std::vector<std::string> findFilesInDirectories(const std::string& path_dir, const std::string& str_search);
    std::vector<std::string> findFilesWithExtInDir(const std::string& path_dir, const std::string& file_ext);

    bool checkParentPathExistence(const std::string& dir);
    bool ensurePathExistence(const std::string& dir);
    bool ensureParentPathExistence(const std::string& dir);

    bool checkPathExistence(const std::string& path);
}