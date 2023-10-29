import subprocess
import os
from pathlib import Path
from datetime import datetime
import open3d as o3d
import numpy as np
import sys
import beepy

import shutil
import platform


from scripts.path import BIN_DIR

FATAL_LEVEL = 0
WARN_LEVEL  = 1
ERROR_LEVEL = 2
INFO_LEVEL  = 3
DEBUG_LEVEL = 4
VERBOSE_LEVEL = 5
VISUAL_LEVEL  = 6

def getAbsPath(path):
    return os.path.abspath(path)

def beep(sound='coin', num_beep=5):
    for i in range(num_beep):
        beepy.beep(sound)

def getLogName(prefix):
    time = datetime.now()
    log_name = prefix + "_" + str(time.year) + "-" \
                            + str(time.month) + "-"   \
                            + str(time.day) + "_"   \
                            + str(time.hour) + "_"   \
                            + str(time.minute) + "_"   \
                            + str(time.second) + ".txt"
    return log_name

def check_existence(path):
    if not os.path.exists(path):
        return False
    else:
        return True
    
def writeLogFile(process_handler, path_log):
    with process_handler.stdout, open(path_log, 'wb') as file:
        for line in process_handler.stdout:
            print(line)  #NOTE: the comma prevents duplicate newlines (softspace hack)
            file.write(line)

def runSubprocess(process_args, path_log):
    # fLog = open(path_log, "a")
    pProcess = subprocess.Popen(process_args)
    #writeLogFile(pProcess, path_log)
    pProcess.wait()
    # fLog.close()


def changeWorkingDir(working_dir):
    try:
        os.chdir(working_dir)
        print(f"Current working directory is { os.getcwd()}.")
    except OSError:
        print("Cann't change current working directory.")
        sys.stdout.flush()
        exit(-1)


def ensureDirExistence(dir):
    try:
        if not os.path.exists(dir):
            INFO_MSG(f"Create directory: {dir}.")
            os.makedirs(dir)
    except Exception:
        INFO_MSG(f"Fail to create dir: {dir}")
        exit(-1)
    

def copyFile(source_path, target_dir):
    try:
        shutil.copy(source_path, target_dir)
    except Exception:
        INFO_MSG(f"Fail to copy file: {source_path}")
        exit(-1)

def removeDir(dir):
    try:
        shutil.rmtree(dir)
    except Exception as ERROR_MSG:
        INFO_MSG(f"{ERROR_MSG}.\nFail to remove dir: {dir}")
        # exit(-1)


def getSystemType():
    return platform.system()

        
def copyDir(source_dir, target_dir):
    try:
        if not os.path.exists(source_dir):
            INFO_MSG(f"source_dir {source_dir} is not exist. Fail to copy directory.")
            exit(-1)

        if not os.path.exists(target_dir):
            shutil.copytree(source_dir, target_dir)
        else:
            INFO_MSG(f"target_dir {target_dir} is already exist. Fail to copy directory.")
            exit(-1)
    except Exception as ERROR_MSG:
        INFO_MSG(f"{ERROR_MSG}.\nFail to copy file: {source_path}")
        exit(-1)

def checkExistence(dir):
    if not os.path.exists(dir):
        # INFO_MSG(f"Dir is not exist: {dir}")
        return False
    else:
        return True


def convertFocalLenthToPixels(f_mm, image_width, sensor_width):
    f_pixels = image_width * f_mm / sensor_width
    return f_pixels


def removeFiles(path_files, file_ext, nInfoLevel = 3):
    """
    Remove files in "path_files" with extension "file_ext"
    """
    INFO_MSG(f"Remove files in {path_files} with extension {file_ext}.")
    paths = Path(path_files).glob("**/*" + file_ext + "*")
    count = 0
    for path in paths:
        path_str = str(path)  # because path is object not string

        if os.path.exists(path_str):	# Remove file
            if nInfoLevel >= VERBOSE_LEVEL:
                print(f"Remove file {path_str}.")
            os.remove(path_str)
            count += 1
        # else:
        #     print(f"File {path_str} doesn't exist."
    
    INFO_MSG(f"Totall remove {count} files")


def printSubprocessInfo(msg):
    print("\n\n\n******************************************************************.")
    print(msg)


def addFileNameSuffix(path_file, suffix):
    ppath, stem, ext = getPathComponents(path_file)
    
    path_name_new = ppath + "/" + stem + str(suffix) + ext
    print(f"File name with suffix: {path_name_new}")
    return path_name_new

def addFileNamePrefix(path_file, prefix):
    ppath, stem, ext = getPathComponents(path_file)
    
    path_name_new = ppath + "/" + str(prefix) + stem  + ext
    print(f"File name with prefix: {path_name_new}")
    return path_name_new

# open3d related functions
def getCloudDensity(path_cloud):
    cloud = o3d.io.read_point_cloud(path_cloud)
    distances = cloud.compute_nearest_neighbor_distance()
    dist_mean = np.mean(distances)
    return dist_mean

def getPathComponents(path):
    path = Path(path)
    ppath = str(path.parent)
    stem = str(path.stem)
    ext = str(path.suffix)
    return ppath, stem, ext


def readLines(filename):
    data=[]
    with open(filename,'r') as f_points:
        lines = f_points.read().splitlines()
        for line in lines:
            line = line.split(" ")
            data.append([float(line[i]) for i in range(len(line))])
    data = np.array(data)
    # print("File: {}; Points: {}".format(filename, data.shape[0]))

    return data

def readLinesWithStr(filename):
    data=[]
    with open(filename,'r') as f_points:
        lines = f_points.read().splitlines()
        for line in lines:
            # line = line.split(" ")
            data.append(line)
    # data = np.array(data)
    print("File: {}; Lines: {}".format(filename, len(data)))

    return data

def getSubfoldersNum(dir):
    list_subfolders_with_paths = [f.path for f in os.scandir(dir) if f.is_dir()]
    return list_subfolders_with_paths

def INFO_MSG(msg):
    print(msg)
    sys.stdout.flush()

def ensureAbsPath(path):
    return os.path.abspath(path)

def getSeconds(t_delta):
    seconds = t_delta.total_seconds()
    return seconds

def getSeconds(t_start, t_end):
    t_delta = t_end - t_start
    seconds = t_delta.total_seconds()
    return seconds

def getCurrentTime():
    str_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return str_time



## from tools
def doColorFilter(path_input, threshold, bRemoveOutliers = True):
    ## remove black(background) points
    ppath, stem, ext = getPathComponents(path_input)

    try: 
        path_output_inliers = ppath + "/" + stem + "_cf" + str(threshold) + ext
        path_output_outliers = ppath + "/" + stem + "_cf" + str(threshold) + "_outliers" + ext
        pColorFilter = subprocess.Popen([BIN_DIR+"/ColorFilter", path_input, 
                                                                    path_output_inliers, 
                                                                    path_output_outliers, 
                                                                    str(threshold)])
        pColorFilter.wait()
        if bRemoveOutliers:
            os.remove(path_output_outliers)
        return path_output_inliers
    except Exception:
        print("Fail to run ColorFilter.")
        sys.stdout.flush()
        return path_output_inliers


def doTransformCloud(path_cloud, path_trans, path_save):
    try:
        pTrans = subprocess.Popen([BIN_DIR+"/TransformCloud", path_cloud, path_trans, path_save])
        pTrans.wait()
        return True
    except Exception:
        print("Fail to run TransformCloud")
        exit(-1)
    
def doUniformSampling(path_cloud, path_save, voxel_leaf):
    try:
        pTrans = subprocess.Popen([BIN_DIR+"/UniformSampling", str(voxel_leaf), path_cloud, path_save])
        pTrans.wait()
        return True
    except Exception:
        print("Fail to run UniformSampling")
        exit(-1)

def addFileNameSuffix(path_file, suffix):
    ppath, stem, ext = getPathComponents(path_file)
    
    path_name_new = ppath + "/" + stem + str(suffix) + ext
    print(f"File name with suffix: {path_name_new}")
    return path_name_new


