import os, sys, shutil, subprocess

from pathlib import Path

import scripts.path as PathConfig
import scripts.Utils as Utils

nInfoLevel = 6

bCoarseReconstruction = False
nNumViews = 5
# SfM
fSensorWidth = 35.9
nImageWidth = 6016
SYSTEM_TYPE = Utils.getSystemType()
if SYSTEM_TYPE == "Linux":
    nNumThreads = 6
elif SYSTEM_TYPE == "Windows":
    nNumThreads = 8 # for SfM

bUseCameraExif = False

# MVS parameters
fFocalLength_mm = 50
nResolutionLevel = 3
nMaxResolution = 3008

fDepthDiffThreshold = 0.005
fNormalDiffThreshold = 25

fMaskThres = 0.8
strOutputDir = "."
verbosity = 2

# heat map
bHeatMap = True
fHeatMapThres = 215

bRemoveDepthMaps = True
bHalfImage = False

# Iterations
nRamdomIters = 6
nEstiIters = 4

def removeFiles(path_files, file_ext):
    """
    Remove files in "path_files" with extension "file_ext"
    """
    paths = Path(path_files).glob("**/*" + file_ext)
    for path in paths:
        path_str = str(path)  # because path is object not string

        if os.path.exists(path_str):	# Remove file
            print(f"Remove file {path_str}.")
            os.remove(path_str)


def denseReconstruction(bRemoveDepthMaps):
    if bRemoveDepthMaps:
        removeFiles(os.getcwd(), ".dmap")
        # removeFiles(os.getcwd(), ".cmap")

    
    path_log = "dense_reso" + str(nResolutionLevel) +"_depthdiff_" + str(fDepthDiffThreshold) + ".log"
    args_dense_reconstruction = [PathConfig.BIN_DENSE_RECONSTRUCTION, "scene.mvs",  
                                    "--resolution-level", str(nResolutionLevel), 
                                    "--max-resolution", str(nMaxResolution),
                                    "--depth-diff-threshold", str(fDepthDiffThreshold),
                                    "--normal-diff-threshold", str(fNormalDiffThreshold),
                                    # "--mask-thres", str(fMaskThres),
                                    "--random-iters", str(nRamdomIters),
                                    "--estimation-iters",str(nEstiIters),
                                    "--verbosity", str(verbosity),
                                    "--number-views", str(nNumViews)
                                    ]
    # reconstruction
    pReconstruction = subprocess.Popen(args_dense_reconstruction)
    pReconstruction.wait()
    
    # remove depth maps
    if bRemoveDepthMaps:
        removeFiles(os.getcwd(), ".dmap")
    
def performMVSs(mvs_working_dirs):
    '''SfM parameters
    fSensorWidth = 23.6
    nImageWidth = 4928
    '''
    for working_dir in mvs_working_dirs:
        # change the current working directory
        Utils.INFO_MSG(f"working_dir: {working_dir}")
        # sys.stdout.flush()
        Utils.changeWorkingDir(working_dir)
        denseReconstruction(bRemoveDepthMaps)

def clearSfMFiles(sfm_dir, sides = ["top", "bottom"]):
    for side in sides:
        dir_output = sfm_dir + "/" + side + "_output"

        dir_matches = dir_output + "/matches"
        os.removedirs(dir_matches)

        dir_reconstruction_sequential = dir_output + "/reconstruction_sequential"
        for file in os.listdir(dir_reconstruction_sequential):
            if file.endswith("Resection.ply"):
                os.remove(file)
                if nInfoLevel > 6:
                    print(f"Remove file: ./reconstruction_sequential/{file}")

def performSfMs(sfm_dir, sides = ["top", "bottom"], board_size = 300.0):
    print(f"Focal Length (mm): {fFocalLength_mm}; Sensor withd: {fSensorWidth}.")
    sys.stdout.flush()
    # for sfm_dir in sfm_dirs:
    for side in sides:
        dir_images = Utils.getAbsPath(sfm_dir + "/" + side + "_images" )
        dir_output = Utils.getAbsPath(sfm_dir + "/" + side + "_output" )
        Utils.ensureDirExistence(dir_output)
        
        dir_undistorted_images = Utils.getAbsPath(dir_output + "/undistorted_images")
        Utils.changeWorkingDir(dir_output) 
        fFocalLength_pixel = Utils.convertFocalLenthToPixels(fFocalLength_mm, nImageWidth, fSensorWidth)
        fFocalLength_pixel = 1.2 * nImageWidth

        if bUseCameraExif:
            print("Use camera exif information to estimate camera intrinsics.")
            sys.stdout.flush()
            args_sfm = ["python",  PathConfig.BIN_MVG_BUILD + "/software/SfM/SfM_SequentialPipelineMultipleCams.py", \
                                        dir_images, dir_output] 
            Utils.runSubprocess(args_sfm, "./" + Utils.getLogName("log_sfm"))
        else:
            if bCoarseReconstruction:
                Utils.INFO_MSG("Use sequential pipeline with intrinsics")
                K_cam_reso1500 = "2359.0;0;750.0;0;2359.0;500.0;0;0;1"
                args_sfm = ["python",  PathConfig.BIN_MVG_BUILD + "/software/SfM/SfM_SequentialPipeline_Intrinsic.py", \
                                        dir_images, dir_output, str(fFocalLength_pixel), str(nNumThreads), K_cam_reso1500] 

            else:
                Utils.INFO_MSG("Use sequential pipeline")
                args_sfm = ["python",  PathConfig.PATH_SFM_PIPELINE, 
                                        dir_images, dir_output, str(fFocalLength_pixel), str(nNumThreads)] 
        Utils.runSubprocess(args_sfm, "./" + Utils.getLogName("log_sfm"))
        
        if bCoarseReconstruction:
            dir_input_sfm2mvs = dir_output + "/reconstruction_sequential/sfm_data.bin"
        else:
            dir_input_sfm2mvs = dir_output + "/reconstruction_sequential/sfm_data.bin"
        
        args_sfm2mvs = [PathConfig.BIN_SFM_MVG2MVS, 
                            "-i", dir_input_sfm2mvs, 
                            "-o", dir_output + "/scene.mvs",
                            "-d", dir_undistorted_images]
        Utils.runSubprocess(args_sfm2mvs, Utils.getLogName("log_sfm2mvs"))

def prepareRegData(dense_output_dirs,export_dir, threshold_color_filter=60):
    """ prepare data for registration;
    working_dir: output dir of dense reconstruction
    export_dir: target path of registration
    side: top or bottom
    """
    for working_dir, side in dense_output_dirs:
        Utils.changeWorkingDir(working_dir)
        print(f'Export data: {side}. Working dir: {working_dir}')

         # export cameras
        export_dir_cameras = export_dir + "/" + side + "_cameras"
        Utils.ensureDirExistence(export_dir_cameras)
        pExportData = subprocess.Popen([PathConfig.BIN_EXPORT_DATA, "scene_dense.mvs", "--export-path", export_dir_cameras])
        pExportData.wait()

        # copy masks folder
        target_masks_dir = export_dir + "/" + side + "_masks"
        dir_masks =  f"{working_dir}/masks"

        if not os.path.exists(target_masks_dir):
            shutil.copytree(dir_masks, target_masks_dir)

        # copy and rename cloud
        path_cloud = working_dir + "/scene_dense.ply"

        path_cloud_cf = Utils.doColorFilter(path_cloud, threshold = threshold_color_filter)
        target_path_cloud = export_dir  + "/" + side+ ".ply"
        shutil.copy(path_cloud_cf, target_path_cloud)
        
