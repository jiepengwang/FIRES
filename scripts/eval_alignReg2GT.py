import subprocess
import os
from datetime import datetime
import sys
import numpy as np
import re
import open3d as o3d
from tqdm import tqdm

DIR_FILE = os.path.dirname(os.path.abspath(__file__))

import scripts.eval_transformRegPlys as TransformPlys
import scripts.postprocess_result as ExtractEvalResult
import scripts.utils_evaluation as EvalUtils
import scripts.Utils as Utils
from scripts.postprocess_getMVSScale import scale_icp_o3d
import scripts.path as PathConfig

nInfoLevel = 4

SourcePLY_merge = "./source_pcl.ply"
TargetPLY_merge = "./target_pcl.ply"
MergePLY = "./merge.ply"
MergePLYSample = "./merge_sample0.0015.ply"


SourcePLY="./merge_sample0.0015.ply"
TargetPLYPPath="/home/hp/Desktop/gt_v3_20210127_final_PCA/piece"
SourcePLY_PCAAlignScale="./source_PCAAlignScale.ply" 

SourceFPFHBin="./source.bin"
TargetFPFHBin="./target.bin"
FGR_Trans="./fgr_trans.txt"
SourcePLY_PCAAlignScale_FGR="./source_PCAAlignScale_FGR.ply" 

SourcePLY_PCAAlignScale_FGR_SICP="./source_PCAAlignScale_FGR_SICP.ply" 
RegEval_Dir="./evaluation"

bEvalMergedCloud = True
bEvalICP = True    # True; alignment + evaluation; False, only evaluation
bEvalEachSide = True
bEvalRegistration = False


def FGR_ALIGN(cloud_source, cloud_target):
    # Step 1: Align scale
    print(f"[pFGRAlingScale] Current time: {datetime.now().time()}")
    cloud_source_align = Utils.addFileNameSuffix(cloud_source,"_align")
    pFGRAlingScale = subprocess.Popen([PathConfig.BIN_DIR + "/FGR_PCAAlignScale", cloud_source, cloud_target,  cloud_source_align])
    pFGRAlingScale.wait()


    # Step 2.1: Uniform sampling and calculate FPFH feature
    print(f"[pFGRFPFH1] Current time: {datetime.now().time()}")
    path_fpfh_source = "./sourcet.bin"
    pFGRFPFH = subprocess.Popen([PathConfig.BIN_DIR + "/FGR_FPFH", cloud_source_align,  path_fpfh_source])
    pFGRFPFH.wait()
    
    print(f"[pFGRFPFH2] Current time: {datetime.now().time()}")
    path_fpfh_target = "./target.bin"
    pFGRFPFH = subprocess.Popen([PathConfig.BIN_DIR + "/FGR_FPFH", cloud_target, path_fpfh_target])
    pFGRFPFH.wait()

    # Step 2.2: FGR alignment (Glocal)
    print(f"[pFGR] Current time: {datetime.now().time()}")
    path_fgr_trans = "./trans_fgr.txt"
    pFGR = subprocess.Popen([PathConfig.BIN_PATH_FGR, path_fpfh_target,   path_fpfh_source,    path_fgr_trans])
    pFGR.wait()

    # Step 2.3: Trans sourceform points
    print(f"[pFGRTrans] Current time: {datetime.now().time()}")
    cloud_source_align_fgr = Utils.addFileNameSuffix(cloud_source_align, "_fgr")
    pFGRTrans = subprocess.Popen([PathConfig.BIN_DIR + "/FGR_TransformSource",  cloud_source_align,  cloud_target,   path_fgr_trans,   cloud_source_align_fgr])
    pFGRTrans.wait()


    # Step 3: Scale ICP: Ground truth and reg model
    print(f"[pSICP] Current time: {datetime.now().time()}")
    cloud_source_align_fgr_sicp = Utils.addFileNameSuffix(cloud_source_align_fgr, "_sicp")
    pSICP = subprocess.Popen([PathConfig.BIN_DIR + "/RegEval",  
                                    cloud_source_align_fgr,  
                                    cloud_target, 
                                    cloud_source_align_fgr_sicp,
                                    str(nInfoLevel)])
    pSICP.wait()


#################################################################
def evaluatePiece_v2(dir_piece_reg, path_gt, name_merge = 'reg'):
    tStart = datetime.now()

    if bEvalICP:
        print(f"[pSampling] Current time: {datetime.now().time()}")
        def downsample_cloud2(path_cloud, path_cloud_downsample, voxel_size=0.4):
            pSampling = subprocess.Popen([PathConfig.BIN_DIR + "/UniformSampling", str(voxel_size), path_cloud, path_cloud_downsample])
            pSampling.wait()            
        
        # Step 2.1: Uniform sampling and calculate FPFH feature
        fpfh_bin_source =  f'{dir_piece_reg}/source_eval.bin'
        fpfh_bin_gt =  f'{dir_piece_reg}/target_eval.bin'
        path_reg_fgr = f'{dir_piece_reg}/trans_fgr_eval.txt'
        
        path_cloud_source = f'{dir_piece_reg}/icp_merge.ply'
        path_cloud_source_downsample = f'{dir_piece_reg}/{name_merge}_downsample.ply'
        path_gt_downsample = f'{dir_piece_reg}/gt_downsample.ply'
        
        voxel_size=0.3
        downsample_cloud2(path_cloud_source, path_cloud_source_downsample, voxel_size=voxel_size)
        downsample_cloud2(path_gt, path_gt_downsample, voxel_size=voxel_size)
        
        path_cloud_source_fgr= f'{dir_piece_reg}/icp_merge_fgr.ply'
        path_cloud_source_fgr_icp= f'{dir_piece_reg}/icp_merge_fgr_final.ply'
        dir_evaluation = f'{dir_piece_reg}/evaluation'
        
        print(f"[pFGRFPFH1] Current time: {datetime.now().time()}")
        pFGRFPFH = subprocess.Popen([PathConfig.BIN_DIR + "/FGR_FPFH", path_cloud_source_downsample, fpfh_bin_source])
        pFGRFPFH.wait()
        
        print(f"[pFGRFPFH2] Current time: {datetime.now().time()}")
        pFGRFPFH = subprocess.Popen([PathConfig.BIN_DIR + "/FGR_FPFH", path_gt_downsample, fpfh_bin_gt])
        pFGRFPFH.wait()

        # Step 2.2: FGR alignment (Glocal)
        print(f"[pFGR] Current time: {datetime.now().time()}")
        pFGR = subprocess.Popen([PathConfig.BIN_PATH_FGR, fpfh_bin_gt,   fpfh_bin_source,    path_reg_fgr])
        pFGR.wait()
        
        
        # transform
        def read_fgr_trans(path):
            trans = []
            with open(path,'r') as ftrans:
                lines = ftrans.readlines()
                for line in lines[1:]:
                    line = list(filter(None, re.split(' |\n', line)))
                    trans.append(line)
            trans = np.array(trans).astype(np.float)
            return trans
            
        trans_fgr = read_fgr_trans(path_reg_fgr)
        cloud_icp = o3d.io.read_point_cloud(path_cloud_source)
        cloud_icp.transform(trans_fgr)
        o3d.io.write_point_cloud(path_cloud_source_fgr, cloud_icp)

        # # o3d sicp
        cloud_source = o3d.io.read_point_cloud(path_cloud_source_fgr)
        cloud_target = o3d.io.read_point_cloud(path_gt)
        trans_sicp, sclae_sicp = scale_icp_o3d(cloud_source, cloud_target, thres_trunc=100, apply_scale = False, max_iteration=100)
        print(f"sclae_sicp: {sclae_sicp}")
        cloud_source.transform(trans_sicp)
        o3d.io.write_point_cloud(path_cloud_source_fgr_icp, cloud_source)
        
    # Step 4: Reg Evaluation
    threshold = 0.2
    print(f"[RegEval] Threshold: {threshold}. Current time: {datetime.now().time()}")
    EvalUtils.run_evaluation(reg_ply=path_cloud_source_fgr_icp,
                                gt_ply=path_gt,
                                out_dir=dir_evaluation,
                                dTau=threshold,
                                scene_name='eval')

    tEnd = datetime.now()
    print(f"Time consumed: {tEnd-tStart}")

def evalutePiecesParallel(dir_pieces_eval):
    dir_piece, path_gt = dir_pieces_eval
    # if gt_id not in [8]:
    #     continue
    tStart_g = datetime.now()

    dir_abs = os.path.abspath(dir_piece)
    Utils.INFO_MSG(f"dir_abs {dir_abs}")
    evaluatePiece_v2(dir_abs, path_gt)
    
    tEnd_g = datetime.now()
    Utils.INFO_MSG(f"[{dir} {path_gt}] Total time: {tEnd_g-tStart_g}.\n\n")

def readCorresFile(filename, dir_root_gt = '/media/ethan/HKUCS/SA2021/GTs'): # '/media/ethan/HKUCS/SA2021/GTs'): # = r'K:\SA2021\GTs'): 
#def readCorresFile(filename, dir_root_gt = r'K:\SA2021\GTs'):
    data=[]
    filename = Utils.getAbsPath(filename)
    with open(filename,'r') as f_points:
        lines = f_points.read().splitlines()
        for line in lines:
            line = line.split(" ")
            if dir_root_gt is not None:
                temp = line[1][21:]
                line[1] = f'{dir_root_gt}/{line[1][21:]}'
            data.append([line[i] for i in range(len(line))])
    # data = np.array(data)
    print("The number of read points in file {} is {}".format(filename, len(data)))

    return data

def findGTCorres(working_dir_pieces, dir_gt, path_corres, find_corres_mode="findGT"):
    Utils.INFO_MSG(f"GT matching working dir: {working_dir_pieces}")
    pCorres = subprocess.Popen([PathConfig.BIN_DIR + "/preprocess_GTMatching",  
                                working_dir_pieces,  
                                dir_gt, 
                                str(nInfoLevel),
                                path_corres,
                                find_corres_mode])
    pCorres.wait()

def getCorres(path_corres, reg_type):
    # read corres file
    corres = readCorresFile(path_corres)

    dirs_pieces_gt = []
    for i in range(len(corres)):
        path_gt = corres[i][1]
        # if 'piece13' not in path_gt:
        #     continue
        dirs_pieces_gt.append((f"./pieces/piece_{int(corres[i][0])}/{reg_type}", path_gt))

    return dirs_pieces_gt

def evaluatePieces(working_dir, dirs_pieces_eval):
    num_fragments = len(dirs_pieces_eval)
    Utils.INFO_MSG(f"************************  Fragments number: {num_fragments}  ************************ ")
    
    os.chdir(working_dir)
    if bEvalMergedCloud:
        Utils.INFO_MSG("Evaluate merged cloud")
        for dir in tqdm(dirs_pieces_eval):
            os.chdir(working_dir)
            t1 = datetime.now()
            evalutePiecesParallel(dir)
            t_cost = (datetime.now() - t1).total_seconds() / 60.0
            print(f'Cost time: {t_cost}')
        
    # transform all plys to registration
    if bEvalEachSide:
        Utils.INFO_MSG("Evaluate each side")
        Utils.changeWorkingDir(working_dir)
        TransformPlys.GT_PATH = TargetPLYPPath

        TransformPlys.alignPiecesRegData2GT(working_dir, dirs_pieces_eval)
        TransformPlys.evaluateTwoSides(working_dir, dirs_pieces_eval)
        
    if bEvalRegistration:
        TransformPlys.evaluateTwoSides_Registration(working_dir, dirs_pieces_eval)

    

