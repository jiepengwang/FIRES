# from functools import partial
# from multiprocessing import Pool
import os
import subprocess
import shutil
import sys, glob
import numpy as np
import open3d as o3d
# from torch import maximum


from datetime import datetime
# import EvalRegionSelection as RegionSelection
import re

DIR_FILE = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(os.path.join(DIR_FILE,"../tools"))

import scripts.Utils as Utils
from scripts.postprocess_getMVSScale import calculate_pca, convert_eigenvalues2transformation, get_aabb, getMVSScale, merge_sides_one_piece, scale_icp_o3d, visualize_img_pieces
# from preprocess_denseReconstruction import DIR_BIN_PIECEREG
# from postprocess_getMVSScale import getMVSScale
import shutil
bMultiple = True
from tqdm import tqdm

# BIN_DIR= r"e:\Users\jiepeng\Desktop\PieceReg\piecereg_bin\registration"


gImageWidth = 6016
gImageHeight = 4016
nInfoLevel = 3

import scripts.path as PathConfig
PIECES_DIR_EXTRACT_TOP = "./top_pieces"
PIECES_DIR_EXTRACT_BOTTOM = "./bottom_pieces"
PIECES_DIR_MATCHING = "./pieces"



#####################################
bExtractPieces = False # don't extract pieces in RegTwoSides when doing quick verification
bPiecesMatching = True
bExtractBoundary = True
bRegSICP = True
bRegBBICP = False
bRegFGR = False
bRegICP_TwoSides = False

bCoarseReconstruction = False


# 1. segment pieces if Multiple Mode
def extractPieces(reg_dir, side):
    path_ply = reg_dir + "/" + side + ".ply"
    Utils.printSubprocessInfo("Begin preprocess_ExtractPieces_NoSample [" + path_ply +"]")
    pExtractcPieces = subprocess.Popen([PathConfig.BIN_EXTRACT_PIECES, 
                                        path_ply, 
                                        str(gImageWidth), 
                                        str(nInfoLevel)])
    pExtractcPieces.wait()

def get_pieces_center(dir_pieces_seg, trans = None):
    # get pieces number
    sub_dir = os.listdir(dir_pieces_seg)
    num_pieces = 0
    for comp in sub_dir:
        if '.ply' in comp:
            num_pieces+=1
    
    vec_path = sorted(glob.glob(os.path.join(dir_pieces_seg, f'pieceseg_**.ply')))
    centers = []
    for i in range(num_pieces):
        # piece_id = i
        # path_cloud = os.path.abspath(os.path.join(dir_pieces_seg, f'pieceseg_{piece_id}.ply'))
        path_cloud = vec_path[i]
        cloud = o3d.io.read_point_cloud(path_cloud)
        if trans is not None:
            cloud.transform(trans)
        center_ = cloud.get_center()
        centers.append(center_)
    
    return (vec_path, np.array(centers))

def matching_2sides_centers(centers_2sides):
    centers_top = centers_2sides['top'][1]
    centers_bottom = centers_2sides['bottom'][1]
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = np.linalg.norm(array - value, axis=1).argmin()
        return idx, array[idx]

    matches_top2bottom = []
    for i in range(len(centers_top)):
        center_top_i = centers_top[i].reshape(1,3)
        id_min, _ = find_nearest(centers_bottom, center_top_i)
        matches_top2bottom.append((i, id_min))
    return matches_top2bottom

def find_matches_by_board(dir_reg, board_size):
    dir_batch = dir_reg + "/.."
    centers_2sides = {}
    for side in ['top','bottom']:
        getMVSScale(dir_batch, board_size, sides=[side], check_existence=True)
        path_sfm2aruco_trans = os.path.join(dir_batch, side + '_output', 'reconstruction_sequential', f'robust_aruco_trans_sfm2aruco.txt')
        trans_sfm2aruco = np.loadtxt(path_sfm2aruco_trans)
        centers_side = get_pieces_center(os.path.join(dir_reg, side + '_pieces'), trans_sfm2aruco)
        centers_2sides[side] = centers_side
    
    matches_top2bottom = matching_2sides_centers(centers_2sides)
    os.makedirs(dir_reg + '/pieces', exist_ok=True)
    np.savetxt(dir_reg + '/pieces/matches_top2bottom.txt', np.array(matches_top2bottom), fmt="%d")

    # copy files
    b_save_files = False
    if b_save_files:
        for i in range(len(matches_top2bottom)):
            idx_top, idx_bottom = matches_top2bottom[i]

            dir_piece_i =  f'{dir_reg}/pieces/piece_{i}'
            os.makedirs(dir_piece_i, exist_ok=True)
            # np.savetxt(dir_piece_i + '/trans_top.txt', np.identity(4))
            # np.savetxt(dir_piece_i + '/trans_bottom.txt', np.identity(4))
            # path_cloud_top = os.path.abspath(os.path.join(dir_reg, 'top_pieces', f'pieceseg_{idx_top}.ply'))
            path_cloud_top = centers_2sides['top'][0][idx_top]
            shutil.copyfile(path_cloud_top, dir_piece_i + '/top.ply')

            # path_cloud_bottom = os.path.abspath(os.path.join(dir_reg, 'bottom_pieces', f'pieceseg_{idx_bottom}.ply'))
            path_cloud_bottom = centers_2sides['bottom'][0][idx_bottom]
            shutil.copyfile(path_cloud_bottom, dir_piece_i + '/bottom.ply')

    return matches_top2bottom

# merge two sides
def merge_two_sides(dir_pieces_icp, dir_pieces_sicp,  board_size):
    # Get number of fragments
    path_files_top = glob.glob(f"{dir_pieces_sicp}/**/top.ply", recursive=True)
    nPieces = len(path_files_top)
    Utils.INFO_MSG(f"***** Fragments number: {nPieces}. *************")
    
    scales_aruco = getMVSScale(dir_pieces_sicp + '/../..', sides=['top', 'bottom'], board_size= board_size, check_existence=True)
    scales_aruco_dict = {scales_aruco[0][0]:scales_aruco[0][1],
                    scales_aruco[1][0]:scales_aruco[1][1]}
    
    for i in range(nPieces):
        sides = ['top', 'bottom']
        cloud_merge = None
        for j_side in range(2):
            side_curr = sides[j_side]
            scales_to_world = scales_aruco_dict[side_curr]
            
            trans_side = np.loadtxt(f'{dir_pieces_sicp}/piece_{i}/trans_{side_curr}.txt')
            scale_1 = np.cbrt( np.linalg.det(trans_side[:3,:3])) # alphashape scale
            scales_2 = scales_to_world / scale_1 # scale to world coordinates
            trans_scale2 = np.diag([scales_2, scales_2, scales_2, 1])
            trans_all = trans_scale2 @ trans_side
            
            cloud = o3d.io.read_point_cloud(f'{dir_pieces_sicp}/piece_{i}/{side_curr}.ply')
            cloud.transform(trans_all)
            
            if side_curr=='top':
                trans_icp = np.loadtxt(f'{dir_pieces_icp}/piece_{i}/reg/trans_aligntwosides_top.txt')
                cloud.transform(trans_icp)
                cloud_merge=cloud
            else:
                cloud_merge += cloud
        
        path_cloud_world = os.path.abspath(os.path.join(dir_pieces_icp, f'piece_{i}/reg/icp_merge_world.ply'))
        o3d.io.write_point_cloud(path_cloud_world, cloud_merge)
                

def prepare_bbicp_fragments(dir_pieces_icp, dir_pieces_sicp,  board_size):
    # Get number of fragments
    path_files_top = glob.glob(f"{dir_pieces_sicp}/**/top.ply", recursive=True)
    nPieces = len(path_files_top)
    Utils.INFO_MSG(f"***** Fragments number: {nPieces}. *************")
    
    scales_aruco = getMVSScale(dir_pieces_sicp + '/../..', sides=['top', 'bottom'], board_size= board_size, check_existence=True)
    scales_aruco_dict = {scales_aruco[0][0]:scales_aruco[0][1],
                    scales_aruco[1][0]:scales_aruco[1][1]}
    
    for i in range(nPieces):
        for side_curr in ['top', 'bottom']:
            scales_to_world = scales_aruco_dict[side_curr]
            
            trans_side = np.loadtxt(f'{dir_pieces_sicp}/piece_{i}/trans_{side_curr}.txt')
            scale_1 = np.cbrt( np.linalg.det(trans_side[:3,:3])) # alphashape scale
            scales_2 = scales_to_world / scale_1 # scale to world coordinates
            trans_scale2 = np.diag([scales_2, scales_2, scales_2, 1])
            trans_all = trans_scale2 @ trans_side
            
            cloud = o3d.io.read_point_cloud(f'{dir_pieces_sicp}/piece_{i}/{side_curr}.ply')
            cloud_boundary = o3d.io.read_point_cloud(f'{dir_pieces_sicp}/piece_{i}/{side_curr}_pclbound_refine.ply')
            
            cloud.transform(trans_all)
            # voxel_size = 0.05
            # cloud = cloud.voxel_down_sample(voxel_size)
            print(len(cloud.points))
            
            cloud_boundary.transform(trans_all)
            
            Utils.ensureDirExistence(f'{dir_pieces_icp}')
            Utils.ensureDirExistence(f'{dir_pieces_icp}/piece_{i}')
            path_cloud = f'{dir_pieces_icp}/piece_{i}/{side_curr}.ply'
            # o3d.io.write_point_cloud(path_cloud, cloud)
            # o3d.io.write_point_cloud(f'{dir_pieces_icp}/piece_{i}/{side_curr}_pclbound_refine.ply', cloud_boundary)
            
            cloud =  o3d.t.geometry.PointCloud.from_legacy(cloud)
            cloud_boundary =  o3d.t.geometry.PointCloud.from_legacy(cloud_boundary)
            o3d.t.io.write_point_cloud(path_cloud, cloud, write_ascii=False, compressed=False)
            o3d.t.io.write_point_cloud(f'{dir_pieces_icp}/piece_{i}/{side_curr}_pclbound_refine.ply', cloud_boundary, write_ascii=False, compressed=False)

        
def alignTwoSidesMultiple(working_dir, board_size, matching_mode = 'contour'):
    # working_dir: registration dir
    Utils.changeWorkingDir(working_dir)
    
    # 1. Extract pieces
    ply_sides = ["top", "bottom"]   # path_ply, image_width
    if bExtractPieces:
        for side in ply_sides:
            extractPieces(working_dir, side)

    # 2. Pieces matching
    if bPiecesMatching:
        Utils.INFO_MSG('Pieces matching mode: {matching_mode}')
        if matching_mode == 'contour':
            Utils.printSubprocessInfo("Begin preprocess_PiecesMatching")
            pMatching = subprocess.Popen([PathConfig.BIN_PIECES_MATCHING, 
                                                PIECES_DIR_EXTRACT_TOP, 
                                                PIECES_DIR_EXTRACT_BOTTOM,
                                                str(nInfoLevel)])
            pMatching.wait()
        elif matching_mode == 'board':
            try:
                find_matches_by_board(working_dir, board_size)
            except Exception:
                print("Failted to get MVS scale")
            pMatching = subprocess.Popen([PathConfig.BIN_PIECES_MATCHING, 
                                                PIECES_DIR_EXTRACT_TOP, 
                                                PIECES_DIR_EXTRACT_BOTTOM,
                                                str(nInfoLevel)])
            pMatching.wait()
            # visualize_img_pieces(working_dir, working_dir + '/pieces')
        else:
            raise NotImplementedError
        # visualize_img_pieces(working_dir, working_dir + '/pieces')
        
    # 3. Extract pcl boundary
    if bExtractBoundary:
        for side in ply_sides:
            try:
                Utils.printSubprocessInfo("Begin preprocess_Extract3DBoundaryAndRefine ["+side + "]")
                pExtractBoundary = subprocess.Popen([PathConfig.BIN_EXTRACT_BOUNDARY, 
                                                        ".", 
                                                        side, 
                                                        str(gImageWidth), 
                                                        str(gImageHeight),
                                                        str(nInfoLevel),
                                                        str(bCoarseReconstruction)])
                pExtractBoundary.wait()
            except Exception:
                print(f"Fail to extact bounday:{side} {working_dir}")
                sys.stdout.flush()

    # 4. Registration
    # if bRegSICP:
    #     # Get number of fragments
    #     path_files_top = glob.glob("./pieces/**/top.ply", recursive=True)
    #     nPieces = len(path_files_top)
    #     Utils.INFO_MSG(f"***** Fragments number: {nPieces}. *************")
        
    #     pICPReg = subprocess.Popen([PathConfig.BIN_FRONT_BACK_REGISTRATION, 
    #                                     PIECES_DIR_MATCHING, 
    #                                     str(nPieces),
    #                                     str(0), # no scale in BBICP
    #                                     str(nInfoLevel)])
    #     pICPReg.wait()

    if bRegBBICP:
        # No scale to register
        path_files_top = glob.glob("./pieces/**/top.ply", recursive=True)
        num_pieces = len(path_files_top)
        Utils.INFO_MSG(f"***** Fragments number: {num_pieces}. *************")
        
        dir_pieces_matching = os.path.abspath('./pieces')
        dir_pieces_sicp = dir_pieces_matching + '_sicp'
        dir_pieces_icp = dir_pieces_matching
        
        if not Utils.check_existence(dir_pieces_sicp):
            os.rename(dir_pieces_matching, dir_pieces_matching + '_sicp')
        
        prepare_bbicp_fragments(dir_pieces_icp, dir_pieces_sicp, board_size)
        
        # registration
        args = [PathConfig.BIN_FRONT_BACK_REGISTRATION, 
                                        dir_pieces_icp, 
                                        str(num_pieces),
                                        '0', # 1 for use sicp, 0 for use icp
                                        str(nInfoLevel)]
        pICPReg = subprocess.Popen(args)
        pICPReg.wait()
                
        merge_two_sides(dir_pieces_icp, dir_pieces_sicp, board_size)                
                    




    

