import argparse, os

import scripts.preprocess_denseReconstruction as DenseReconstruction
import scripts.reg_alignTwoSides as RegTwoSides
import scripts.eval_alignReg2GT as Evaluation
import scripts.Utils as Utils

from scripts.postprocess_getMVSScale import visualize_img_pieces

from datetime import datetime
import shutil

def reconstructFragments(mvs_working_dir, DIR_GT, INFO_LEVEL):
    Utils.changeWorkingDir(mvs_working_dir)
    if not Utils.checkExistence(mvs_working_dir):
        Utils.INFO_MSG("Input dir is not existent.")
        exit()

    Utils.INFO_MSG(f"MVS working dir: {mvs_working_dir}; GT dir: {DIR_GT}.")
    
    sides = [ "top", "bottom"]
    board_size = 299  # measured board size

    # ***************************************************
    mvs_working_dirs =[]
    for side in sides:
        mvs_working_dirs.append(mvs_working_dir+ "/" + side + "_output")

    Utils.INFO_MSG(f"Current working dir: {mvs_working_dir}")

    # SfM
    IMAGE_WIDTH_FULL = 6016
    IMAGE_HEIGHT_FULL = 4016
    DenseReconstruction.fFocalLength_mm = 50      # 
    DenseReconstruction.nImageWidth = IMAGE_WIDTH_FULL
    DenseReconstruction.fSensorWidth = 35.9  # D750: 35.9, D7000:23.6, cannon 22.3

    # MVS
    DenseReconstruction.nResolutionLevel = 1
    DenseReconstruction.verbosity = 2
    DenseReconstruction.nRamdomIters = 2
    DenseReconstruction.fMaskThres = int(0.95*255)            # Significant parameter, piece2: 0.95
    DenseReconstruction.fDepthDiffThreshold = 0.005
    DenseReconstruction.fNormalDiffThreshold = 25        # Significant parameter, piece2: 0.95
    DenseReconstruction.bRemoveDepthMaps = True
    DenseReconstruction.bHalfImage = "false"

    DenseReconstruction.bHeatMap = False
    DenseReconstruction.fHeatMapThres = int(0.5 * 255)             # Significant parameter

    # Registration
    REGISTRATION_DIR = mvs_working_dir + "/registration" + "_reso" + str(DenseReconstruction.nResolutionLevel) + "_maskthres" + str(int(DenseReconstruction.fMaskThres))
    RegTwoSides.gImageWidth = IMAGE_WIDTH_FULL / pow(2, DenseReconstruction.nResolutionLevel)
    RegTwoSides.gImageHeight = IMAGE_HEIGHT_FULL / pow(2, DenseReconstruction.nResolutionLevel)
    RegTwoSides.nInfoLevel = INFO_LEVEL

    
    # ***********************************************
    bPieceRecons = False
    if bPieceRecons:
        bSfM = False
        bDenseReconstruction = False
        bPrepareData = True

    bRegistration = False
    if bRegistration:
        RegTwoSides.bExtractPieces = False
        RegTwoSides.bPiecesMatching = False
        matching_mode_pieces = 'contour'  # board/contour

        RegTwoSides.bExtractBoundary = False
        RegTwoSides.bRegSICP = False
        RegTwoSides.bRegBBICP = True
        
        RegTwoSides.bRegFGR = False
        RegTwoSides.bRegICP_TwoSides = False
          
    bEvaluation = False
    if bEvaluation:
        bFindCorres = False
        find_corres_mode = "findGT"  # default: findGT or "findFrag"
        Evaluation.bEvalMergedCloud = True # global alignment + evaluation
        Evaluation.bEvalICP = True  # global alignment
        Evaluation.bEvalEachSide = False
        Evaluation.bEvalRegistration = False
    
    b_clear_cache = False   # clear cache(intermediate) files

    fwrite_time = open("./time_cost.txt", 'a+')
    T_START_GLOBAL = datetime.now()
    # Part 1
    t_sfm, t_seg, t_pre_data, t_reg = 0,0,0,0
    if bPieceRecons:
        # 1. SfM and Dense reconstruction
        if bSfM:
            t1 = datetime.now()
            Utils.INFO_MSG("\n\n\n\n\n******************************************************************")
            Utils.INFO_MSG(f"SFM: Begin at {datetime.now()}")

            DenseReconstruction.performSfMs(mvs_working_dir, sides)

            bClearSfMFiles = False
            if bClearSfMFiles:
                DenseReconstruction.clearSfMFiles(mvs_working_dir)
            t2 = datetime.now()
            t_sfm = (t2-t1).total_seconds()
            fwrite_time.write(f"SfM: {t_sfm}.\n")
            Utils.INFO_MSG(f"Consumed time: {(t2-t1).total_seconds()}")
            Utils.INFO_MSG(f"SfM: END at {datetime.now()}")

        if bDenseReconstruction:
            Utils.INFO_MSG("\n\n\n\n\n******************************************************************")
            t1 = datetime.now()
            Utils.INFO_MSG(f"[{t1}] Begin dense reconstrution")

            DenseReconstruction.performMVSs(mvs_working_dirs)
            
            t2 = datetime.now()
            t_mvs = (t2-t1).total_seconds()
            fwrite_time.write(f"MVS: {t_mvs}.\n")
            Utils.INFO_MSG(f"[{t2}] End of reconstrution. Consumed time: {(t2-t1).total_seconds()}")

        # 2. Copy related data to reg dir
        if bPrepareData:
            Utils.INFO_MSG("\n\n\n\n\n******************************************************************")
            t1 = datetime.now()
            Utils.INFO_MSG(f"[{t1}] Begin preparing reg data.\n")
            Utils.ensureDirExistence(REGISTRATION_DIR)

            dense_output_dirs = [(mvs_working_dirs[0], "top"),
                                    (mvs_working_dirs[1], "bottom")]   # dense output dir, side
            DenseReconstruction.prepareRegData(dense_output_dirs, REGISTRATION_DIR)
            t2 = datetime.now()
            t_pre_data = (t2-t1).total_seconds()
            fwrite_time.write(f"PreData: {t_pre_data}.\n")
            Utils.INFO_MSG(f"[{t2}] Finish preparing reg data. Consumed time: {(t2-t1).total_seconds()}\n")


    # Part 2. Registration
    if bRegistration:
        Utils.INFO_MSG("\n\n\n\n\n******************************************************************")
        t1 = datetime.now()
        Utils.INFO_MSG(f"[{t1}] Begin Registration.\n")
        
        RegTwoSides.nInfoLevel = INFO_LEVEL
        RegTwoSides.alignTwoSidesMultiple(REGISTRATION_DIR, board_size=board_size, matching_mode=matching_mode_pieces)

        t2 = datetime.now()
        t_reg = (t2-t1).total_seconds()
        fwrite_time.write(f"Reg: {t_reg}.\n")
        t_total = (t_sfm + t_seg + t_pre_data + t_reg)
        fwrite_time.write(f"Total time: {t_total}. Current time: {datetime.now()}.\n\n")
        Utils.INFO_MSG(f"[{t2}] End of  Registration. Consumed time: {t2-t1}\n")

        # visualization of matching
        visualize_img_pieces(REGISTRATION_DIR)
        

    T_END_GLOBAL_MVS = datetime.now()

    # Part 3 Evaluation
    if bEvaluation:
        print("\n\n\n\n\n******************************************************************.")
        t1 = datetime.now()
        Utils.INFO_MSG(f"[{t1}] Begin evaluation.\n")
        Evaluation.TargetPLYPPath = DIR_GT + "/piece"
        Evaluation.nInfoLevel = INFO_LEVEL
    
        path_corres = REGISTRATION_DIR + "/pieces/frag_gt_corres.txt"
        if bFindCorres:
            if find_corres_mode == "findFrag":
                Evaluation.findGTCorres(REGISTRATION_DIR + "/pieces", DIR_GT, path_corres, find_corres_mode)
            elif find_corres_mode == "findGT":
                Evaluation.findGTCorres(REGISTRATION_DIR + "/pieces", DIR_GT, path_corres)

        dirs_pieces_gt = Evaluation.getCorres(path_corres, reg_type='reg')
        Evaluation.evaluatePieces(REGISTRATION_DIR, dirs_pieces_gt)
        
        t2 = datetime.now()
        t_eval = t2-t1
        fwrite_time.write(f"Reg: {t_eval}.\n")
        Utils.INFO_MSG(f"[{t2}] End of  Evaluation. Consumed time: {t2-t1}\n")
    T_END_GLOBAL_EVAL = datetime.now()
    fwrite_time.close()

    Utils.INFO_MSG(f"End of pipeline. Total time (including eval): {T_END_GLOBAL_MVS-T_START_GLOBAL} ({T_END_GLOBAL_EVAL-T_START_GLOBAL}).\n")

    if b_clear_cache:
        # clear cache
        Utils.INFO_MSG('clear cache files')
        # input('Clear cache: (comment this line when want clearing cache files)')
        dirs_clear = []
        for side in ['top', 'bottom']:
            dirs_clear.append(f'{mvs_working_dir}/{side}_images')
            dirs_clear.append(f'{mvs_working_dir}/{side}_output')
            # dirs_clear.append(REGISTRATION_DIR + f"/{side}_pieces")

            # dirs_clear.append(REGISTRATION_DIR + f"/{side}_masks")
            # dirs_clear.append(REGISTRATION_DIR + f"/{side}_cameras")
            # dirs_clear.append(REGISTRATION_DIR + f"/{side}.ply")

        dirs_clear.append(REGISTRATION_DIR + "/pieces")
        dirs_clear.append(f'{mvs_working_dir}/bottom_images_b_sharp')
        for dir_remove in dirs_clear:
            if Utils.check_existence(dir_remove):
                shutil.rmtree(dir_remove)

def parseArgs():
    parser = argparse.ArgumentParser(description="Parse piecereg arguments")
    parser.add_argument("--dir_batch", type=str, required=True,help="Directory of photos of a batch of fragments")
    parser.add_argument("--dir_gt", type=str, required=True, help="Directory of GT models for accuracy evaluation")
    parser.add_argument("--info_level", type=int, required=True, help="Level of log information.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parseArgs()

    mvs_working_dir = os.path.abspath(args.dir_batch)
    DIR_GT = os.path.abspath(args.dir_gt)  # 20210425_gt_pot2_framents_num17
    INFO_LEVEL = args.info_level

    reconstructFragments(mvs_working_dir, DIR_GT, INFO_LEVEL)