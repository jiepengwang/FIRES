import os
import subprocess
from datetime import datetime
import numpy as np
import open3d as o3d

import scripts.Utils as Utils
import scripts.path as PathConfig
import scripts.utils_evaluation as EvalUtils

nInfoLevel = 4

def alignRegData2GT(working_dir, path_gt):
    """Align registration clouds to GT coordinates
    working_dir: directory of registration clouds
    """
    # dirs
    working_dir = os.path.abspath(working_dir)
    os.chdir(working_dir)

    source_pcl = "./source_pcl.ply"
    source_pcl_trans = "./source_pcl_trans.ply"

    target_pcl = "./target_pcl.ply"
    target_pcl_trans = "./target_pcl_trans.ply"

    # boudary
    source_bound_pcl = "./source_bound_pcl.ply"
    source_bound_pcl_trans = "./source_bound_pcl_trans.ply"

    target_bound_pcl = "./target_bound_pcl.ply"
    target_bound_pcl_trans = "./target_bound_pcl_trans.ply"

    # merged
    merge_pcl="./merge_sample0.0015.ply"
    merge_pcl_trans="./merge_sample0.0015_Eval_trans.ply"

    trans_PCAAlign=os.path.abspath("./trans_FGR_PCAAlignScale.txt")
    trans_FGR=os.path.abspath("./fgr_trans.txt")
    trans_SICP=os.path.abspath("./trans_RegEval.txt")


    print("working_dir: ", working_dir)
    print("trans_PCAAlign: ", trans_PCAAlign)
    plys = [(source_pcl, source_pcl_trans), 
            (target_pcl, target_pcl_trans),
            (merge_pcl, merge_pcl_trans),
            (source_bound_pcl, source_bound_pcl_trans),
            (target_bound_pcl, target_bound_pcl_trans)]
    for ply, ply_trans in plys:
        pAlignSource = subprocess.Popen([PathConfig.BIN_DIR + "/Eval_MergeTrans", ply, ply_trans,  trans_PCAAlign, trans_FGR,  trans_SICP])
        pAlignSource.wait()

def alignCloud2GT(cloud, cloud_trans, working_dir):
    trans_alphashape = working_dir + "/trans_top.txt"
    trans_sicp_aligntwosides = working_dir + "/reg/trans_aligntwosides_top.txt"
    trans_identity = working_dir + "/trans_identity_fgrformat.txt"

    cloud_temp = str(Utils.Path(cloud).parent) + "/cloud_temp.ply"
    print(cloud_temp)
    pAlignSource = subprocess.Popen([PathConfig.BIN_DIR + "/Eval_MergeTrans", cloud, cloud_temp,  trans_alphashape, trans_identity,  trans_sicp_aligntwosides])
    pAlignSource.wait()


    trans_PCAAlign = working_dir + "/reg/trans_FGR_PCAAlignScale.txt"
    trans_FGR = working_dir + "/reg/fgr_trans.txt"
    trans_SICP = working_dir + "/reg/trans_RegEval.txt"

    pAlignSource = subprocess.Popen([PathConfig.BIN_DIR + "/Eval_MergeTrans", cloud_temp, cloud_trans,  trans_PCAAlign, trans_FGR,  trans_SICP])
    pAlignSource.wait()

    # refinement between source and gt
    trans_sicp_source = working_dir + "/reg/source_pcl_trans/evaluation/trans_RegEval.txt"
    pSourceAdj = subprocess.Popen([PathConfig.BIN_DIR + "/TransformCloud", cloud_trans, trans_sicp_source,  cloud_trans])
    pSourceAdj.wait()
    os.remove(cloud_temp)

def evaluateTwoSidesParallel(dir_pieces_eval):
    dir_piece, path_gt = dir_pieces_eval

    tStart = datetime.now()
    dir_abs = os.path.abspath(dir_piece)
    target_ply = path_gt # GT_PATH + str(gt_id) + ".ply"

    sides = ["source_pcl_trans", "target_pcl_trans"]
    for side in sides:
        source_ply = dir_abs  + "/" + side  + ".ply"
        
        dir_eval =  dir_abs + "/"  + side + "/evaluation"
        source_ply_sicp = dir_eval + "/"  + side + "_eval.ply"

        Utils.ensureDirExistence(dir_eval)
        Utils.changeWorkingDir(dir_eval)

        # Step 1: Scale ICP: Ground truth and single side
        print(f"[pSICP] Current time: {datetime.now().time()}")
        pSICP = subprocess.Popen([PathConfig.BIN_DIR + "/RegEval",  
                                        source_ply,  
                                        target_ply, 
                                        source_ply_sicp,
                                        str(nInfoLevel)])
        pSICP.wait()

        # Step 2: Reg Evaluation
        threshold = 0.2
        EvalUtils.run_evaluation(reg_ply=source_ply_sicp,
                                    gt_ply=target_ply,
                                    out_dir=dir_eval,
                                    dTau=threshold,
                                    scene_name='eval')

        tEnd = datetime.now()
        print(f"Time consumed: {tEnd-tStart}")

def evaluateTwoSides(working_dir, dirs_pieces_eval):
    # change working directory
    try:
        os.chdir(working_dir)
        print(f"Current working directory is { os.getcwd()}.")
    except OSError:
        print("Cann't change current working directory.")

    num_piece = len(dirs_pieces_eval)
    Utils.INFO_MSG(f"******** Number of parallel processes: {num_piece}****************\n\n\n\n\n\n")
    from multiprocessing import Pool
    p = Pool(num_piece)
    p.map(evaluateTwoSidesParallel, dirs_pieces_eval)
    p.close()
    p.join()

def evaluateTwoSides_Registration(working_dir, dirs_pieces_eval):
    # change working directory
    try:
        os.chdir(working_dir)
        print(f"Current working directory is { os.getcwd()}.")
    except OSError:
        print("Cann't change current working directory.")
    
    side_source  = "source_pcl_trans" 
    side_target  = "target_pcl_trans"

    file_registration_results = working_dir + "/pieces/reg_result.txt"
    fRegFile = open(file_registration_results, 'w')
    fRegFile.write("Using ground truth id.\n\n")

    ave_mean = 0
    ave_std = 0

    list_means = []
    liest_stds = []

    num_pieces = len(dirs_pieces_eval)
    #for dir, gt_path_temp in dirs_pieces_eval:
    for i in range(num_pieces):
        dir_piece, path_gt = dirs_pieces_eval[i]
        
        tStart = datetime.now()
        dir_abs = os.path.abspath(os.path.join(working_dir, dir_piece))  # "./reg/"
        Utils.INFO_MSG(f"[{i}] Current working dir of registration evaluation: {dir_abs}")

        dir_eval =  dir_abs + "/"  + side_source + "/evaluation_registration"

        # Groud truth cloud for evaluation of registration
        source_ply = dir_abs + "/source_pcl_trans.ply"
        source_ply_sicp2gt = dir_abs + "/source_pcl_trans/evaluation/source_pcl_trans_eval.ply"
      
        # align to target cloud for tranformation as source cloud
        file_trans_matrix_target = dir_abs + "/target_pcl_trans/evaluation/trans_RegEval.txt"
        
        Utils.checkExistence(source_ply)
        Utils.checkExistence(source_ply_sicp2gt)
        Utils.checkExistence(file_trans_matrix_target)

        Utils.INFO_MSG(f"file_trans_matrix_target: file_trans_matrix_target")

        source_ply_align2target = Utils.addFileNameSuffix(source_ply, "_align")
        Utils.doTransformCloud(source_ply, file_trans_matrix_target, source_ply_align2target)



        # Step 2: Reg Evaluation
        threshold = 0.2
        dir_eval_reg = source_ply_align2target[:-4] + "/evaluation"
        EvalUtils.run_evaluation(reg_ply=source_ply_align2target,
                                    gt_ply=source_ply_sicp2gt,
                                    out_dir=dir_eval_reg,
                                    dTau=threshold,
                                    scene_name='eval')
                                    



        # Step 2: Compute RMSE MAE
        cloud_source = o3d.io.read_point_cloud(source_ply_align2target)
        cloud_gt = o3d.io.read_point_cloud(source_ply_sicp2gt)

        distances = cloud_source.compute_point_cloud_distance(cloud_gt)

        dist_mean = np.mean(np.array(distances))
        dist_std = np.std(np.array(distances))
        Utils.INFO_MSG(f"ID {i}: Mean Dist {dist_mean} std {dist_std}")

        fRegFile.write(f"{i} &&  {dist_mean:.04f} & {dist_std:.04f}\n")
        ave_mean += dist_mean
        ave_std += dist_std

        list_means.append(dist_mean)
        liest_stds.append(dist_std)

        tEnd = datetime.now()
        print(f"Time consumed: {tEnd-tStart}")
    
    ave_mean = ave_mean/ float(len(dirs_pieces_eval))
    ave_std = ave_std/ float(len(dirs_pieces_eval))
    fRegFile.write(f"average  && {ave_mean:.04f} & {ave_std:.04f}\n\n")
    

    fRegFile.write(f"List means   {list_means}\n\n")
    fRegFile.write(f"List stds   {liest_stds}\n\n")

    fRegFile.close()

def alignPiecesRegData2GT(working_dir, dirs_pieces_eval):
    """
    working_dir: parent path of pieces
    num_pieces: number of pieces
    """
    num_pieces = len(dirs_pieces_eval)
    for i in range(num_pieces):
        os.chdir(working_dir)
        dir = dirs_pieces_eval[i][0]
        path_gt = dirs_pieces_eval[i][1]
        alignRegData2GT(dir, path_gt)
