import os
import numpy as np

import scripts.Utils as Utils

num_framents_in_batches = np.array([9, 9, 7,   9, 8, 8,   8,9 ,9,   7, 7,4,  4,7,18])
FRAGMENTS_DATASET = np.sum(num_framents_in_batches)

PRECISION_ACC = 2

def mergeFragmentsEvalResult(dir_batch, num_fragments, skip = [], thres_completess = 0.2):
    working_dir_reg_pieces = os.path.join(dir_batch, 'registration_reso1_maskthres242/pieces')
    extractMergeResults(working_dir_reg_pieces, skip, total_fragments = num_fragments, thres_completess = thres_completess)

def parseEvalResults(path_file, thres_completess = 0.2):
    # Step 1: parse evaluation results of single pices
    handle_file = open(path_file, "r")
    i=0
    precision = 0
    recall = 0
    ind_line = 0
    mean_err = 0
    std_err = 0
    
    for line in handle_file:
        if ind_line == 2:
            # print(line)
            line = line.split(" ")
            precision = line[2][:-1]

        if ind_line == 5:
            
            line = line.split(" ")
            mean_err = line[0]
            std_err = line[3][:6]

        line_num_comp = 10 if thres_completess == 0.2 else 9 #
        if ind_line == line_num_comp:   # 14 for coarse reconstruction; 11 for fine(0.15mm), 12 for fine (0.2mm);
            line = line.split(" ")
            recall = line[4]

        ind_line +=1
    
    results = f"{precision} {float(recall)*100:.02f} {mean_err} {std_err}"
    return precision, f"{float(recall)*100:.02f}", mean_err, std_err

def extractMergeResults(working_dir_reg_pieces, skip = [], total_fragments = 9, thres_completess = 0.2, folder_eval = 'evaluation', reg_type = 'reg_icp'):
    if not Utils.check_existence(working_dir_reg_pieces):
        Utils.INFO_MSG(f'No pieces; {working_dir_reg_pieces}\n\n\n')
        return

    path_file_integrage_results = working_dir_reg_pieces + "/results_integration_comp" + f"{thres_completess}".replace(".", "_") + f"_{folder_eval}_{reg_type}.txt"
    fIntegrate = open(path_file_integrage_results, "w")

    num_fragments = 0
    p_ave = 0
    r_ave = 0
    mean_ave = 0
    std_ave = 0
    for i in range(total_fragments):
        if i in skip:
            continue

        path_results_piece = working_dir_reg_pieces + "/piece_" + str(i) + f"/{reg_type}/{folder_eval}/eval_results.txt"
        if not os.path.exists(path_results_piece):
            print(f'Error: file not existent. {path_results_piece}')
            continue

        precision, recall, mean_err, std_err = parseEvalResults(path_results_piece, thres_completess=thres_completess)
        fIntegrate.write(f"{i} &&  {precision} & {float(recall):.02f} & {mean_err} & {std_err} \n")

        p_ave += float(precision)
        r_ave += float(recall)
        mean_ave += float(mean_err)
        std_ave += float(std_err)
        num_fragments +=1
        
    if num_fragments == 0:
        return
    
    p_ave = p_ave /float(num_fragments)
    r_ave = r_ave /float(num_fragments)
    mean_ave = mean_ave /float(num_fragments)
    std_ave = std_ave /float(num_fragments)

    print(f"Fragments: {num_fragments}. {p_ave:.04f} {float(r_ave):.02f} {mean_ave:.04f} {std_ave:.04f}")
    fIntegrate.write(f"\n Average && {p_ave:.04f} &  {float(r_ave):.02f} &  {mean_ave:.04f} &  {std_ave:.04f} \n")

    fIntegrate.close()

def extractSingleSide(working_dir_reg_pieces, side = "source_pcl_trans", skip = [], total_fragments = 9):
    path_file_integrage_results = working_dir_reg_pieces + "/results_integration_" + side + '.txt'
    fIntegrate = open(path_file_integrage_results, "w")

    num_fragments = 0
    
    p_ave = 0
    r_ave = 0
    mean_ave = 0
    std_ave = 0
    for i in range(total_fragments):
        if i in skip:
            continue

        path_results_piece = working_dir_reg_pieces + "/piece_" + str(i) + "/reg/"  + side + "/evaluation/eval_results.txt"
        
        precision, recall, mean_err, std_err = parseEvalResults(path_results_piece)
        fIntegrate.write(f"{i} &&  {precision} & {float(recall):.02f} & {mean_err} & {std_err} \n")

        p_ave += float(precision)
        r_ave += float(recall)
        mean_ave += float(mean_err)
        std_ave += float(std_err)
        num_fragments +=1

    p_ave = p_ave /float(num_fragments)
    r_ave = r_ave /float(num_fragments)
    mean_ave = mean_ave /float(num_fragments)
    std_ave = std_ave /float(num_fragments)

    print(f"\nFragments: {num_fragments}. Average {p_ave:.04f} {float(r_ave):.02f} {mean_ave:.04f} {std_ave:.04f} \n")
    fIntegrate.write(f"\n  Average && {p_ave:.04f} &  {float(r_ave):.02f} &  {mean_ave:.04f} &  {std_ave:.04f} \n")

    fIntegrate.close()
