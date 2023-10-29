# ----------------------------------------------------------------------------
# Heavily borrowed from
# -                   TanksAndTemples Website Toolbox                        -
# -                    http://www.tanksandtemples.org                        -
# ----------------------------------------------------------------------------

import json
import copy
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
# from datetime import datetime
# from plot import plot_graph
# import matplotlib.pyplot as plt
from cycler import cycler


def run_evaluation(reg_ply, gt_ply, out_dir, dTau, scene_name):
    scene = scene_name
    make_dir(out_dir)

    # Load reconstruction and according GT
    pcd = o3d.io.read_point_cloud(reg_ply)
    print("Source Path:",os.path.abspath(reg_ply), pcd)

    gt_pcd = o3d.io.read_point_cloud(gt_ply)
    print("GT Path:", os.path.abspath(gt_ply), gt_pcd, "\n\n")

    plot_stretch = 5
    EvaluateHisto(
        pcd,
        gt_pcd,
        np.identity(4),  # r.transformation,
        dTau / 3.0,
        dTau,
        out_dir,
        plot_stretch,
        scene,
    )


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    

def plot_graph(
    scene,
    precision11,
    recall11,
    fscore,
    dist_threshold,
    edges_source,
    cum_source,
    edges_target,
    cum_target,
    plot_stretch,
    mvs_outpath,
    show_figure=False,
):
    f = plt.figure()
    plt_size = [14, 7]
    pfontsize = "medium"

    ax = plt.subplot(111)
    label_str = "precision"
    ax.plot(
        edges_source[1::],
        cum_source * 100,
        c="red",
        label=label_str,
        linewidth=2.0,
    )

    label_str = "recall"
    ax.plot(
        edges_target[1::],
        cum_target * 100,
        c="blue",
        label=label_str,
        linewidth=2.0,
    )

    ax.grid(True)
    plt.rcParams["figure.figsize"] = plt_size
    plt.rc("axes", prop_cycle=cycler("color", ["r", "g", "b", "y"]))
    plt.title(
        "Precision:%02.2f, " % (precision11 * 100)
        + "Recall:%02.2f, "    % (recall11 * 100)
        + "F-score:%02.2f "   % (fscore * 100)
    )
    plt.axvline(x=dist_threshold, c="black", ls="dashed", linewidth=2.0)

    plt.ylabel("# of points (%)", fontsize=15)
    plt.xlabel("Meters", fontsize=15)
    plt.axis([0, dist_threshold * plot_stretch, 0, 100])
    ax.legend(shadow=True, fancybox=True, fontsize=pfontsize)
    # plt.axis([0, dist_threshold*plot_stretch, 0, 100])

    plt.setp(ax.get_legend().get_texts(), fontsize=pfontsize)

    plt.legend(loc=2, borderaxespad=0.0, fontsize=pfontsize)
    plt.legend(loc=4)
    leg = plt.legend(loc="lower right")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.setp(ax.get_legend().get_texts(), fontsize=pfontsize)
    png_name = mvs_outpath + "/PR_{0}_@d_th_0_{1}.png".format(
        scene, "%04d" % (dist_threshold * 10000)
    )
    pdf_name = mvs_outpath + "/PR_{0}_@d_th_0_{1}.pdf".format(
        scene, "%04d" % (dist_threshold * 10000)
    )

    # save figure and display
    f.savefig(png_name, format="png", bbox_inches="tight")
    f.savefig(pdf_name, format="pdf", bbox_inches="tight")
    if show_figure:
        plt.show()

def read_alignment_transformation(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
    return np.asarray(data["transformation"]).reshape((4, 4)).transpose()

def get90PercentPrecision(distance1, percentage = 0.9, sort_kind = 'heapsort'):
    distance1_cpy = np.array(distance1)
    distance1_cpy.sort(kind=sort_kind)

    size = distance1_cpy.shape[0]
    threshold = distance1_cpy[int(size*percentage)]
    
    print(f"Points size: {size}. ")
    print(f"90% percision: {threshold:.04f}")

    return threshold
    
def EvaluateHisto(
    source,
    target,
    trans,
    voxel_size,
    threshold,
    filename_mvs,
    plot_stretch,
    scene_name,
    verbose=True,
    b_plot_graph = False
):
    print("[EvaluateHisto]")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    s = copy.deepcopy(source)
    s.transform(trans)

    voxel_size = 0.02
    s = s.voxel_down_sample(voxel_size)
    s.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    

    t = copy.deepcopy(target)
    t = t.voxel_down_sample(voxel_size)
    t.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))

    distance1 = s.compute_point_cloud_distance(t)
    distance2 = t.compute_point_cloud_distance(s)
    np.save("./distance1.npy", distance1)
    np.save("./distance2.npy", distance2)
    print(f"Mean distance of S2T: {np.mean(distance1)}, max distance: {np.max(distance1)}, min distance {np.min(distance1)}, std {np.std(distance1)}.")   
    print(f"Mean distance of T2S: {np.mean(distance2)}, max distance: {np.max(distance2)}, min distance {np.min(distance2)}, std {np.std(distance2)}.")

    thres_90p = get90PercentPrecision(distance1, sort_kind='quicksort')
    

    thresholds = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    thresholds = [0.15, 0.2, 0.25]
    fwrite_eval = open(filename_mvs + "/eval_results.txt","w") 
    
    fwrite_eval.write(f"Source size: {np.array(distance1).shape[0]}; GT size: {np.array(distance2).shape[0]}\n\n")
    
    fwrite_eval.write(f"90% precision: {thres_90p:.04f}\n\n")
    
    fwrite_eval.write(f"mean  max  min std\n")   
    fwrite_eval.write(f"{np.mean(distance1):.04f} {np.max(distance1):.04f} {np.min(distance1):.04f} {np.std(distance1):.04f}\n")   
    fwrite_eval.write(f"{np.mean(distance2):.04f} {np.max(distance2):.04f} {np.min(distance2):.04f} {np.std(distance2):.04f}.\n\n")


    fwrite_eval.write(f"  T      P      R       F\n")
    for threshold in thresholds:
        print("\n")
        if threshold == 0:
            continue
        # get histogram and f-score
        [
            precision,
            recall,
            fscore,
            edges_source,
            cum_source,
            edges_target,
            cum_target,
        ] = get_f1_score_histo2(
            threshold, filename_mvs, plot_stretch, distance1, distance2, b_plot_graph = b_plot_graph
        )


        eva = [precision, recall, fscore]
        print("==============================")
        print("evaluation result :")
        print("==============================")
        print("distance tau : %.3f" % threshold)
        print("precision : %.4f" % eva[0])
        print("recall : %.4f" % eva[1])
        print("f-score : %.4f" % eva[2])
        print("==============================")
        fwrite_eval.write(f"{threshold:.02f}  {eva[0]:.04f}  {eva[1]:.04f}  {eva[2]:.04f}\n")

        # Plotting
        if b_plot_graph:
            plot_graph(
                scene_name,
                precision,
                recall,
                fscore,
                threshold,
                edges_source,
                cum_source,
                edges_target,
                cum_target,
                plot_stretch,
                filename_mvs,
            )
    
    fwrite_eval.close()


def get_f1_score_histo2(
    threshold, filename_mvs, plot_stretch, distance1, distance2, verbose=True, b_plot_graph =  True
):
    print("[get_f1_score_histo2]")
    dist_threshold = threshold
    if len(distance1) and len(distance2): 
        distance_arr_1 = np.array(distance1)
        distance_arr_2 = np.array(distance2)
        
        
        precision = float((distance_arr_1 < threshold).sum()) / float(
            len(distance1)
        )
        recall = float((distance_arr_2 < threshold).sum()) / float(
            len(distance2)
        )

        
        fscore = 2 * recall * precision / (recall + precision)
        
        if b_plot_graph:
            num = len(distance1)
            bins = np.arange(0, dist_threshold * plot_stretch, dist_threshold / 100)
            hist, edges_source = np.histogram(distance1, bins)
            cum_source = np.cumsum(hist).astype(float) / num

            num = len(distance2)
            bins = np.arange(0, dist_threshold * plot_stretch, dist_threshold / 100)
            hist, edges_target = np.histogram(distance2, bins)
            cum_target = np.cumsum(hist).astype(float) / num
        else:
            edges_source = np.array([0])
            cum_source = np.array([0])
            edges_target = np.array([0])
            cum_target = np.array([0])


    else:
        precision = 0
        recall = 0
        fscore = 0
        edges_source = np.array([0])
        cum_source = np.array([0])
        edges_target = np.array([0])
        cum_target = np.array([0])

    return [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ]
