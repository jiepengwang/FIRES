import trimesh,os,re
import glob,copy,cv2

import cv2 as cv
import open3d as o3d
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pathlib import Path

import scripts.Utils as Utils

from scripts.path import DIR_BIN_PIECEREG
BIN_PATH = f"{DIR_BIN_PIECEREG}/postprecess/calibrate_mvs_scale/postprocess_calibrateMVSScale"

nInfoLevel = 3

def get_path_components(path):
    path = Path(path)
    ppath = str(path.parent)
    stem = str(path.stem)
    ext = str(path.suffix)
    return ppath, stem, ext

def read_poses_txt(path_txt):
    fTxt = open(path_txt, "r")
    lines = fTxt.readlines()
    stems = []
    poses = []
    for line in lines:
        line = line.split(' ')
        stems.append(line[0])
        poses.append(np.array(line[1:-1]).astype(np.float32))
    return stems, np.array(poses)

def get_files_stem(dir, ext_file):
    '''Get stems of all files in directory with target extension
    Return:
        vec_stem
    '''
    vec_path = sorted(glob.glob(f'{dir}/**{ext_file}'))
    vec_stem = []
    for i in range(len(vec_path)):
        pparent, stem, ext = get_path_components(vec_path[i])
        vec_stem.append(stem)
    return vec_stem

def getMVSScale(work_dir, board_size, sides=['bottom', 'top'], check_existence = False):
    # @return: (side,scale)

    scales = []
    for side in sides:
        img_dir = os.path.join(work_dir, side + '_output', 'undistorted_images')
        if not Utils.check_existence(img_dir):
            img_dir = os.path.join(work_dir, side + '_images')

        ext_img = '.tif'

        use_gray_imgs = True
        if use_gray_imgs:
            img_dir = os.path.join(work_dir, side + '_images_b_sharp')
            ext_img = '.png'

            if not Utils.check_existence(img_dir) or (len(glob.glob(f'{img_dir}/*.png'))<48):
                get_gray_images(os.path.join(work_dir, side + '_images'), img_dir)
        
        os.makedirs(os.path.join(work_dir, side + '_output'), exist_ok=True)
        cal_camera = 0 # 1 for true, o for false
        board_coord = 1
        thres_color = 60    
        
        use_sfm_intrin = 1
        if use_sfm_intrin:
            path_sfm_intrin = f'{work_dir}/{side}_output/intrinsics_sfm.txt'
            if not Utils.check_existence(path_sfm_intrin):
                path_KRC = f'{work_dir}/registration_reso1_maskthres242/{side}_cameras/KRC.txt'
                intrin_sfm = parse_cameras_intrinsics(path_KRC, reso=0)
                np.savetxt(path_sfm_intrin, intrin_sfm, fmt='%f')
        else:
            path_sfm_intrin = ''

        path_pose = os.path.join(work_dir, side + '_output', f'pose_aruco_calcam{cal_camera}_board{board_coord}_thres{thres_color}_sfm{use_sfm_intrin}_autoboard_icp.txt')
        if not ( check_existence and Utils.check_existence(path_pose)):
            os.system(f'{BIN_PATH} {img_dir} {nInfoLevel} {path_pose} {ext_img} {thres_color} {board_size} {cal_camera} {board_coord} {path_sfm_intrin}')
        
        
        stems, poses = read_poses_txt(path_pose)
        idx_sort = np.argsort(np.array(stems))
        stems2 = np.array(stems)[idx_sort]
        poses2 = poses[idx_sort]
        poses=poses2
        
        n = len(poses)

        rvecs = poses[:, :3]
        tvecs = poses[:, 3:]
        rot_mat = np.zeros([n, 3, 3])

        positions = []
        trans_all = []
        for i in range(n):
            rot_vec_t = rvecs[i].reshape(3,1)
            rot_mat_t = rot_mat[i]
            out, jacob = cv.Rodrigues(rot_vec_t, rot_mat_t)
            rot_mat[i] = np.linalg.inv(out)
            positions.append(np.squeeze(rot_mat[i] @ tvecs[i, :, None]))

            trans = np.identity(4)
            trans[:3, :3] = out
            trans[:3, 3] = tvecs[i]
            trans_all.append(trans)
        trans_all = np.array(trans_all).reshape(-1, 16)
        np.savetxt(os.path.join(work_dir, side + '_output', f'{side}_extrinsics_cameras_aruco.txt'), trans_all)

        positions = -np.stack(positions, 0)
        
        # sfm_data = trimesh.load(os.path.join(work_dir, side + '_output', 'reconstruction_sequential', 'robust.ply'))
        path_sfm = os.path.join(work_dir, side + '_output', 'reconstruction_sequential', 'robust.ply')
        os.makedirs(os.path.join(work_dir, side + '_output', 'reconstruction_sequential'), exist_ok=True)

        if not os.path.exists(path_sfm):
            generate_sparse_ply(work_dir, side)
        
        path_cam_aruco = os.path.join(work_dir, side + '_output', 'reconstruction_sequential', f'robust_aruco_calcam{cal_camera}_board{board_coord}.ply')
        
        board_points = []
        scale_board = 300/4025
        for idx in range(21):
            for idy in range(21):
                board_points.append([idx*200.0*scale_board, idy*200.0*scale_board, 0])
        board_points = np.array(board_points)
        
        pts_ply = np.concatenate([positions, board_points], axis=0)
        header = f'''ply
format ascii 1.0
comment VCGLIB generated
element vertex {len(pts_ply)}
property float x
property float y
property float z
element face 0
property list uchar int vertex_indices
end_header'''

                
        np.savetxt(path_cam_aruco, pts_ply, header=header,  delimiter=' ', comments='')
        
        # get trans from sfm to aruco
        trans_, scale_ = register_sfmCam_to_arucoCam(os.path.join(work_dir, side + '_output', 'reconstruction_sequential'), path_cam_aruco)
        path_trans_sfm2aruco = os.path.join(work_dir, side + '_output', 'reconstruction_sequential', f'robust_aruco_trans_sfm2aruco.txt')
        np.savetxt(path_trans_sfm2aruco, trans_)
        path_sfm2aruco_scale = os.path.join(work_dir, side + '_output', 'reconstruction_sequential', f'robust_aruco_sfm2aruco_scale.txt')
        np.savetxt(path_sfm2aruco_scale, np.array([scale_]))

        scales.append((side, scale_))
        Utils.INFO_MSG(f"Scale {side} to ArUco board: {scale_}")        

    return scales  # (side,scale)

def read_scales_txt(path):
    fTxt = open(path, "r")
    lines = fTxt.readlines()
    scales = []
    for line in lines:
        line = re.split('; | |\*|\n', line)
        scales.append(np.array(line[:-1]).astype(np.float32))
    return np.array(scales)
    
def calibrate_mvs_clouds_scale(dir_reg, board_size, sfm_mode = 'openMVG'):
    '''Align clouds to real scale. Calibrate MVS scale by Aruco size and scale pieces
    Unit of clouds: cm
    '''
    dir_batch = os.path.join(dir_reg, "..")
    side_scale_cal = 'bottom'
    scale_aruco = getMVSScale(dir_batch, board_size, sides=[side_scale_cal])[0][1]
    dir_pieces = f'{dir_reg}/pieces'
    sub_dir = os.listdir(dir_pieces)
    
    num_pieces = 0
    for comp in sub_dir:
        if 'piece_' in comp:
            num_pieces+=1
    
    print(f'There are {num_pieces} pieces')
    
    # scale clouds
    IDS_fail = []
    for i in range(num_pieces):
        # piece_id, scale_curr_alphashape = scales_pieces_target[i]
        piece_id = i
        
        path_reg = os.path.join(dir_pieces, f'piece_{piece_id}/reg/icp_merge.ply')
        if not Utils.checkExistence(path_reg):
            Utils.INFO_MSG(f'No reg file: {path_reg}')
            continue
        
        scale_bottom_reg, scale_top_reg = merge_sides_one_piece(os.path.join(dir_pieces, f'piece_{piece_id}'))
        path_cloud = os.path.abspath(os.path.join(dir_pieces, f'piece_{piece_id}/reg/icp_merge_o3d.ply'))
        
        # check existence
        if not Utils.check_existence(path_cloud):
            Utils.INFO_MSG(f'Failed piece: {path_cloud}')
            continue
        
        path_cloud_world = os.path.abspath(os.path.join(dir_pieces, f'piece_{piece_id}/reg/icp_merge_world.ply'))
        path_trans =  os.path.abspath(os.path.join(dir_pieces, f'piece_{piece_id}/reg/icp_merge_trans_world.txt'))

        if not os.path.exists(path_cloud):
            Utils.INFO_MSG(f'Cloud is not existent. Fail to scale cloud: {path_cloud}.')
            IDS_fail.append(path_cloud)
            continue
        # scale_curr = scale_aruco / scale_bottom_reg
        if side_scale_cal == 'top':
            scale_curr = scale_aruco / scale_top_reg
        else:
            scale_curr = scale_aruco / scale_bottom_reg
        if sfm_mode == 'aruco':
            scale_curr = 1.0 / scale_bottom_reg
        Utils.INFO_MSG(f'Scale: {scale_curr}. Scale Aruco: {scale_aruco}; Scale of alphashape: {scale_bottom_reg}.')
        scale_mat = np.diag([scale_curr, scale_curr, scale_curr, 1.0])
        np.savetxt(path_trans, scale_mat)
        # Utils.doTransformCloud(path_cloud, path_trans, path_cloud_world)
        cloud = o3d.io.read_point_cloud(path_cloud)
        cloud.transform(scale_mat)
        o3d.io.write_point_cloud(path_cloud_world, cloud)
        if len(IDS_fail)>0:
            Utils.INFO_MSG(f'Error: there are failed reg clouds.')
            path_fail_scale = os.path.join(dir_reg, 'log_calibrate_mvs_scale_fail.txt')
            with open(path_fail_scale, 'w') as fscale:
                for idx_fail in range(len(IDS_fail)):
                    fscale.write(f"{IDS_fail[idx_fail]}\n")
    return scale_aruco
        
def merge_sides_one_piece(dir_piece):
    cloud_top = o3d.io.read_point_cloud(os.path.join(dir_piece, 'top.ply'))
    cloud_bottom = o3d.io.read_point_cloud(os.path.join(dir_piece, 'bottom.ply'))
    
    trans_top_alpha = np.loadtxt(os.path.join(dir_piece, 'trans_top.txt'))
    trans_top_icp = np.loadtxt(os.path.join(dir_piece, 'reg',  'trans_aligntwosides_top.txt'))
    trans_top = trans_top_icp @ trans_top_alpha
    cloud_top.transform(trans_top)
    scale_top = np.cbrt(np.linalg.det(trans_top[:3,:3]))
    
    trans_bottom = np.loadtxt(os.path.join(dir_piece, 'trans_bottom.txt'))
    cloud_bottom.transform(trans_bottom)
    scale_bottom = np.cbrt(np.linalg.det(trans_bottom[:3,:3]))
    
    cloud_bottom += cloud_top
    o3d.io.write_point_cloud(os.path.join(dir_piece, 'reg', 'icp_merge_o3d.ply'), cloud_bottom)
    return scale_bottom, scale_top

def readCamerasInfo(filepath):
    fcam = open(filepath, 'r')
    lines = fcam.readlines()
    num_cams = len(lines) // 13
    cams_R_all, cams_C_all = [], []
    for i in range(num_cams):
        lines_cur = lines[i*13:(i+1)*13]

        cam_K = []
        for j in range(3,6):
            line = re.split('\[|\]|;| |,|\n', lines_cur[j])
            line = list(filter(None, line))

            cam_K.append(line)
        
        cam_R = []
        for j in range(7,10):
            line = re.split('\[|\]|;| |,|\n', lines_cur[j])
            line = list(filter(None, line))

            cam_R.append(line)
        cams_R_all.append(cam_R)

        line = re.split('\[|\]|;| |,|\n', lines_cur[11])
        cam_C = list(filter(None, line))
        cams_C_all.append(cam_C)
    
    cams_R_all = np.array(cams_R_all).astype(np.float32)
    cams_C_all = np.array(cams_C_all).astype(np.float32)

    cam_K_homo = np.identity(4)
    cam_K = np.array(cam_K, dtype=np.float32)
    cam_K_homo[:3,:3] = cam_K

    trans = np.array([np.identity(4)]*num_cams)
    trans[:,:3, 3] = -(cams_R_all @ cams_C_all[...,None]).squeeze()
    trans[:,:3, :3] = cams_R_all
    # trans = cam_K_homo[None, ...] @ trans

    return cam_K_homo, trans


def read_cloud(path_cloud, path_img, cam_K,  trans):
    # name_clouds = sorted(os.listdir(dir_cloud))
    count_clouds = 0

    clouds = o3d.io.read_point_cloud(path_cloud)
    # clouds = o3d.io.read_point_cloud(os.path.join(dir_cloud, f'piece_{count_clouds}.ply'))
    point0 = np.array(clouds.points)[0]
    # print(point0)

    points0_homo = np.ones(4)
    points0_homo[:3] = point0

    points_img = cam_K @ trans @ points0_homo[:, None]
    points_img = points_img[:2] / points_img[2]

    org = (int(points_img[0]), int(points_img[1]) )
    return org



def reproject_pieces_to_img(path_img_cord, dir_reg, side='bottom', img_id = 17):

    vec_imgs = sorted(glob.glob(f'{dir_reg}/../{side}_images/*.tif'))
    path_img = vec_imgs[img_id]

    dir_pieces = os.path.join(dir_reg, 'pieces_sicp')
    folders_name = os.listdir(dir_pieces)
    count_pieces = 0
    for name in folders_name:
        if 'piece_' in name:
            count_pieces+=1
    
    # read cameras and do reprojection
    path_cams = os.path.join(dir_reg, f'{side}_cameras', 'KRC.txt')
    cam_K, trans_all = readCamerasInfo(path_cams)  

    img = cv.imread(path_img)
    img = cv.resize(img, (3008, 2008))
    for i in range(count_pieces):
        path_cloud = os.path.join(dir_pieces, f'piece_{i}', f'{side}.ply')
        org = read_cloud(path_cloud, path_img, cam_K, trans_all[img_id])

        fontScale=3
        color=(0,0,255)
        thickness=5
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img, f'{i}', org, font,fontScale, color, thickness, cv.LINE_AA)
    
    if path_img_cord is not None:
        cv.imwrite(path_img_cord, img)
    return img

def visualize_img_pieces(dir_reg, dir_save = None, imd_id = 17):
    print("Build correspondences between pieces and images")
    imgs = []
    for side in ['top', 'bottom']:
        img = reproject_pieces_to_img(None, dir_reg, side, imd_id)
        imgs.append(img)
    img_cat = np.concatenate([imgs[0], np.ones((img.shape[0], 20, 3))*255, imgs[1]], axis=1)
    Utils.ensureDirExistence(os.path.join(dir_reg, 'final_output'))
    if dir_save is not None:
        cv.imwrite(os.path.join(dir_save, 'corres_img_pieces.png'), img_cat)  
    else:
        cv.imwrite(os.path.join(dir_reg, 'final_output', '00_corres_img_pieces.png'), img_cat)  
       

def scale_icp_o3d(cloud_source, cloud_target, thres_trunc = 500, apply_scale = True, max_iteration=100):
    # source = o3d.io.read_point_cloud(path_mvs)
    # target = o3d.io.read_point_cloud(path_board)
    # o3d.io.write_point_cloud('./test1.ply', cloud_source)
    # o3d.io.write_point_cloud('./test2.ply', cloud_target)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        cloud_source, cloud_target, thres_trunc, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(apply_scale),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    rotation = reg_p2p.transformation[:3,:3]
    det_rotate = np.linalg.det(rotation)
    scale = np.cbrt(det_rotate)
    Utils.INFO_MSG(f"RMSE: {reg_p2p.inlier_rmse}")
    # det_rotate2 = np.linalg.det(np.identity(3)*5)

    # scale2 = np.cbrt(det_rotate2)
    return reg_p2p.transformation, scale

def parse_sfm_sparse_cloud(path_cloud, num_cameras):
    sfm_data = trimesh.load(path_cloud)

    # for i in range(num_cameras):
    #     assert(np.all(sfm_data.colors[i] == [0, 255, 0, 255]))
    points_cam = sfm_data.vertices[:num_cameras]
    points_sparse = sfm_data.vertices[num_cameras:]
    return sfm_data, points_cam, points_sparse
    

def calculate_pca(data, correlation=False, sort=True):
    '''
    Return:
        eigen_vectors: i-th vector, [:,i]
    '''
    # https://blog.csdn.net/eurus_/article/details/113688237
    # data: N*3
    average_data = np.mean(data,axis=0)       #求 NX3 向量的均值
    decentration_matrix = data - average_data   #去中心化
    H = np.dot(decentration_matrix.T,decentration_matrix)  #求解协方差矩阵 H
    eigenvectors, eigenvalues, eigenvectors_T = np.linalg.svd(H)    # SVD求解特征值、特征向量

    if sort:
        sort = eigenvalues.argsort()[::-1]      #降序排列
        eigenvalues = eigenvalues[sort]         #索引
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors, average_data

def convert_eigenvalues2transformation(eigen_vectors, center):
    # convert eigen values to transformatino
    eigen_vectors = np.array(eigen_vectors)
    eigen_vectors[:, 1] = np.cross(eigen_vectors[:, 2], eigen_vectors[:, 0])
    eigen_vectors[:, 0] = np.cross(eigen_vectors[:, 1], eigen_vectors[:, 2])
    eigen_vectors[:, 2] = np.cross(eigen_vectors[:, 0], eigen_vectors[:, 1])
    norm  = np.linalg.norm(eigen_vectors, axis=0, keepdims=True)
    eigen_vectors = eigen_vectors / norm

    trans_pca2world = np.identity(4)
    trans_pca2world[:3,:3] = eigen_vectors
    trans_pca2world[:3, 3] = center # points_cam.mean(axis=0)
    trans_world2pca = np.linalg.inv(trans_pca2world)
    return trans_world2pca

def convert_points_to_o3d_cloud(points):
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(points)
    return geometry

def get_aabb(points, scale=1.0):
    '''
    Args:
        points; 1) numpy array (converted to '2)'; or 
                2) open3d cloud
    Return:
        min_bound
        max_bound
        center: bary center of geometry coordinates
    '''
    if isinstance(points, np.ndarray):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        points = point_cloud
    min_max_bounds = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(points)
    min_bound, max_bound = min_max_bounds.min_bound, min_max_bounds.max_bound
    center = (min_bound+max_bound)/2
    # center = points.get_center()
    if scale != 1.0:
        min_bound = center + scale * (min_bound-center)
        max_bound = center + scale * (max_bound-center)

    # logging.info(f"min_bound, max_bound, center: {min_bound, max_bound, center}")
    box_size = np.linalg.norm(max_bound - min_bound)
    return min_bound, max_bound, center,  box_size

def get_rough_scale(points_cam_sfm, point_cam_aruco):
    box_sfm = get_aabb(points_cam_sfm)[-1]
    box_aruco = get_aabb(point_cam_aruco)[-1]
    return box_aruco/box_sfm


def align_cam_points(data_source, data_target, sort=True):
    # data_source = data_source[:,:2]
    # data_target = data_target[:,:2]

    average_data_sorce = np.mean(data_source,axis=0)       #求 NX3 向量的均值
    decentration_matrix_source = data_source - average_data_sorce   #去中心化

    average_data_target = np.mean(data_target,axis=0)       #求 NX3 向量的均值
    decentration_matrix_target = data_target - average_data_target   #去中心化

    H = np.dot(decentration_matrix_source.T,decentration_matrix_target)  #求解协方差矩阵 H
    # H = np.dot(decentration_matrix_target.T,decentration_matrix_source)  #求解协方差矩阵 H
    eigen_vectors, eigenvalues, eigenvectors_T = np.linalg.svd(H)    # SVD求解特征值、特征向量

    eigen_vectors = eigenvectors_T.transpose() @ eigen_vectors.transpose()

    # if sort:
    #     sort = eigenvalues.argsort()[::-1]      #降序排列
    #     eigenvalues = eigenvalues[sort]         #索引
    #     eigen_vectors = eigen_vectors[:, sort]

    # convert eigen values to transformatino
    if np.linalg.det(eigen_vectors) < 0:
        eigen_vectors = -eigen_vectors
        print('Error in determinant of align cameras.')
        # exit(-1)
    
    # eigen_vectors2 = np.identity(3)
    # eigen_vectors2[:2,:2] = eigen_vectors
    # eigen_vectors = eigen_vectors2

    eigen_vectors[:, 1] = np.cross(eigen_vectors[:, 2], eigen_vectors[:, 0])
    eigen_vectors[:, 0] = np.cross(eigen_vectors[:, 1], eigen_vectors[:, 2])
    eigen_vectors[:, 2] = np.cross(eigen_vectors[:, 0], eigen_vectors[:, 1])
    norm  = np.linalg.norm(eigen_vectors, axis=0, keepdims=True)
    eigen_vectors = eigen_vectors / norm

    trans_pca2world = np.identity(4)
    trans_pca2world[:3,:3] = eigen_vectors
    trans_pca2world[:3, 3] = (np.mean(data_target,axis=0).reshape(3,1) - eigen_vectors @ np.mean(data_source,axis=0).reshape(3,1)).squeeze()
    return trans_pca2world

def register_sfmCam_to_arucoCam(dir_sfm_seq, path_cloud_aruco):
    # parse sfm sparse point cloud
    path_cloud_sfm = os.path.join(dir_sfm_seq, 'robust.ply')

    cloud_sfm_o3d = o3d.io.read_point_cloud(path_cloud_sfm)
    cloud_aruco_o3d = o3d.io.read_point_cloud(path_cloud_aruco)
    points_cam_aruco, points_board_aruco = np.array(cloud_aruco_o3d.points)[:48], np.array(cloud_aruco_o3d.points)[48:]

    _, points_cam, points_sparse = parse_sfm_sparse_cloud(path_cloud_sfm, num_cameras=48)
    dir_board2cam = points_cam.mean(axis=0) - points_sparse.mean(axis=0)
    dir_board2cam = dir_board2cam / np.linalg.norm(dir_board2cam)

    # 1. PCA: based on sparse points of the board
    eigen_values, eigen_vectors, center_sparse = calculate_pca(points_sparse)
    # print(eigen_vectors, '\n',)

    angle_z = np.arccos( dir_board2cam.dot(eigen_vectors[:,2]) / (np.linalg.norm(dir_board2cam) * np.linalg.norm(eigen_vectors[:,2])))
    angle_z = np.rad2deg(angle_z)
    if angle_z > 90:
        eigen_vectors[:, 2] = -eigen_vectors[:, 2]

    trans_world2pca = convert_eigenvalues2transformation(eigen_vectors, points_cam.mean(axis=0))
    cloud_sfm_o3d.transform(trans_world2pca)

    # 2. scale cloud based on the sclae of bounding boxes of sfm camera points and aruco camera points
    points_cam_pca = np.array(cloud_sfm_o3d.points)[:48]
    scale_aabb = get_rough_scale(points_cam_pca, points_cam_aruco)
    trans_scale_aabb = np.diag([scale_aabb, scale_aabb, scale_aabb, 1.0])
    cloud_sfm_o3d.transform(trans_scale_aabb)
    
    # 3. align centers of two kind of cameras
    trans_board_center = np.identity(4)
    trans_board_center[:3, 3] = points_cam_aruco.mean(axis=0) #np.array([150,150, 0])
    cloud_sfm_o3d.transform(trans_board_center)
    # o3d.io.write_point_cloud(f'{dir_sfm_seq}/robust_align_test1.ply', cloud_sfm_o3d)

    # 4. align camera points by SVD
    b_visualize_cam_corres = False
    if b_visualize_cam_corres:
        points_all = np.concatenate([np.array(cloud_sfm_o3d.points)[:48], points_cam_aruco], axis=0)
        cloud_all = convert_points_to_o3d_cloud(points_all)
        lines, colors = [], []
        for i in range(48):
            lines.append([i,i+48])
            colors.append([1,0,0])
        # lines = [[0,1],[0,2],[0,3]]      #画出三点之间两两连线
        # colors = [[1,0,0],[0,1,0],[0,0,1]]

        line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points_all),lines=o3d.utility.Vector2iVector(lines))
        line_set.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([cloud_all, line_set]) # 显示原始点云和PCA后的连线  

    trans_align_cams = align_cam_points(np.array(cloud_sfm_o3d.points)[:48], points_cam_aruco,)
    cloud_sfm_o3d.transform(trans_align_cams)
    # o3d.io.write_point_cloud(f'{dir_sfm_seq}/robust_align_test2.ply', cloud_sfm_o3d)

    # 4. scale icp
    points_cam_sfm_pca = np.array(cloud_sfm_o3d.points)[:48]
    vec_trans, vec_scale = [], []
    # for idx in [1, 2, 3, 5,-1, 2]:  
    for idx in [-1]:  
        cams_type = idx
        if cams_type==-1:
            cloud_cams_reg_sfm = convert_points_to_o3d_cloud(points_cam_sfm_pca)
            cloud_cams_reg_aruco = convert_points_to_o3d_cloud(points_cam_aruco)
        elif cams_type==5: # camera 2&3
            cloud_cams_reg_sfm = convert_points_to_o3d_cloud(points_cam_sfm_pca[16:48])
            cloud_cams_reg_aruco = convert_points_to_o3d_cloud(points_cam_aruco[16:48])
        else:
            cloud_cams_reg_sfm = convert_points_to_o3d_cloud(points_cam_sfm_pca[16*(cams_type-1):16*cams_type])
            cloud_cams_reg_aruco = convert_points_to_o3d_cloud(points_cam_aruco[16*(cams_type-1):16*cams_type])

        trans_sicp, scale_icp = scale_icp_o3d(cloud_cams_reg_sfm, cloud_cams_reg_aruco,
                                                thres_trunc=200,
                                                apply_scale=True)
        cloud_sfm_o3d_trans = copy.deepcopy(cloud_sfm_o3d)
        cloud_sfm_o3d_trans.transform(trans_sicp)
        o3d.io.write_point_cloud(f'{dir_sfm_seq}/robust_align_{cams_type}.ply', cloud_sfm_o3d_trans)
        # print(f'Scale ICP (align cams): {scale_icp}')

        # align cams
        trans_all = trans_sicp @ trans_align_cams @ trans_board_center @ trans_scale_aabb @ trans_world2pca
        scale_all = scale_aabb * scale_icp
        scale_trans_all = np.cbrt(np.linalg.det(trans_all[:3,:3]))

        print(f"Cam [{idx}]. ArUco scale: {scale_trans_all}")
        vec_trans.append(trans_all)
        vec_scale.append(scale_all)

    cam_final_idx = -1
    scale_all = vec_scale[cam_final_idx]
    trans_all = vec_trans[cam_final_idx]

    return trans_all, scale_all       

def register_sfmCam_to_arucoCam2(dir_sfm_seq):
    # parse sfm sparse point cloud
    path_cloud_sfm = os.path.join(dir_sfm_seq, 'robust.ply')
    path_cloud_aruco = os.path.join(dir_sfm_seq, 'robust_aruco_calcam0_board1.ply')
    
    num_cameras_calib = 48
    
    cloud_sfm_o3d = o3d.io.read_point_cloud(path_cloud_sfm)
    cloud_aruco_o3d = o3d.io.read_point_cloud(path_cloud_aruco)
    points_cam_aruco, points_board_aruco = np.array(cloud_aruco_o3d.points)[:48], np.array(cloud_aruco_o3d.points)[48:]

    _, points_cam, points_sparse = parse_sfm_sparse_cloud(path_cloud_sfm, num_cameras=48)
    if num_cameras_calib == 32:
        pts_aruco = np.array(cloud_aruco_o3d.points)
        num_aruco = len(pts_aruco)
        num_board = 441
        num_cams = num_aruco - num_board
        points_cam = points_cam[16:]
        points_cam_aruco = pts_aruco[num_cams-32:num_cams]
        # pts_debug = pts_aruco[:num_cams-32]

    dir_board2cam = points_cam.mean(axis=0) - points_sparse.mean(axis=0)
    dir_board2cam = dir_board2cam / np.linalg.norm(dir_board2cam)

    # 1. PCA: based on sparse points of the board
    eigen_values, eigen_vectors, center_sparse = calculate_pca(points_sparse)

    angle_z = np.arccos( dir_board2cam.dot(eigen_vectors[:,2]) / (np.linalg.norm(dir_board2cam) * np.linalg.norm(eigen_vectors[:,2])))
    angle_z = np.rad2deg(angle_z)
    if angle_z > 90:
        eigen_vectors[:, 2] = -eigen_vectors[:, 2]

    trans_world2pca = convert_eigenvalues2transformation(eigen_vectors, points_cam.mean(axis=0))
    cloud_sfm_o3d.transform(trans_world2pca)

    # 2. scale cloud based on the sclae of bounding boxes of sfm camera points and aruco camera points
    points_cam_pca = np.array(cloud_sfm_o3d.points)[:48]
    if num_cameras_calib == 32:
        points_cam_pca = np.array(cloud_sfm_o3d.points)[num_cams-32:num_cams]
    scale_aabb = get_rough_scale(points_cam_pca, points_cam_aruco)
    trans_scale_aabb = np.diag([scale_aabb, scale_aabb, scale_aabb, 1.0])
    cloud_sfm_o3d.transform(trans_scale_aabb)
    
    # 3. align centers of two kind of cameras
    trans_board_center = np.identity(4)
    trans_board_center[:3, 3] = points_cam_aruco.mean(axis=0) #np.array([150,150, 0])
    cloud_sfm_o3d.transform(trans_board_center)

    # 4. align camera points by SVD
    b_visualize_cam_corres = False
    if b_visualize_cam_corres:
        points_all = np.concatenate([np.array(cloud_sfm_o3d.points)[:48], points_cam_aruco], axis=0)
        cloud_all = convert_points_to_o3d_cloud(points_all)
        lines, colors = [], []
        for i in range(48):
            lines.append([i,i+48])
            colors.append([1,0,0])
        # lines = [[0,1],[0,2],[0,3]]      #画出三点之间两两连线
        # colors = [[1,0,0],[0,1,0],[0,0,1]]

        line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points_all),lines=o3d.utility.Vector2iVector(lines))
        line_set.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([cloud_all, line_set]) # 显示原始点云和PCA后的连线  

    points_cam_sfm = np.array(cloud_sfm_o3d.points)[:48]
    if num_cameras_calib == 32:
        points_cam_sfm = points_cam_sfm[16:]
    trans_align_cams = align_cam_points(points_cam_sfm, points_cam_aruco,)
    cloud_sfm_o3d.transform(trans_align_cams)

    # 4. scale icp
    points_cam_sfm_pca = np.array(cloud_sfm_o3d.points)[:48]
    if num_cameras_calib == 32:
        points_cam_sfm_pca = points_cam_sfm_pca[16:]
    trans_sicp, scale_icp = scale_icp_o3d(convert_points_to_o3d_cloud(points_cam_sfm_pca), convert_points_to_o3d_cloud(points_cam_aruco),
                                            thres_trunc=200,
                                            apply_scale=True)
    cloud_sfm_o3d.transform(trans_sicp)
    o3d.io.write_point_cloud(f'{dir_sfm_seq}/robust_align.ply', cloud_sfm_o3d)

    # align cams
    trans_all = trans_sicp @ trans_align_cams @ trans_board_center @ trans_scale_aabb @ trans_world2pca
    scale_all = scale_aabb * scale_icp
    scale_trans_all = np.cbrt(np.linalg.det(trans_all[:3,:3]))

    b_test = False
    if b_test:
        cloud_sfm_o3d2 = o3d.io.read_point_cloud(path_cloud_sfm)
        cloud_sfm_o3d2.transform(trans_all)
        o3d.io.write_point_cloud(f'{dir_sfm_seq}/test3.ply', cloud_sfm_o3d)
    print(f"Scale is {scale_trans_all}")
    return trans_all, scale_all      


def save_points(path_save, pts, colors = None, normals = None, BRG2RGB=False):
    '''save points to point cloud using open3d
    '''
    assert len(pts) > 0
    if colors is not None:
        assert colors.shape[1] == 3
    assert pts.shape[1] == 3
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        # Open3D assumes the color values are of float type and in range [0, 1]
        if np.max(colors) > 1:
            colors = colors / np.max(colors)
        if BRG2RGB:
            colors = np.stack([colors[:, 2], colors[:, 1], colors[:, 0]], axis=-1)
        cloud.colors = o3d.utility.Vector3dVector(colors) 
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals) 

    o3d.io.write_point_cloud(path_save, cloud)
    
def parse_cameras(path_cams):
    fcams = open(path_cams, 'r')
    lines = fcams.readlines()
    count_lines = 0
    vec_cameras = []
    for line in lines:
        if count_lines%13 == 11:
            line = list(filter(None, re.split('\[|,|\]|\n| ', line.strip())))
            vec_cameras.append(line)
        count_lines +=1
    vec_cameras = np.array(vec_cameras).astype(np.float32)
    return vec_cameras

def parse_cameras_intrinsics(path_cams, reso = 1):
    fcams = open(path_cams, 'r')
    lines = fcams.readlines()
    count_lines = 0
    vec_cameras = []
    for line in lines[3:6]:
        line = list(filter(None, re.split('\[|,|\]|\n| |;', line.strip())))
        vec_cameras.append(line)  
    vec_cameras = np.array(vec_cameras).astype(np.float32)
    if reso == 0:
        vec_cameras = vec_cameras*2
        vec_cameras[-1,-1] = 1
    return vec_cameras

def generate_sparse_ply(dir_batch, side='bottom'):
    path_cams_bottom = f'{dir_batch}/registration_reso1_maskthres242/{side}_cameras/KRC.txt'
    cameras_bottom =  parse_cameras(path_cams_bottom)

    path_bottom_ply = f'{dir_batch}/registration_reso1_maskthres242/{side}.ply'
    cloud_bottom = o3d.io.read_point_cloud(path_bottom_ply)
    points_bottom = np.array(cloud_bottom.points)
    idx_sample = np.random.choice(len(points_bottom), 2000)
    points_sample = points_bottom[idx_sample]

    dir_output_bottom = f'{dir_batch}/{side}_output/reconstruction_sequential'
    os.makedirs(dir_output_bottom, exist_ok=True)
    path_robust = f'{dir_output_bottom}/robust.ply'
    points_all = np.concatenate([cameras_bottom, points_sample], axis=0)
    save_points(path_robust, points_all)

def sharp_img(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst

def get_gray_images(dir_imgs, dir_imgs_gray, ext_img = '.tif', gray_channel = 0,  reverse = False, thres  = 200):
    vec_path_imgs = glob.glob(f'{dir_imgs}/*{ext_img}')
    os.makedirs(dir_imgs_gray, exist_ok=True)
   
    
    for i in tqdm(range(len(vec_path_imgs))):
        path_img= vec_path_imgs[i]
        stem = Path(path_img).stem
        img = cv2.imread(path_img, cv2.IMREAD_UNCHANGED)

        img_b_sharp = sharp_img(img[...,gray_channel])
        cv2.imwrite(f'{dir_imgs_gray}/{stem}.png', img_b_sharp)

        if reverse:
            ret, binary = cv2.threshold(img_b_sharp, thres, 255, cv2.THRESH_BINARY_INV)
            cv2.imwrite(f'{dir_imgs_gray}/{stem}.png', binary)

