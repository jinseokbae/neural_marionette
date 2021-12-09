import torch
import numpy as np
import sys
import os
import pickle
import open3d as o3d
from model.neural_marionette import NeuralMarionette
from utils.dataset_utils import crop_sequence, episodic_normalization, voxelize
import cv2
import imageio
from copy import deepcopy

exp_dir = 'pretrained/aist'
sample_id = 0

opt_file = os.path.join(exp_dir, 'opt.pickle')
with open(opt_file, 'rb') as f:
    opt = pickle.load(f)
opt.Ttot = 40

def extract_skin_weights(A, priority, parents, points, keypoints, HARDNESS=8.0, THRESHOLD=0.2):
    '''
    :param A:
    :param points: (N, 3) -- numpy
    :param keypoints: (K, 4) -- torch
    :return:
    '''
    # skin weights
    N, _ = points.shape
    K, _ = keypoints.size()

    invalids = torch.where((keypoints[:, -1] < THRESHOLD))[0]
    points_torch = torch.from_numpy(points).to(A.device)
    bones = torch.zeros_like(keypoints[:, :3])
    new_parents = deepcopy(parents)
    for k in range(K):
        parent = parents[k]
        if parent == k:
            bones[k] = keypoints[k, :3].clone()
        else:
            while (parent in invalids):
                parent = parents[parent]
            new_parents[k] = parent
            bones[k] = (keypoints[k, :3].clone() + keypoints[parent, :3].clone()) / 2

    dist = (points_torch[:, None] - bones[None, :, :3]).pow(2).sum(dim=-1).sqrt()  # (N, K)

    dist[:, invalids] = 1e4
    dist[:, priority.indices[0]] = 1e4  # never choose root


    nearests = dist.argmin(dim=-1)  # (N, )
    skin_weights = torch.zeros(N, K).to(A.device)
    for n in range(N):
        child = nearests[n]  # child of bone
        parent = parents[child]  # parent of
        child_dist = ((points_torch[n] - keypoints[child, :3]).pow(2).sum(dim=-1).sqrt() * HARDNESS).exp()
        parent_dist = ((points_torch[n] - keypoints[parent, :3]).pow(2).sum(dim=-1).sqrt() * HARDNESS).exp()
        skin_weights[n, parent] = child_dist / (child_dist + parent_dist)
        skin_weights[n, child] = parent_dist / (child_dist + parent_dist)

    return skin_weights.detach().cpu().numpy()



def load_voxel(file, start, scale=1.0, x_trans=0.0, z_trans=0.0):
    x = np.load(file)[..., :3]
    x = crop_sequence(x,
                      start=start,
                      T=opt.Ttot,
                      sample_rate=opt.sample_rate)
    x = episodic_normalization(x, scale, x_trans, z_trans)
    vox_seq = []
    for t in range(opt.Ttot):
        # idx = np.random.choice(N, self.N)
        try:
            vox_seq.append(voxelize(x[t], (opt.grid_size,) * 3, is_binarized=True))
        except:
            raise ValueError("Dataset voxelizer error")
    vox_seq = torch.from_numpy(np.stack(vox_seq, axis=0)).float().cuda()

    return vox_seq, x

def load_voxel_from_real_data(file, scale=1.0, x_trans=0.0, z_trans=0.0):
    pcd = o3d.io.read_point_cloud(file)
    x = np.asarray(pcd.points)  # (N, 3)
    x = episodic_normalization(x[None],  scale, x_trans, z_trans)
    vox = voxelize(x[0], (opt.grid_size,) * 3, is_binarized=True)[0]  # (64, 64, 64)
    coords = np.stack(np.meshgrid(np.arange(opt.grid_size), np.arange(opt.grid_size), np.arange(opt.grid_size)), axis=-1)
    return torch.from_numpy(vox).float().cuda()[None], x[0], np.asarray(pcd.colors)

def load_voxel_from_real_data_for_mesh(file, scale=1.0, x_trans=0.0, z_trans=0.0, is_bind=False):
    mesh = o3d.io.read_triangle_mesh(file, True)
    x = np.asarray(mesh.vertices)
    if is_bind:
        x = np.stack([x[:, 0], -x[:, 2], x[:, 1]], axis=-1)
    x = episodic_normalization(x[None],  scale, x_trans, z_trans)
    vox = voxelize(x[0], (opt.grid_size,) * 3, is_binarized=True)[0]  # (64, 64, 64)
    coords = np.stack(np.meshgrid(np.arange(opt.grid_size), np.arange(opt.grid_size), np.arange(opt.grid_size)), axis=-1)
    return torch.from_numpy(vox).float().cuda()[None], x[0], None, mesh

def drawCone2(bottom_center, top_position, color=[0.6, 0.9, 0.6]):
    '''
    cone = open3d.geometry.TriangleMesh.create_cone(radius=np.linalg.norm(top_position - bottom_center) / 10,
                                                 height=np.linalg.norm(top_position - bottom_center) * 0.2 + 1e-6,
                                                 resolution=80)
    '''
    cone = o3d.geometry.TriangleMesh.create_cone(0.03,
                                                 height=np.linalg.norm(top_position - bottom_center) * 0.2 + 1e-6,
                                                 resolution=80)
    cone = cone.rotate(cone.get_rotation_matrix_from_xyz((np.pi, 0, 0)))
    line1 = np.array([0.0, 0.0, 1.0])
    line2 = (top_position - bottom_center) / (np.linalg.norm(top_position - bottom_center) + 1e-6)
    v = np.cross(line1, line2)
    c = np.dot(line1, line2) + 1e-8
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + k + np.matmul(k, k) * (1 / (1 + c))
    if np.abs(c + 1.0) < 1e-4:  # the above formula doesn't apply when cos(∠(𝑎,𝑏))=−1
        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    margin = np.linalg.norm(top_position - bottom_center) * 0.195
    T = bottom_center + margin * line2

    cone.transform(
        np.concatenate((np.concatenate((R, T[:, np.newaxis]), axis=1), np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0))
    cone.paint_uniform_color(color)
    cone.compute_vertex_normals()
    return cone

def drawCone1(bottom_center, top_position, color=[0.6, 0.9, 0.6]):
    cone = o3d.geometry.TriangleMesh.create_cone(radius=0.03, height=np.linalg.norm(top_position - bottom_center)*0.8+1e-6, resolution=80)
    line1 = np.array([0.0, 0.0, 1.0])
    line2 = (top_position - bottom_center) / (np.linalg.norm(top_position - bottom_center)+1e-6)
    v = np.cross(line1, line2)
    c = np.dot(line1, line2) + 1e-8
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + k + np.matmul(k, k) * (1 / (1 + c))
    if np.abs(c + 1.0) < 1e-4: # the above formula doesn't apply when cos(∠(𝑎,𝑏))=−1
        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    margin = np.linalg.norm(top_position - bottom_center) * 0.2
    T = bottom_center + margin * line2
    cone.transform(np.concatenate((np.concatenate((R, T[:, np.newaxis]), axis=1), np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0))
    cone.paint_uniform_color(color)
    cone.compute_vertex_normals()
    return cone

def drawSphere(center, color=[0.6, 0.9, 0.6], radius=0.03):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    return sphere

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=int, default=0, help='seed for random')
    parser.add_argument('--is_bind', type=int, default=0, help='seed for random')
    parser.add_argument('--hardness', type=float, default=8.0, help='seed for random')
    parser.add_argument('--skel_mode', type=int, default=1, help='0 --> output only textured version')
    parser.add_argument('--source_mode', type=int, default=0, help='seed for random')
    parser.add_argument('--subdivide_iter', type=int, default=3, help='seed for random')
    arg = parser.parse_args()

    HARDNESS = arg.hardness
    VIS_THRESHOLD = 0.2
    np.random.seed(10000)
    colors_new = np.random.randn(opt.nkeypoints, 3)
    colors_min = colors_new.min()
    colors_max = colors_new.max()
    joint_colors = (colors_new - colors_min) / (colors_max - colors_min)

    resume_file = os.path.join(exp_dir, 'aist_pretrained.pth')
    checkpoint = torch.load(resume_file)
    network = NeuralMarionette(opt).cuda()
    network.load_state_dict(checkpoint)


    ##################################################################################################
    source_list = []
    source_list.append(
        dict(
            source_start=0,
            sample_rate=1,
            source_file='data/demo/source/gHO_sBM_cAll_d20_mHO1_ch05.npy'
        )
    )


    ##################################################################################################

    target_list = []
    target_list.append('data/demo/target/ninja')

    output_list = []
    output_list.append('output/demo/retarget/target/ninja')

    ##################################################################################################
    if arg.debug:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1025, height=1025)
    else:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1025, height=1023, visible=False)
        vis_2 = o3d.visualization.Visualizer()
        vis_2.create_window(width=1025, height=1023, visible=False)
        vis_3 = o3d.visualization.Visualizer()
        vis_3.create_window(width=1025, height=1023, visible=False)


    for target_idx, target_folder in enumerate(target_list):
        output_folder = output_list[target_idx]
        if (arg.source_mode or arg.debug) and target_idx > 0:
            break
        for source_idx, source_dict in enumerate(source_list):
            if arg.debug and source_idx > 0:
                break
            for is_baseline in range(1):
                arg.ours = not bool(is_baseline)
                source_file = source_dict['source_file']
                source_start = source_dict['source_start']
                opt.sample_rate = source_dict['sample_rate']

                source_seq = source_file.split('/')[-1].split('.')[0]
                print(target_folder, source_seq)
                source_voxel, source_points = load_voxel(source_file, source_start)

                file_name = 'target.obj'
                target_file = os.path.join(target_folder, file_name)
                scale = 0.8
                x_trans = 0.0

                target_voxel, points, colors, mesh = load_voxel_from_real_data_for_mesh(target_file, scale=scale, x_trans=x_trans, is_bind=arg.is_bind)

                ###################################################################################################################
                network.eval()
                network.anneal(1)  # to enable extracting affinity
                with torch.no_grad():
                    source_detector_log = network.kypt_detector(source_voxel[None])
                    source_keypoints = source_detector_log['keypoints']
                    # to consistently visualize rigs
                    source_keypoints[:, 1:, :, -1] = source_keypoints[:, :1, :, -1].expand(-1, opt.Ttot - 1, -1)
                    affinity = source_detector_log['affinity']
                    source_dyna_log = network.dyna_module.encode(source_keypoints, affinity)
                    R = source_dyna_log['R'][0]  # (T, K, 3, 3)
                    A = network.dyna_module.A
                    priority = network.dyna_module.priority
                    parents = network.dyna_module.parents
                    pos = source_keypoints[0, :, :, :3][..., None]  # (T, K, 3, 1)
                    T4x4 = torch.cat([R, pos], dim=-1)  # (T, K, 3, 4)
                    homo = torch.Tensor([0.0, 0.0, 0.0, 0.1]).to(R.device)[None, None, None].expand(R.size(0), R.size(1), -1,
                                                                                                    -1)  # (K, 1, 4)
                    T4x4 = torch.cat([T4x4, homo], dim=-2).detach().cpu().numpy()  # (K, 4, 4)


                ###################################################################################################################
                with torch.no_grad():
                    target_detector_log = network.kypt_detector(target_voxel[None, None])
                    target_keypoints = target_detector_log['keypoints']
                    target_keypoints = torch.cat([target_keypoints[..., :3], source_keypoints[:1, :1, :, -1:]], dim=-1)
                    target_dyna_log = network.dyna_module.encode(target_keypoints, affinity)
                    R_inv = target_dyna_log['R'][0, 0].transpose(1, 2).detach().cpu().numpy()  # (K, 3, 3)
                    joints = target_keypoints[0, 0, :, :3]  # (K, 3)
                    # skin weights
                    skin_weights = extract_skin_weights(A, priority, parents, points, target_keypoints[0, 0], HARDNESS)

                    if arg.ours:
                        pos = target_keypoints[0, 0, :, :3].detach().cpu().numpy()  # (K, 3)
                        offsets = points[:, None] - pos[None]  # (N, K, 3)
                        points_local = np.einsum('kij,nkj->nki', R_inv, offsets)
                    else:
                        pos = target_keypoints[0, 0, :, :3].detach().cpu().numpy()  # (K, 3)
                        points_local = points[:, None] - pos[None]  # (N, K, 3)

                    offset = network.dyna_module.get_offset(target_keypoints)
                    root = priority.indices[0].item()
                    new_keypoints = []
                    R = R[None]
                    if arg.ours:
                        for t in range(R.size(1)):
                            R_t = R[:, t]
                            root_pos = source_keypoints[:, t, root][..., :3]
                            pos = torch.zeros(1, R.size(2), 3).to(offset.device)
                            pos[:, root] = root_pos
                            for idx in priority.indices[1:]:
                                pos[:, idx] = torch.bmm(R_t[:, idx.item()], offset[:, idx]).squeeze(-1) + pos[:, parents[idx]]
                            new_keypoints.append(pos)
                    else:
                        for t in range(R.size(1)):
                            root_pos = source_keypoints[:, t, root][..., :3]
                            pos = torch.zeros(1, R.size(2), 3).to(offset.device)
                            pos[:, root] = root_pos
                            for idx in priority.indices[1:]:
                                source_offset = source_keypoints[0, t, idx, :3] - source_keypoints[0, t, parents[idx], :3]
                                source_len = (source_keypoints[0, t, idx, :3] - source_keypoints[0, t, parents[idx], :3]).pow(2).sum().sqrt().item()
                                target_len = (target_keypoints[0, 0, idx, :3] - target_keypoints[0, 0, parents[idx], :3]).pow(2).sum().sqrt().item()
                                pos[:, idx] = pos[:, parents[idx]] + source_offset * (target_len / source_len)
                            new_keypoints.append(pos)
                    new_keypoints = torch.stack(new_keypoints, dim=1).clip(-1, 1)
                    new_keypoints = torch.cat([new_keypoints, source_keypoints[..., 3:]], dim=-1)
                    R = R[0]

                    pos = new_keypoints[0, :, :, :3][..., None]  # (T, K, 3, 1)

                    if arg.ours:
                        T4x4 = torch.cat([R, pos], dim=-1)  # (T, K, 3, 4)
                    else:
                        R_iden = torch.eye(3)[None, None].expand(R.size(0), R.size(1), -1, -1).to(R.device)
                        T4x4 = torch.cat([R_iden, pos], dim=-1)  # (T, K, 3, 4)

                    homo = torch.Tensor([0.0, 0.0, 0.0, 0.1]).to(R.device)[None, None, None].expand(R.size(0), R.size(1), -1,
                                                                                                    -1)  # (K, 1, 4)
                    T4x4 = torch.cat([T4x4, homo], dim=-2).detach().cpu().numpy()  # (K, 4, 4)

                    homo_points = np.concatenate([points_local, np.ones((points.shape[0], joints.shape[0], 1))],
                                                 axis=-1)  # (N, K, 4)
                    new_points = []
                    for t in range(T4x4.shape[0]):
                        kin = np.einsum('kij,nkj->kin', T4x4[t], homo_points)
                        new_p = np.einsum('nk, kin->ni', skin_weights, kin)[..., :3]
                        new_points.append(new_p)
                    new_points = np.stack(new_points, axis=0)

                    if arg.debug:
                        ###################################################################################################################
                        if arg.debug == 1:
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(source_points[0])
                            vis.add_geometry(pcd)
                            vis.run()
                        elif arg.debug == 2:# debug == 2
                            vis.clear_geometries()
                            mesh.vertices = o3d.utility.Vector3dVector(points)
                            vis.add_geometry(mesh)
                            vis.run()
                        else:
                            invalids = torch.where((target_keypoints[0, 0, :, -1] < VIS_THRESHOLD))[0]
                            bones = torch.zeros_like(target_keypoints[0, 0, :, :3])
                            for k in range(bones.size(0)):
                                parent = parents[k]
                                if parent == k:
                                    bones[k] = target_keypoints[0, 0, k, :3].clone()
                                else:
                                    while (parent in invalids):
                                        parent = parents[parent]
                                    bones[k] = (target_keypoints[0, 0, k, :3].clone() + target_keypoints[0, 0, parent, :3].clone()) / 2
                            bones = bones.detach().cpu().numpy()
                            ###################################
                            target_colors = np.einsum('ki,nk->ni', joint_colors, skin_weights)
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(points)
                            pcd.colors = o3d.utility.Vector3dVector(target_colors)
                            vis.add_geometry(pcd)
                            target_keypoint = target_keypoints[0, 0].detach().cpu().numpy()
                            kypts = target_keypoint[..., :3]
                            alphas = source_keypoints[0, 0, :, -1]
                            for k in range(alphas.size(0)):
                                if alphas[k] < VIS_THRESHOLD:
                                    continue
                                vis.add_geometry(drawSphere(kypts[k], joint_colors[k]))
                                ##########################################
                                vis.add_geometry(drawSphere(bones[k], joint_colors[k]))
                                #########################################
                                parent = parents[k].item()
                                if alphas[parent] < VIS_THRESHOLD or k == parent:
                                    continue
                                vis.add_geometry(drawCone1(kypts[parent], kypts[k], color=[0, 0.6, 0.1]))
                                vis.add_geometry(drawCone2(kypts[parent], kypts[k], color=[0, 0.6, 0.1]))
                            vis.run()
                        ###################################################################################################################

                    elif arg.source_mode:
                        read_dir = os.path.join(target_folder, source_seq)
                        save_dir = os.path.join(output_folder, source_seq)
                        os.makedirs(os.path.join(save_dir, 'source_imgs'), exist_ok=True)
                        imgs = []
                        vis.clear_geometries()
                        source_keypoints = source_keypoints.squeeze(0)
                        min_z = source_points.reshape(source_points.shape[0] * source_points.shape[1], -1).min(axis=0)[-1]
                        max_z = source_points.reshape(source_points.shape[0] * source_points.shape[1], -1).max(axis=0)[-1]
                        z_len = (max_z - min_z)
                        for t in range(R.size(0)):
                            source_point = source_points[t]
                            source_keypoint = source_keypoints[t]
                            skin_weights = extract_skin_weights(A, priority, parents, source_point, source_keypoint, HARDNESS)
                            for n in range(10000):
                                sphere_color = list(np.array([0.9, 0.9, 0.5]) *
                                                    ((source_point[n, -1] - min_z) / z_len * 0.9 + 0.1))
                                vis.add_geometry(drawSphere(source_point[n], color=sphere_color, radius=0.015))
                            ctr = vis.get_view_control()
                            parameters = o3d.io.read_pinhole_camera_parameters(os.path.join(read_dir, "source.json"))
                            ctr.convert_from_pinhole_camera_parameters(parameters)
                            img = vis.capture_screen_float_buffer(True)
                            img = (np.asarray(img) * 255).astype(np.uint8)
                            cv2.imwrite(os.path.join(save_dir, 'source_imgs', '%02d.png' % t), img[..., ::-1])
                            imgs.append(img)
                            vis.clear_geometries()
                        imageio.mimsave(os.path.join(save_dir, 'source.gif'), imgs, duration=0.1)
                    else:
                        read_dir = os.path.join(target_folder, source_seq)
                        save_dir = os.path.join(output_folder, source_seq)
                        if source_idx == 0 and is_baseline == 0:
                            temp_mesh = deepcopy(mesh)
                            temp_mesh.vertices = o3d.utility.Vector3dVector(points)
                            vis.add_geometry(temp_mesh)
                            ctr = vis.get_view_control()
                            parameters = o3d.io.read_pinhole_camera_parameters(os.path.join(target_folder, "target.json"))
                            ctr.convert_from_pinhole_camera_parameters(parameters)
                            img = vis.capture_screen_float_buffer(True)
                            img = (np.asarray(img) * 255).astype(np.uint8)
                            cv2.imwrite(os.path.join(output_folder, 'target.png'), img[..., ::-1])
                            vis.clear_geometries()


                            target_colors = np.einsum('ki,nk->ni', joint_colors, skin_weights)
                            temp_mesh = o3d.geometry.TriangleMesh()
                            temp_mesh.vertices = o3d.utility.Vector3dVector(points)
                            temp_mesh.vertex_colors = o3d.utility.Vector3dVector(target_colors)
                            temp_mesh.vertex_normals = deepcopy(mesh.vertex_normals)
                            temp_mesh.triangles = deepcopy(mesh.triangles)
                            temp_mesh = temp_mesh.subdivide_loop(arg.subdivide_iter)
                            temp_mesh = temp_mesh.compute_vertex_normals()
                            vis.add_geometry(temp_mesh)
                            ctr = vis.get_view_control()
                            parameters = o3d.io.read_pinhole_camera_parameters(os.path.join(target_folder, "target.json"))
                            ctr.convert_from_pinhole_camera_parameters(parameters)
                            img = vis.capture_screen_float_buffer(True)
                            img = (np.asarray(img) * 255).astype(np.uint8)
                            cv2.imwrite(os.path.join(output_folder, 'target_skin.png'), img[..., ::-1])
                            vis.clear_geometries()


                            temp_mesh = o3d.geometry.TriangleMesh()
                            temp_mesh.vertices = o3d.utility.Vector3dVector(points)
                            temp_mesh.vertex_colors = o3d.utility.Vector3dVector(target_colors)
                            temp_mesh.vertex_normals = deepcopy(mesh.vertex_normals)
                            temp_mesh.triangles = deepcopy(mesh.triangles)
                            temp_mesh = temp_mesh.subdivide_loop(arg.subdivide_iter)
                            temp_mesh = temp_mesh.compute_vertex_normals()
                            vis.add_geometry(temp_mesh)
                            ctr = vis.get_view_control()
                            parameters = o3d.io.read_pinhole_camera_parameters(os.path.join(target_folder, "target.json"))
                            ctr.convert_from_pinhole_camera_parameters(parameters)
                            img = np.array(vis.capture_screen_float_buffer(True))
                            vis.clear_geometries()
                            target_keypoint = target_keypoints[0, 0].detach().cpu().numpy()
                            kypts = target_keypoint[..., :3]
                            alphas = source_keypoints[0, 0, :, -1]
                            for k in range(alphas.size(0)):
                                if alphas[k] < VIS_THRESHOLD:
                                    continue
                                vis_2.add_geometry(drawSphere(kypts[k], joint_colors[k]))
                                parent = parents[k].item()
                                if alphas[parent] < VIS_THRESHOLD or k == parent:
                                    continue
                                vis_2.add_geometry(drawCone1(kypts[parent], kypts[k], color=[0, 0.6, 0.1]))
                                vis_2.add_geometry(drawCone2(kypts[parent], kypts[k], color=[0, 0.6, 0.1]))
                            ctr = vis_2.get_view_control()
                            parameters = o3d.io.read_pinhole_camera_parameters(os.path.join(target_folder, "target.json"))
                            ctr.convert_from_pinhole_camera_parameters(parameters)
                            cone_img = np.array(vis_2.capture_screen_float_buffer(True))
                            vis_2.clear_geometries()
                            img[cone_img.sum(axis=-1) != 3] = cone_img[cone_img.sum(axis=-1) != 3]
                            img = (img * 255).astype(np.uint8)
                            cv2.imwrite(os.path.join(output_folder, 'overlay_target.png'), img[..., ::-1])
                        ##################################################################################################
                        if not arg.skel_mode:
                            os.makedirs(os.path.join(save_dir, 'textured_imgs'), exist_ok=True)
                            imgs = []
                            for t in range(new_points.shape[0]):
                                mesh.vertices = o3d.utility.Vector3dVector(new_points[t])
                                vis.add_geometry(mesh)
                                # vis.run() # use for break
                                ctr = vis.get_view_control()
                                parameters = o3d.io.read_pinhole_camera_parameters(os.path.join(read_dir, "source.json"))
                                ctr.convert_from_pinhole_camera_parameters(parameters)
                                img = vis.capture_screen_float_buffer(True)
                                vis.clear_geometries()
                                img = (np.asarray(img) * 255).astype(np.uint8)
                                cv2.imwrite(os.path.join(save_dir, 'textured_imgs', '%02d.png' % t), img[..., ::-1])
                                imgs.append(img)

                            imageio.mimsave(os.path.join(save_dir, 'result_textured.gif'), imgs, duration=0.1)
                        else:
                            os.makedirs(os.path.join(save_dir, 'overlay_imgs'), exist_ok=True)
                            os.makedirs(os.path.join(save_dir, 'smooth_imgs'), exist_ok=True)
                            os.makedirs(os.path.join(save_dir, 'skel_imgs'), exist_ok=True)
                            os.makedirs(os.path.join(save_dir, 'source_skel_imgs'), exist_ok=True)

                            new_keypoints = new_keypoints.squeeze(0)
                            source_keypoints = source_keypoints.squeeze(0)

                            smooth_imgs = []
                            skel_imgs = []
                            source_skel_imgs = []
                            overlay_imgs = []
                            for t in range(new_points.shape[0]):
                                # smoothed_mesh
                                temp_mesh = o3d.geometry.TriangleMesh()
                                temp_mesh.vertices = o3d.utility.Vector3dVector(new_points[t])
                                temp_mesh.vertex_normals = deepcopy(mesh.vertex_normals)
                                temp_mesh.triangles = deepcopy(mesh.triangles)
                                temp_mesh = temp_mesh.subdivide_loop(arg.subdivide_iter)
                                temp_mesh.compute_vertex_normals()
                                vis.add_geometry(temp_mesh)
                                ctr = vis.get_view_control()
                                parameters = o3d.io.read_pinhole_camera_parameters(os.path.join(read_dir, "source.json"))
                                ctr.convert_from_pinhole_camera_parameters(parameters)
                                img = np.asarray(vis.capture_screen_float_buffer(True))
                                smooth_imgs.append((img * 255).astype(np.uint8))
                                cv2.imwrite(os.path.join(save_dir, 'smooth_imgs', '%02d.png' % t), (img * 255).astype(np.uint8)[..., ::-1])
                                vis.clear_geometries()
                                # skels
                                keypoint = new_keypoints[t].detach().cpu().numpy()
                                kypts = keypoint[..., :3]
                                alphas = keypoint[..., -1].clip(0, 1)

                                source_keypoint = source_keypoints[t].detach().cpu().numpy()
                                source_kypts = source_keypoint[..., :3]
                                for k in range(new_keypoints.shape[1]):
                                    if alphas[k] < VIS_THRESHOLD:
                                        continue
                                    vis_2.add_geometry(drawSphere(kypts[k], [0.7, 0.1, 0]))
                                    vis_3.add_geometry(drawSphere(source_kypts[k], [0.7, 0.1, 0]))
                                    parent = parents[k].item()
                                    if alphas[parent] < VIS_THRESHOLD or k == parent:
                                        continue
                                    vis_2.add_geometry(drawCone1(kypts[parent], kypts[k], color=[0, 0.6, 0.1]))
                                    vis_2.add_geometry(drawCone2(kypts[parent], kypts[k], color=[0, 0.6, 0.1]))
                                    vis_3.add_geometry(drawCone1(source_kypts[parent], source_kypts[k], color=[0, 0.6, 0.1]))
                                    vis_3.add_geometry(drawCone2(source_kypts[parent], source_kypts[k], color=[0, 0.6, 0.1]))
                                ctr = vis_2.get_view_control()
                                parameters = o3d.io.read_pinhole_camera_parameters(os.path.join(read_dir, "source.json"))
                                ctr.convert_from_pinhole_camera_parameters(parameters)
                                ctr = vis_3.get_view_control()
                                parameters = o3d.io.read_pinhole_camera_parameters(os.path.join(read_dir, "source.json"))
                                ctr.convert_from_pinhole_camera_parameters(parameters)

                                cone_img = np.asarray(vis_2.capture_screen_float_buffer(True))
                                skel_imgs.append((cone_img * 255).astype(np.uint8))
                                cv2.imwrite(os.path.join(save_dir, 'skel_imgs', '%02d.png' % t), (cone_img * 255).astype(np.uint8)[..., ::-1])
                                vis_2.clear_geometries()

                                source_cone_img = np.asarray(vis_3.capture_screen_float_buffer(True))
                                source_skel_imgs.append((source_cone_img * 255).astype(np.uint8))
                                cv2.imwrite(os.path.join(save_dir, 'source_skel_imgs', '%02d.png' % t), (source_cone_img * 255).astype(np.uint8)[..., ::-1])
                                vis_3.clear_geometries()

                                # overlay
                                img[cone_img.sum(axis=-1) != 3] = cone_img[cone_img.sum(axis=-1) != 3]
                                img = (img * 255).astype(np.uint8)
                                cv2.imwrite(os.path.join(save_dir, 'overlay_imgs', '%02d.png' % t), img[..., ::-1])
                                overlay_imgs.append(img)
                            imageio.mimsave(os.path.join(save_dir, 'result_smooth.gif'), smooth_imgs, duration=0.1)
                            imageio.mimsave(os.path.join(save_dir, 'result_skel.gif'), skel_imgs, duration=0.1)
                            imageio.mimsave(os.path.join(save_dir, 'result_source_skel.gif'), source_skel_imgs, duration=0.1)
                            imageio.mimsave(os.path.join(save_dir, 'result_overlay.gif'), overlay_imgs, duration=0.1)

                        #################################################################################################3



