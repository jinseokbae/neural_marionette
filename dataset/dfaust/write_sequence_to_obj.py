# -*- coding: utf-8 -*-
# Script to write registrations as obj files
# Copyright (c) [2015] [Gerard Pons-Moll]

from argparse import ArgumentParser
import h5py
import os
import sys
import numpy as np
import open3d as o3d
import trimesh

def write_mesh_as_obj(fname, verts, faces):
    with open(fname, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

def sample_faces(mesh, N=20000):
    P, t = trimesh.sample.sample_surface(mesh, N)
    sampled_faces = np.hstack([P, mesh.face_normals[t, :]]).astype(np.float32)
    return sampled_faces # (N, 3 + 3)

if __name__ == '__main__':

    # Subject ids
    np.random.seed(0)

    sids = ['50002', '50004', '50007', '50009', '50020',
            '50021', '50022', '50025', '50026', '50027']
    # Sequences available for each subject id are listed in scripts/subjects_and_sequences.txt

    parser = ArgumentParser(description='Save sequence registrations as obj')
    # parser.add_argument('--reg_path', type=str, default='data/D-FAUST/registrations_f.hdf5',
    #                     help='dataset path in hdf5 format')
    parser.add_argument('--path', type=str, default='../../../data/D-FAUST/',
                        help='dataset path in hdf5 format')
    # parser.add_argument('--seq', type=str, default='jiggle_on_toes',
    #                     help='sequence name')
    parser.add_argument('--sid', type=str, default='50002', choices=sids,
                        help='subject id')
    parser.add_argument('--pnum', type=int, default=4096,
                        help='point number')
    # parser.add_argument('--tdir', type=str, default='./',
    #                     help='target directory')
    args = parser.parse_args()

    for sid in sids:

        match_flag = False

        seq_list = []
        with open('subjects_and_sequences.txt', 'r') as f:
            info = f.read().splitlines()
            for line in info:
                if len(line.split()) == 2 and not match_flag:
                    seq_id, gender = line.split()
                    if not seq_id == sid:
                        continue
                    if gender == '(male)':
                        reg_path = os.path.join(args.path, 'registrations_m.hdf5')
                    elif gender == '(female)':
                        reg_path = os.path.join(args.path, 'registrations_f.hdf5')
                    match_flag = True
                elif len(line.split()) == 1 and match_flag:
                    seq_list.append(line)
                elif len(line.split()) == 2:
                    break

        for seq in seq_list:
            sidseq = sid + '_' + seq
            with h5py.File(reg_path, 'r') as f:
                if sidseq not in f:
                    print('Sequence %s from subject %s not in %s' % (seq, sid, reg_path))
                    continue
                    # f.close()
                    # sys.exit(1)
                verts = np.array(f[sidseq]).transpose([2, 0, 1])
                assert verts.shape[1] >= args.pnum
                faces = np.array(f['faces'])

            save_pc_dir = os.path.join(args.path, 'surface', sid)
            os.makedirs(save_pc_dir, exist_ok=True)
            # normalization to xlim(0, 1) ylim(0, 1) zlim(0, 1)
            # bmin = np.amin(verts, axis=(0, 1))
            # bmax = np.amax(verts, axis=(0, 1))
            # blen = np.max(bmax - bmin)
            # verts -= bmin
            # verts = (verts / blen)


            # for iv, v in enumerate(verts):
            #     fname = os.path.join(save_obj_dir, '%05d.obj' % iv)
            #     print('Saving mesh %s' % fname, v.shape[0])
            #     write_mesh_as_obj(fname, v, faces)
            #     mesh = o3d.geometry.TriangleMesh()
            #     mesh.vertices = o3d.utility.Vector3dVector(v)
            #     mesh.triangles = o3d.utility.Vector3iVector(faces)
            #     mesh.compute_vertex_normals()
            #     normals = np.asarray(mesh.vertex_normals)
            #     v = np.concatenate([v, normals], axis=-1)
            #     # np.random.shuffle(v)
            #     # v = v[:args.pnum]
            #     np.save(os.path.join(save_pc_dir, '%05d.npy' % iv), v)

            sampled = []
            for iv, v in enumerate(verts):
                mesh = trimesh.Trimesh(
                    vertices=v,
                    faces=faces
                )
                sampled.append(sample_faces(mesh, N=20000))
            sampled = np.stack(sampled, axis=0)
            np.save(os.path.join(save_pc_dir, seq) + '.npy', sampled)
            print(os.path.join(save_pc_dir, seq) + ' saved')
