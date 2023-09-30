import os
import torch
import numpy as np

import utils.general as utils
from utils import rend_util
from glob import glob

import re
import sys
import cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET
import json 
from PIL import Image
import skimage
from collections import defaultdict
from mesh_to_sdf import sample_sdf_near_surface, mesh_to_sdf
import trimesh
import pyrender
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import datetime
from sklearn.neighbors import KDTree
from h3ds.dataset import H3DS


class H3DSDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        '''
        kwargs:
        conf: dataset config
        '''
        super(H3DSDataset, self).__init__()
        
        self.conf = kwargs['conf']
        self.img_res = self.conf['img_res']
        self.total_pixels = self.img_res[0] * self.img_res[1]
        self.num_id = 0
        self.resize_flag = self.conf['resize_flag']
        self.white_flag = self.conf['white_flag']

        self.sampling_idx = None
        self.scan_id = self.conf.get_int('scan_id')
        self.cam_scale = self.conf.get_float('cam_scale')
        self.data_dir = self.conf['data_dir']
        assert os.path.exists(self.data_dir), "Data directory is empty"

        self.render_cameras_name = self.conf.get_string('render_cameras_name')
        self.scale_mat_scale = self.conf.get_float('cam_scale', default=1.1)

        # there are 15 wrinkle id and 15 smooth id in base_id.txt
        st_30id_path = self.conf.get_string('id_path', default='./camera_file/base/base_id.txt')

        self.id_list = np.loadtxt(st_30id_path, dtype=np.str_).tolist()
        self.id_list.sort()

        self.rgb_images_all = []
        self.ids = []
        self.names = []
        
        init_view_path = self.conf.get_string('init_view_path', default=None)
        sparse_view_num = self.conf.get_int('sparse_view_num', default=-1)

        h3ds_name = ['1b2a8613401e42a8', '3b5a2eb92a501d54','444ea0dc5e85ee0b','5ae021f2805c0854',\
                     '5cd49557ea450c89','609cc60fd416e187','7dd427509fe84baa','868765907f66fd85', \
                        'e98bae39fad2244e','f7e930d8a9ff2091']

        # fine-tune config
        self.fine_tune_flag = self.conf.get_bool('fine_tune_flag',default=False)
        if self.fine_tune_flag:
            print('for fine-tune, id will be added in previous id_list as last item')
            self.id_list.append(h3ds_name[self.scan_id])
        print('PR-facedataset load {} identities: {}'.format(len(self.id_list),self.id_list))

        h3ds = H3DS(path=self.data_dir)
        mesh, images, masks, cameras = h3ds.load_scene(scene_id=self.id_list[-1], views_config_id='8', normalized=True)
        #mesh: trimesh
        # mesh.save('./mesh/gt_h3ds/{}.obj'.format(self.id_list[-1]))
        self.H, self.W = np.array(images[0]).shape[0], np.array(images[0]).shape[1]
        self.image_pixels = self.H * self.W

        self.rgb_images_all = []
        self.masks_all = []
        self.images_np = np.stack([np.array(img).reshape(-1, 3) for img in images]) / 255.0
        self.masks_np = np.stack([np.array(mask).reshape(-1, 3) for mask in masks]) / 255.0
        bg_img = np.ones_like(self.images_np)
        self.images_np = self.images_np * self.masks_np + (1-self.masks_np) * bg_img
        self.n_images = self.images_np.shape[0]

        # world_mat is a projection matrix from world to image
        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.names = []
        self.intrinsics_all = []
        self.pose_all = []
        self.ids = []
        for cam in cameras:
            intrinsics = np.eye(4)
            intrinsics[:3,:3] = cam[0]
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            pose = cam[1]
            pose[:3, 3] = pose[:3, 3] * 0.3
            pose[1:3,:] = pose[1:3,:] * -1
            self.pose_all.append(torch.from_numpy(pose).float())
            self.names.append(str(self.scan_id))
            self.ids.append(0)


        self.rgb_images_all = torch.from_numpy(self.images_np.astype(np.float32))

        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32))
        self.intrinsics_all = torch.stack(self.intrinsics_all)  # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all) # [n_images, 4, 4]
        self.ids = torch.from_numpy(np.array(self.ids))

        print('Load data: End')

    def __len__(self):
        return len(self.pose_all)

    def __getitem__(self, idx):
        idx = idx % len(self.pose_all)    
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)
        rgb = self.rgb_images_all[idx]
        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx],
            "id": self.ids[idx]
        }
        ground_truth = {
            "rgb": rgb
        }
        if self.sampling_idx is not None:
            ground_truth["rgb"] = rgb[self.sampling_idx, :]
            sample["uv"] = uv[self.sampling_idx, :]
        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)
    
    def change_sampling_idx(self, sampling_size, semantic_pred=None, semantic_gt=None, rgb_pred=None, rgb_gt=None, w=0):
        self.sampling_size = sampling_size
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            if semantic_pred is None:
                self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]
            else:
                # get sampling number given semantic loss in each zone
                # print("change sampling idx needs the semantic info")
                with torch.no_grad():
                    sem_area_loss = torch.zeros((self.semantic_class_num,))
                    rgb_area_loss = torch.zeros((self.semantic_class_num,))
                    for c in range(self.semantic_class_num):
                        index_c = (semantic_gt==c)
                        if index_c.sum() != 0: 
                            # not all classes in one face
                            sem_area_loss[c] = torch.nn.functional.cross_entropy(semantic_pred[index_c], semantic_gt[index_c], reduction='mean')
                            rgb_area_loss[c] = torch.nn.functional.l1_loss(rgb_pred[index_c], rgb_gt[index_c], reduction='mean')
                    area_loss = w * sem_area_loss + rgb_area_loss
                    self.area_sample_num = (area_loss / area_loss.sum() * sampling_size).floor().cpu().numpy().astype(int)
    
    def get_specific_item(self, idx=-1, cam_id=-1):
        # cam_id \in [1,30]
        if idx == -1:
            person_id = str(self.scan_id)
            idx = self.id_list.index(person_id)
            # print('in get specific cam, scan_id=',person_id)
        else:
            person_id = self.id_list[idx]
        cam_file = os.path.join(self.data_dir, person_id, 'cameraParameters.xml')
        cameras = load_xml(cam_file)
        c2w_all = []
        intrinsics_all = []
        for name, (K, dist, c2w, hw) in cameras.items():
            if int(name[:2]) == cam_id:
                select_img_path = os.path.join(self.data_dir, person_id, name)
                c2w = c2w[...,:3,:4].reshape(-1,3,4)
                c2w[:,:3,3] = c2w[:,:3,3]*self.cam_scale
                c2w = np.concatenate([c2w, [[[0,0,0,1]]]*len(c2w)], 1)
                c2w_all.append(c2w.astype(np.float32))
                K = K[...,:2,:3].reshape(-1,2,3)
                K = np.concatenate([K, [[[0,0,1]]]*len(K)], 1)
                intrinsics_all.append(K.astype(np.float32))
        c2w_all = torch.from_numpy(np.concatenate(c2w_all, 0))
        intrinsics_all = torch.from_numpy(np.concatenate(intrinsics_all, 0))
        idx = torch.from_numpy(np.array([idx]))
        rgb = rend_util.load_rgb(select_img_path, resize_size=self.img_res,
                                resize_flag=self.resize_flag, white_flag=self.white_flag)
        rgb = rgb[:3,:,:].reshape(3, -1).transpose(1, 0)
        rgb = torch.from_numpy(rgb).float()

        return c2w_all, intrinsics_all, idx, rgb.unsqueeze(0)

    def get_scale_mat(self):
        scale_mat_path = os.path.join(self.data_dir,self.id_list[-1],self.render_cameras_name)
        return np.load(scale_mat_path)['scale_mat_0']

    def get_id_unique_list(self):
        return self.id_list
    
    def get_novel_pose_between(self, idx_0, idx_1, i, n_frames=60):
        ratio = np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5
        pose_0 = self.pose_all[idx_0].numpy()
        pose_1 = self.pose_all[idx_1].numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        
        return torch.from_numpy(pose)
    
    def load_K_Rt_from_P(self, filename, P=None):
        if P is None:
            lines = open(filename).read().splitlines()
            if len(lines) == 4:
                lines = lines[1:]
            lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
            P = np.asarray(lines).astype(np.float32).squeeze()

        out = cv2.decomposeProjectionMatrix(P)
        K = out[0]
        R = out[1]
        t = out[2]

        K = K / K[2, 2]
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]

        return intrinsics, pose