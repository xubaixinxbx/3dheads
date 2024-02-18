import os
import torch
import numpy as np

import utils.general as utils
from utils import rend_util
from glob import glob
import cv2
import json
import trimesh
import skimage

def load_rgb_resize(path, factor):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H = img.shape[0]
    W = img.shape[1]

    img = cv2.resize(img, (W // factor, H // factor), interpolation=cv2.INTER_AREA)
    img = skimage.img_as_float32(img)
    # pixel values between [-1,1]
    # img -= 0.5
    # img *= 2.
    img = img.transpose(2, 0, 1)
    return img


def load_mask_resize(path, factor):
    alpha = cv2.imread(path, 0)
    H = alpha.shape[0]
    W = alpha.shape[1]
    alpha = cv2.resize(alpha, (W // factor, H // factor), interpolation=cv2.INTER_AREA)
    alpha = skimage.img_as_float32(alpha)
    object_mask = alpha > 0.5
    #object_mask = alpha < 0.5
    return object_mask

class SceneDataset(torch.utils.data.Dataset):

    def __init__(self,**kwargs):
        
        expression = ['neutral', 'smile', 'mouth_stretch', 'anger', 'jaw_left', 'jaw_right', 'jaw_forward',
                'mouth_left', 'mouth_right', 'dimpler', 'chin_raiser',
                'lip_puckerer', 'lip_funneler', 'sadness', 'lip_roll', 'grin', 'cheek_blowing', 'eye_closed',
                'brow_raiser', 'brow_lower']
        id_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.id_list = id_list
        self.conf = kwargs['conf']
        self.img_res = self.conf['img_res']
        
        data_dir = self.conf['data_dir']
        scan_id = self.conf['scan_id']
        views = self.conf['views']
        split = self.conf['split']
        factor = self.conf['factor']
        self.st1_flag = self.conf.get_bool('st1_flag', default=True)

        i = scan_id
        j = facescape_exp = 16
        self.views = views
        self.instance_dir = data_dir
        image_dir = '{0}/image'.format(self.instance_dir)
        mask_dir = '{0}/mask'.format(self.instance_dir)
        params_dir = '{0}/image'.format(self.instance_dir)
        ply_path = '{0}/model'.format(self.instance_dir)
        
        self.intrinsics_all = []
        self.pose_all = []
        self.rgb_images = []
        self.h = []
        self.w = []
        self.object_masks = []
        self.id = []
        self.exp = []
        self.exp_num = []
        self.sample_list = []
        self.have_sample = []
        self.ids = []
        self.names = []
        if self.st1_flag:
            for idx in id_list:
                if scan_id!=-1 and idx != scan_id:
                    continue
                self.read_single_dataset(image_dir, params_dir, ply_path, mask_dir, idx, j, factor, expression, split)        
        else:
            self.id_list = [scan_id]
            self.read_single_dataset(image_dir, params_dir, ply_path, mask_dir, scan_id, j, factor, expression, split)        
        self.n_images = len(self.pose_all)
        print('image num: {0}'.format(self.n_images))
        self.ids = torch.from_numpy(np.array(self.ids))
        self.total_pixels = None
        self.img_res = None
        self.sampling_idx = None

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.h[idx], 0:self.w[idx]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx],
            "id": self.ids[idx]
        }

        ground_truth = {
            "rgb": self.rgb_images[idx],
            'hw': torch.from_numpy(np.array([self.h[idx], self.w[idx]])),
            'mask': self.object_masks[idx]
        }

        if self.sampling_idx is not None:
            s_idx = self.sampling_idx[idx]
            ground_truth["rgb"] = self.rgb_images[idx][s_idx, :]
            sample["uv"] = uv[s_idx, :]
            ground_truth['mask'] = ground_truth['mask'][s_idx]

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

    def change_sampling_idx(self, sampling_size):
        # if sampling_size == -1:
        #     self.sampling_idx = None
        # else:
        #     self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = []
            for i in range(self.n_images):
                total = self.sample_list[i].shape[0]
                s_idx = torch.randperm(total)[:sampling_size]
                self.have_sample[i][self.sample_list[i][s_idx]] = 0
                self.sampling_idx.append(self.sample_list[i][s_idx])

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

    def read_single_dataset(self, image_dir, params_dir, ply_path, mask_dir, i, j, factor, expression, split):
        land_dir = os.path.join(self.instance_dir, 'landmark_indices.npz')
        landmark = np.load(land_dir)['v10'][30]
        with open(os.path.join(self.instance_dir, f"Rt_scale_dict.json"), 'r') as f:
            Rt_scale_dict = json.load(f)
        if split == 'train':
            if self.st1_flag:
                i_dir = os.path.join(image_dir, str(i), '{0}_{1}'.format(j, expression[j-1]), split) # for template 
            else:
                i_dir = os.path.join(image_dir, str(i), '{0}_{1}'.format(j, expression[j-1]), split + f'_{self.views}') # for stage2
        elif split == 'test':
            i_dir = os.path.join(image_dir, str(i), '{0}_{1}'.format(j, expression[j-1]), split) # for test
            
        par_dir = os.path.join(params_dir, str(i), '{0}_{1}'.format(j, expression[j-1]))
        with open(os.path.join(par_dir, f"params.json"), 'r') as f:
            meta = json.load(f)
        image_path = os.listdir(i_dir)
        image_path.sort()
        scale = Rt_scale_dict['%d' % i]['%d' % j][0]
        Rt = np.array(Rt_scale_dict['%d' % i]['%d' % j][1])
        
        ply_name = '{0}_{1}.obj'.format(j, expression[j - 1])
        ply = os.path.join(ply_path, str(i), ply_name)
        mview_mesh = trimesh.load(ply, process=False, maintain_order=True)
        # align multi-view model to TU model
        point = mview_mesh.vertices[landmark]
        for p in image_path:
            if not p.endswith('.jpg'): continue
            self.id.append(torch.tensor(0))
            self.exp.append(torch.tensor(0))
            num = p[:-4]
            self.exp_num.append(int(num))
            r_path = os.path.join(i_dir, p)
            rgb = load_rgb_resize(r_path, factor)

            self.h.append(rgb.shape[1])
            self.w.append(rgb.shape[2])
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())

            m_path = os.path.join(mask_dir, str(i), '{0}_{1}'.format(j, expression[j-1]), split, num + '.jpg')
            object_mask = load_mask_resize(m_path, factor)
            h, w = object_mask.shape
            x_axis, y_axis = np.where(object_mask == True)
            x_min = max(x_axis.min() - 30, 0)
            x_max = min(x_axis.max() + 30, h - 1)
            y_min = max(y_axis.min() - 30, 0)
            y_max = min(y_axis.max() + 30, w - 1)
            sam_list = []
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    sam_list.append(x * w + y)
            # train with the pixels in the mask area
            self.sample_list.append(torch.tensor(sam_list))
            self.have_sample.append(torch.zeros(h * w) + 255)
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(torch.from_numpy(object_mask).bool())

            # obtain camera's intrinsics and extrinsics
            K = np.array(meta[num + '_K'])
            K[:2, :3] /= factor
            pose = np.array(meta[num + '_Rt'])[:3, :4]
            R_cv2gl = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            pose = R_cv2gl.dot(pose)
            cam_pose = np.eye(4)
            cam_pose[:3, :3] = pose[:3, :3].T
            cam_pose[:3, 3] = -pose[:3, :3].T.dot(pose[:, 3])

            rays_o = cam_pose[:3, 3]
            rays_o *= scale
            rays_o = np.dot(Rt[:3, :3], rays_o.T).T + Rt[:3, 3]
            rays_o -= point
            rays_o[2] += 40
            rays_o /= 100

            r = cam_pose[:3, :3]
            r = np.dot(Rt[:3, :3], r)

            cam_pose[:3, 3] = rays_o
            cam_pose[:3, :3] = r
            # cam_pose[:3,:3] = 9
            intrinsics = np.eye(4)
            intrinsics[:3, :3] = K

            self.pose_all.append(torch.from_numpy(cam_pose).float())
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.ids.append(self.id_list.index(i))
            self.names.append(str(p.split('/')[-1]))
