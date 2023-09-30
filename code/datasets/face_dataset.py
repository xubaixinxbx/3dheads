import os
import torch
import numpy as np

import utils.general as utils
from utils import rend_util
from glob import glob

import re
import sys
from tqdm import tqdm
import xml.etree.ElementTree as ET
import json 
from collections import defaultdict
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
def load_xml(file_name):
	if not os.path.exists(file_name):
		return {}
	tree = ET.parse(file_name)
	root = tree.getroot()
	sens = {}
	for sensor in root.iter('sensor'):
		ID = sensor.attrib['id']
		resol = sensor.find('resolution')
		resol = (int(resol.attrib['height']), int(resol.attrib['width']))
		calib = sensor.find('calibration')
		K = np.identity(3)
		D = []
		for c in list(calib):
			if c.tag == 'f':
				K[0,0] = K[1,1] = float(c.text)
			elif c.tag == 'fx':
				K[0,0] = float(c.text)
			elif c.tag == 'fy':
				K[1,1] = float(c.text)
			elif c.tag == 'cx':
				K[0,2] = float(c.text)
				if True:# K[0,2] <= resol[1]/3.:
					K[0,2] += resol[1]/2.
			elif c.tag == 'cy':
				K[1,2] = float(c.text)
				if True:# K[1,2] <= resol[0]/3.:
					K[1,2] += resol[0]/2.
			elif c.tag == 'k1':
				D = D + [0]*max(1-len(D),0)
				D[0] = float(c.text)
			elif c.tag == 'k2':
				D = D + [0]*max(2-len(D),0)
				D[1] = float(c.text)
			elif c.tag == 'p1':
				D = D + [0]*max(3-len(D),0)
				D[2] = float(c.text)
			elif c.tag == 'p2':
				D = D + [0]*max(4-len(D),0)
				D[3] = float(c.text)
			elif c.tag == 'k3':
				D = D + [0]*max(5-len(D),0)
				D[4] = float(c.text)
			elif c.tag == 'k4':
				D = D + [0]*max(6-len(D),0)
				D[5] = float(c.text)
			elif c.tag == 'k5':
				D = D + [0]*max(7-len(D),0)
				D[6] = float(c.text)
			elif c.tag == 'k6':
				D = D + [0]*max(8-len(D),0)
				D[7] = float(c.text)
		if len(D) < 5 and len(D) > 0:
			D = D + [0]*(5-len(D))
		D = np.array(D, K.dtype) if len(D) > 0 else None
		sens[ID] = (K, D, resol)
	cams = {}
	for view in root.iter('camera'):
		ID = view.attrib['sensor_id']
		name = view.attrib['label']
		if not name[-4:].lower() in ['.jpg','.png']:
			name += '.jpg'
		if len(name[:-4]) == 1:
			name = '0' + name
		if view.find('transform') is None:
			continue
		cam2world = np.array([float(f) \
			for f in re.split(' |\n|\r|\t', view.find('transform').text) \
			if len(f) > 0], K.dtype).reshape(4,4)
		cams[name] = (sens[ID][0], sens[ID][1], \
			cam2world.reshape(4,4), sens[ID][2])
		# Intrinsics, distCoeff, cam2world_cv, height_width
	return	cams
def load_obj(obj_name):
	if obj_name[-4:].lower() != '.obj' or not os.path.exists(obj_name):
		return
	with open(obj_name, 'r') as fp:
		text = fp.readlines()
	v = []; fv = []
	pbar = tqdm(text)
	if hasattr(pbar, 'set_description'):
		pbar.set_description('Loading %s' % os.path.basename(obj_name))
	for line in pbar:
		line = [s for s in line.strip().split(' ') if len(s) > 0]
		if len(line) >= 3 and line[0] == 'v':
			v += [[float(f) for f in line[1:]]]
		elif len(line) >= 4 and line[0] == 'f':
			f = [[],[],[]]
			for i in range(1, len(line)):
				l = line[i].split('/')
				for j in range(3):
					if j < len(l) and len(l[j]) > 0:
						f[j] += [int(l[j]) - 1]
			fv += [[f[0][0], f[0][i-1], f[0][i]] for i in range(2,len(f[0]))]
	v  = torch.from_numpy(np.array(v, np.float32))
	fv = torch.from_numpy(np.array(fv, np.int64))
	return	v, fv

class FaceDataset(torch.utils.data.Dataset):

    def __init__(self, **kwargs):
        '''
        kwargs:
        conf: dataset config
        '''
        super(FaceDataset, self).__init__()
        
        self.conf = kwargs['conf']
        self.img_res = self.conf['img_res']
        self.total_pixels = self.img_res[0] * self.img_res[1]
        self.num_id = 0
        self.white_flag = self.conf['white_flag']

        self.sampling_idx = None
        self.scan_id = self.conf.get_int('scan_id')
        self.cam_scale = self.conf.get_float('cam_scale')
        self.data_dir = self.conf['data_dir']
        assert os.path.exists(self.data_dir), "Data directory is empty"
        image_dir = self.data_dir
        # there are 15 wrinkle id and 15 smooth id in base_id.txt
        pr_30id_path = self.conf.get_string('id_path', default='./confs/id.txt')
        id_list = np.loadtxt(pr_30id_path, dtype=np.str_).tolist()
        id_list.sort()

        self.intrinsics_all = []
        self.pose_all = []
        self.rgb_images_path = []
        self.rgb_images_all = []
        self.ids = []
        self.names = []
        self.id_list = id_list
        print('PR-facedataset load {} identities: {}'.format(len(id_list), id_list))
        init_view_path = self.conf.get_string('init_view_path', default=None)
        sparse_view_num = self.conf.get_int('sparse_view_num', default=-1)
    
        # fine-tune config
        self.fine_tune_flag = self.conf.get_bool('fine_tune_flag',default=False)
        if self.fine_tune_flag:
            print('for fine-tune, id will be added in previous id_list as last item')
            id_list.append(str(self.scan_id))
            # select random view for fine-tune
            view_array = np.arange(1, 31)
            sparse_view_num = self.conf.get_int('fine_tune_view')
            select_view = np.random.choice(view_array, sparse_view_num, replace=False).tolist()
            if 15 not in select_view:
                select_view[-1]=15
            print('{} fine-tune view: {}'.format(self.scan_id,select_view))
        
            
        if sparse_view_num != -1:
            assert init_view_path != None, print('there is no init_view_path specified, when assign a sparse_view_num')
            with open(init_view_path,'r') as f:
                sparse_info = json.load(f)
                assert sparse_info['sparse_view_num'] == sparse_view_num, print('load sparse view wrong')
                sparse_info = sparse_info['info']
    
        for id in id_list:
            instance_image_path = os.path.join(image_dir, id)
            instance_list = os.listdir(instance_image_path)
            if self.scan_id != -1 and not str(self.scan_id) == id:
                # if given a scan_id to specify, we filter other id
               continue
            if sparse_view_num != -1:
                if init_view_path is None:
                    view_array = np.array([int(i[:-4]) for i in instance_list if 'png' in i[-4:]])
                    sparse_view_list = np.random.choice(view_array, sparse_view_num, replace=False).tolist()
                    if 15 not in sparse_view_list:
                        # make sure the frontal face image in datset
                        sparse_view_list[-1] = 15
                else:
                    sparse_view_list = sparse_info[id]
                if self.fine_tune_flag and init_view_path is None:
                    sparse_view_list = select_view
            else:
                # if no view specified, we use all views~[1, 30] in PR-dataset
                sparse_view_list = [i for i in range(1, 31)]
            print('{} training views: {}'.format(id,sparse_view_list))
            for item in instance_list:
                file_name = os.path.join(instance_image_path, item)
                if '.xml' in item:
                    cameras = load_xml(file_name)
                    pbar = tqdm(cameras.items())
                    if hasattr(pbar, 'set_description'):
                        pbar.set_description('Loading Dataset: %s' % id)    
                    for name, (K, dist, c2w, hw) in pbar:
                        if self.fine_tune_flag and init_view_path is None:
                            if int(name.split('.')[0]) not in select_view:
                                continue
                        fname = os.path.join(instance_image_path, name) 
                        if not os.path.exists(fname):
                            # print(fname,"dont exist")
                            continue
                        if sparse_view_num != -1 and int(name[:-4]) not in sparse_view_list:
                            continue
            
                        self.rgb_images_path.append(fname)
                        rgb = rend_util.load_rgb(fname, white_flag=self.white_flag)
                        rgb = rgb[:3,:,:].reshape(3, -1).transpose(1, 0)
                        rgb = torch.from_numpy(rgb).float()
                        self.rgb_images_all.append(rgb)
                        self.names.append(name)
                        self.ids.append(id_list.index(id))
                        c2w = c2w[...,:3,:4].reshape(-1,3,4)
                        c2w = np.concatenate([c2w, [[[0,0,0,1]]]*len(c2w)], 1)
                        # import pdb; pdb.set_trace()
                        c2w[:,:3,:4] = c2w[:,:3,:4]*self.cam_scale
                        # c2w[:,:3,3] = c2w[:,:3,3]*self.cam_scale
                        self.pose_all.append(c2w.astype(np.float32))
                        K = K[...,:2,:3].reshape(-1,2,3)
                        K = np.concatenate([K, [[[0,0,1]]]*len(K)], 1)
                        self.intrinsics_all.append(K.astype(np.float32))         
        self.pose_all = torch.from_numpy(np.concatenate(self.pose_all, 0))
        self.intrinsics_all = torch.from_numpy(np.concatenate(self.intrinsics_all, 0))
        self.ids = torch.from_numpy(np.array(self.ids)) # [1,1,0,0,0,1,2]
        
    def __len__(self):
        return len(self.rgb_images_path)

    def __getitem__(self, idx):
        idx = idx % len(self.rgb_images_path)    
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
    
    def change_sampling_idx(self, sampling_size):
        self.sampling_size = sampling_size
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]
    
    def get_specific_item(self, people_order=-1, cam_id=-1):
        # people_order: [0, 29]
        # cam_id \in [1, 30]
        if people_order == -1:
            person_id = str(self.scan_id) # person_id means: 340 / 393 etc...
            idx = self.id_list.index(person_id)
        else:
            person_id = self.id_list[people_order]
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
        idx = torch.from_numpy(np.array([people_order]))
        rgb = rend_util.load_rgb(select_img_path, white_flag=self.white_flag)
        rgb = rgb[:3,:,:].reshape(3, -1).transpose(1, 0)
        rgb = torch.from_numpy(rgb).float()
        return c2w_all, intrinsics_all, idx, rgb.unsqueeze(0)

    def get_scale_mat(self):
        print("eval call get_scale_mat in FaceDataset")
        return None 
    
    def get_id_list(self):
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