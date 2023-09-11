import sys
import cv2
import os.path as op
import numpy as np

import torch
import torchvision.transforms as transforms
from src.datasets.utils import *
from src.modeling._smpl import SMPL, SMPLH36M


class mmBodyDataset():
    def __init__(self, args):
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.args = args
        self.img_res = 224
        self.data_path = op.join(args.data_path, 'train') if args.train else op.join(args.data_path, 'test')
        self.seq_idxes = eval(args.seq_idxes) if args.seq_idxes else range(20)
        self.inputs = args.inputs
        self.init_index_map()
            
    def init_index_map(self):
        self.index_map = [0,]
        if self.args.train:
            seq_dirs = ['sequence_{}'.format(i) for i in self.seq_idxes]
            self.seq_paths = [op.join(self.data_path, p) for p in seq_dirs]
        else:
            seq_dirs = ['sequence_{}'.format(i) for i in range(2)]
            self.seq_paths = [op.join(self.data_path, self.args.test_scene, p) for p in seq_dirs]
        
        print('Data path: ', self.seq_paths)

        self.seq_loaders = {}
        for path in self.seq_paths:
            # init result loader, reindex
            seq_loader = mmBodySequenceLoader(path, self.args.skip_head, self.args.skip_tail, resource=self.inputs)
            self.seq_loaders.update({path:seq_loader})
            self.index_map.append(self.index_map[-1] + len(seq_loader))
            
    def global_to_seq_index(self, global_idx:int):
        for seq_idx in range(len(self.index_map)-1):
            if global_idx in range(self.index_map[seq_idx], self.index_map[seq_idx+1]):
                frame_idx = global_idx - self.index_map[seq_idx]
                return seq_idx, frame_idx
        raise IndexError
    
    def process_image(self, image, joints_3d=None, trans_mat=None, need_crop=False, need_normal=True):
        if need_crop:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # crop person area using joints
            image, box_min, box_max = crop_image(joints_3d, image, trans_mat, square=True, return_box=True)
            image = cv2.resize(image, [self.img_res, self.img_res], interpolation=cv2.INTER_LINEAR)
        image = np.transpose(image.astype('float32'), (2,0,1))/255.0
        image = torch.from_numpy(image).float()
        if need_normal:
            image = self.normalize_img(image)
        if need_crop:
            return image, box_min, box_max
        return image
    
    def process_pcl(self, pcl, joints, padding_points=1024, mask_limbs=False, need_filter=False, 
                    pelvis_idx=0, mask_ratio=0.99, num_mask_limbs=None):
        # filter person pcl using joints
        if need_filter:
            pcl = filter_pcl(joints, pcl)
        if mask_limbs:
            # mask points
            lowwer_body_mask = pcl[:, 2] < joints[1,2] - 0.2
            limb_joints = [15, 18, 19, 20, 21]
            if num_mask_limbs is None:
                mask_ind = gen_random_indices(num_indices=len(limb_joints), min_size=0, includ_right=True)
            else:
                mask_ind = gen_random_indices(num_indices=len(limb_joints), size=num_mask_limbs)
            if len(mask_ind):
                limb_ind = np.array(limb_joints)[mask_ind]
                dists = np.minimum.reduce([np.linalg.norm(pcl[:,:3] - joints[i], axis=1) for i in limb_ind])
                points_mask_ind = np.logical_or(dists < 0.2, lowwer_body_mask)
            else:
                points_mask_ind = lowwer_body_mask
            mask_points = pcl[points_mask_ind]
            sel_res_ind = gen_random_indices(len(mask_points), max_ratio=1-mask_ratio)
            pcl = np.vstack((pcl[~points_mask_ind], mask_points[sel_res_ind]))
            
        if not pcl.shape[0]:
            return torch.zeros((padding_points, 6)).float()
        # normalize pcl
        pcl[:,:3] -= joints[pelvis_idx]
        # padding pcl
        pcl = pad_pcl(pcl, padding_points)
        pcl = torch.from_numpy(pcl).float()
        return pcl
    
    def gen_mask(self, num_mask, mask_ratio):
        mask = np.ones((num_mask))
        if self.args.train:
            indices = gen_random_indices(num_mask, max_ratio=mask_ratio)
            mask[indices] = 0.0
        return torch.from_numpy(mask).float()
    
    def load_data(self, seq_loader, frame_idx):
        frame = seq_loader[frame_idx]
        # get mesh parameters
        mesh = dict(frame['mesh'])
        joints_3d = mesh['joints'][:22]
        verts = mesh['vertices']
        result = {}
        
        # masking tokens
        joint_mask = self.gen_mask(num_mask=22, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
        vert_mask = self.gen_mask(num_mask=655, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
        pelvis_idx = 0
            
        # process image
        orig_imgs = {}
        trans_mat = {}
        bbox = {}
        joints_2d = {}
        for image in self.args.input_dict['image']:
            orig_imgs[image] = frame[image].copy()
            result[image], bbox_min, bbox_max = self.process_image(frame[image], joints_3d, seq_loader.calib[image], need_crop=True)
            trans_mat[image] = seq_loader.calib[image]
            bbox[image] = np.concatenate([bbox_min, bbox_max]).reshape(-1)
            resize_ratio = [self.img_res, self.img_res] / (bbox_max - bbox_min)
            if self.args.joints_2d_loss:
                joints_2d[image] = ((project_pcl(joints_3d, trans_mat=seq_loader.calib[image], intrinsic=INTRINSIC[image]) 
                                     - bbox[image][0:2]) * resize_ratio).astype(np.int64)

        # process depth pcl
        for depth in self.args.input_dict['depth']:
            result[depth] = self.process_pcl(frame[depth], joints_3d, self.args.num_points, mask_limbs=self.args.mask_limbs, 
                                             pelvis_idx=pelvis_idx, mask_ratio=self.args.point_mask_ratio, need_filter=True,
                                             num_mask_limbs=self.args.num_mask_limbs)
            trans_mat[depth] = seq_loader.calib[depth]
            
        # process radar pcl
        radar_pcl = torch.Tensor([])
        if 'radar0' in self.args.inputs:
            radar_pcl = frame['radar0']
            radar_pcl[:,3:] /= np.array([5e-38, 5., 150.])
            result['radar0'] = self.process_pcl(radar_pcl, joints_3d, 1024, need_filter=True, pelvis_idx=pelvis_idx)
            trans_mat['radar0'] = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]).float()
                            
        # return result
        result['joints_3d'] = torch.from_numpy(joints_3d).float()
        result['root_pelvis'] = torch.from_numpy(joints_3d[pelvis_idx]).float()
        result['vertices'] = torch.from_numpy(verts).float()

        result['joint_mask'] = joint_mask
        result['vert_mask'] = vert_mask

        result['trans_mat'] = trans_mat
        result['orig_img'] = orig_imgs
        result['bbox'] = bbox
        result['joints_2d'] = joints_2d
        
        return result

    def __len__(self):
        return self.index_map[-1]

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.global_to_seq_index(idx)
        seq_path = self.seq_paths[seq_idx]
        seq_loader =self.seq_loaders[seq_path]
        
        try:
            result = self.load_data(seq_loader, frame_idx)
        except Exception as err:
            print(idx, seq_path, frame_idx)
            print(err)
            sys.exit(1)
        return result
        # return self.load_data(seq_loader, frame_idx)
    

class HuMManDataset(mmBodyDataset):
    def __init__(self, args, **kwargs):
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.args = args
        self.img_res = 224
        self.data_path = op.join(args.data_path, 'train') if args.train else op.join(args.data_path, 'test')
        self.resource = args.inputs
        self.smpl_model = kwargs.get('smpl_model', SMPL())
        self.init_index_map()
        
    def init_index_map(self):
        self.index_map = [0,]
        self.seq_paths = [op.join(self.data_path, p) for p in os.listdir(self.data_path)]
        
        self.seq_loaders = {}
        for path in self.seq_paths:
            # init result loader, reindex
            seq_loader = HuMManSequenceLoader(path, resource=self.resource)
            self.seq_loaders.update({path:seq_loader})
            self.index_map.append(self.index_map[-1] + len(seq_loader))
    
    def load_data(self, seq_loader, frame_idx):
        frame = seq_loader[frame_idx]
        # get mesh parameters
        smpl_para = dict(frame['mesh'])
        pose = np.concatenate((smpl_para['transl'], smpl_para['global_orient'], smpl_para['body_pose']))
        betas = smpl_para['betas']
        mesh = self.smpl_model(torch.from_numpy(pose[None]).float(), torch.from_numpy(betas[None]).float())
        joints = copy2cpu(mesh['joints'][0])
        verts = copy2cpu(mesh['vertices'][0])
        result = {}
        
        # masking tokens
        joint_mask = self.gen_mask(num_mask=24, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
        vert_mask = self.gen_mask(num_mask=431, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
        pelvis_idx = 0
            
        trans_mat = {}
        # process image
        for image in self.args.input_dict['image']:
            result[image] = self.process_image(frame[image], joints, seq_loader.calib[image])
            trans_mat[image] = seq_loader.calib[image]

        # process depth pcl
        for depth in self.args.input_dict['depth']:
            result[depth] = self.process_pcl(frame[depth], joints, self.args.num_points, mask_limbs=False, pelvis_idx=pelvis_idx)
            trans_mat[depth] = seq_loader.calib[depth]

        # return result
        result['joints_3d'] = torch.from_numpy(joints).float()
        result['root_pelvis'] = torch.from_numpy(joints[pelvis_idx]).float()
        result['vertices'] = torch.from_numpy(verts).float()
        result['trans_mat'] = trans_mat

        result['joint_mask'] = joint_mask
        result['vert_mask'] = vert_mask
        
        return result


class BEHAVEDataset(mmBodyDataset):
    def __init__(self, args, **kwargs):
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.args = args
        self.img_res = 224
        self.data_path = op.join(args.data_path, 'train') if args.train else op.join(args.data_path, 'test')
        self.resource = args.inputs
        self.smpl_model = kwargs.get('smpl_model', SMPL())
        self.init_index_map()
        
    def init_index_map(self):
        self.index_map = [0,]
        self.seq_paths = [op.join(self.data_path, p) for p in os.listdir(self.data_path)]
        self.seq_paths.sort()
        
        self.seq_loaders = {}
        for path in self.seq_paths:
            # init result loader, reindex
            seq_loader = BEHAVESequenceLoader(path, resource=self.resource)
            self.seq_loaders.update({path:seq_loader})
            self.index_map.append(self.index_map[-1] + len(seq_loader))
    
    def load_data(self, seq_loader, frame_idx):
        frame = seq_loader[frame_idx]
        # get mesh parameters
        smpl_para = dict(frame['mesh'])
        pose = np.concatenate((smpl_para['trans'], smpl_para['pose']))
        betas = smpl_para['betas']
        mesh = self.smpl_model(torch.from_numpy(pose[None]).float(), torch.from_numpy(betas[None]).float())
        joints = copy2cpu(mesh['joints'][0])
        verts = copy2cpu(mesh['vertices'][0])
        result = {}
        
        # masking tokens
        joint_mask = self.gen_mask(num_mask=24, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
        vert_mask = self.gen_mask(num_mask=431, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
        pelvis_idx = 0
            
        trans_mat = {}
        # process image
        for image in self.args.input_dict['image']:
            result[image] = self.process_image(frame[image], joints, seq_loader.calib[image])
            trans_mat[image] = seq_loader.calib[image]

        # process depth pcl
        for depth in self.args.input_dict['depth']:
            result[depth] = self.process_pcl(frame[depth], joints, self.args.num_points, mask_limbs=False, pelvis_idx=pelvis_idx)
            trans_mat[depth] = seq_loader.calib[depth]

        # return result
        result['joints_3d'] = torch.from_numpy(joints).float()
        result['root_pelvis'] = torch.from_numpy(joints[pelvis_idx]).float()
        result['vertices'] = torch.from_numpy(verts).float()
        result['trans_mat'] = trans_mat

        result['joint_mask'] = joint_mask
        result['vert_mask'] = vert_mask

        return result
    

class Human36MDataset(BEHAVEDataset):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.smpl_model = SMPLH36M()
        
    def init_index_map(self):
        self.index_map = [0,]
        self.seq_paths = [op.join(self.data_path, p) for p in os.listdir(self.data_path) if p[0]=='s']
        # self.seq_paths = [op.join(self.data_path, p) for p in os.listdir(self.data_path) if p=='s_01_act_02_subact_01']
        self.seq_paths.sort()
        
        self.seq_loaders = {}
        for path in self.seq_paths:
            # init result loader, reindex
            seq_loader = Human36MSequenceLoader(path, resource=self.resource)
            self.seq_loaders.update({path:seq_loader})
            self.index_map.append(self.index_map[-1] + len(seq_loader))

    def load_data(self, seq_loader, frame_idx):
        frame = seq_loader[frame_idx]
        # get mesh parameters
        smpl_para = dict(frame['mesh'])
        pose = smpl_para['pose']
        betas = smpl_para['shape']
        verts = self.smpl_model(torch.from_numpy(pose[None]).float(), torch.from_numpy(betas[None]).float())
        joints_3d = copy2cpu(self.smpl_model.get_joints(verts)[0])
        joints_h36m = copy2cpu(self.smpl_model.get_h36m_joints(verts)[0])
        verts = copy2cpu(verts[0]) - joints_h36m[0] + smpl_para['trans'][0]
        joints_3d = joints_3d - joints_h36m[0] + smpl_para['trans'][0]
        joints_h36m = joints_h36m - joints_h36m[0] + smpl_para['trans'][0]
        result = {}
        
        # masking tokens
        joint_mask = self.gen_mask(num_mask=24, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
        vert_mask = self.gen_mask(num_mask=431, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
        pelvis_idx = 0
            
        trans_mat = {}
        joints_2d = {}
        orig_imgs = {}
        # process image
        for image in self.args.input_dict['image']:
            orig_imgs[image], box_min, box_max = self.process_image(frame[image], joints_3d, seq_loader.calib[image],
                                                                    need_crop=True, need_normal=False)
            result[image] = self.normalize_img(orig_imgs[image])
            trans_mat[image] = seq_loader.calib[image]
            box_min, box_max = box_min[::-1], box_max[::-1]
            joint_2d = project_pcl(joints_3d, trans_mat=seq_loader.calib[image], image_size=[1000,1002], return_int=False)
            joint_2d[:,:2] = 2. * (joint_2d[:,:2] - box_min) / (box_max - box_min) - 1
            joints_2d[image] = torch.from_numpy(joint_2d).float()
        if self.args.joints_2d_loss:
            joints_2d_trans_mat = seq_loader.calib[self.args.joints_2d_loss]
            joints_3d = (joints_3d - joints_2d_trans_mat['t']) @ joints_2d_trans_mat['R']
            joints_h36m = (joints_h36m - joints_2d_trans_mat['t']) @ joints_2d_trans_mat['R']
            verts = (verts - joints_2d_trans_mat['t']) @ joints_2d_trans_mat['R']
        joints_3d = np.hstack((joints_3d, np.ones([joints_3d.shape[0], 1])))
        
        # return result
        result['joints_3d'] = torch.from_numpy(joints_3d).float()
        result['root_pelvis'] = torch.from_numpy(joints_h36m[pelvis_idx]).float()
        result['vertices'] = torch.from_numpy(verts).float()
        result['joints_2d'] = joints_2d
        result['trans_mat'] = trans_mat
        result['orig_img'] = orig_imgs

        result['joint_mask'] = joint_mask
        result['vert_mask'] = vert_mask

        return result

class CLIFFDataset(mmBodyDataset):
    def __init__(self, args, data_path):
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.args = args
        self.img_res = 224
        self.data_path = data_path
        self.labels = dict(np.load(os.path.join(self.data_path, 'cliffGT.npz')))
        self.smpl_model = SMPL()
        
    def __len__(self):
        return len(self.labels['imgname'])

    def __getitem__(self, idx):
        result = {}
        pose = np.concatenate((self.labels['global_t'][idx], self.labels['pose'][idx]))
        betas = self.labels['shape'][idx]
        mesh = self.smpl_model(torch.from_numpy(pose[None]).float(), torch.from_numpy(betas[None]).float())
        joints_3d = copy2cpu(mesh['joints'][0])
        verts = copy2cpu(mesh['vertices'][0])
        # joints_3d = self.labels['S'][idx]
        joints_2d = self.labels['part'][idx]
        
        # masking tokens
        joint_mask = self.gen_mask(num_mask=24, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
        vert_mask = self.gen_mask(num_mask=431, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
        pelvis_idx = 0
        
        orig_imgs = {}
        trans_mat = {}
        joints_2d_dict = {}
        img = cv2.imread(os.path.join(self.data_path, self.labels['imgname'][idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        box_scale = int(self.labels['scale'][idx] * 100)
        center = self.labels['center'][idx].astype(int)
        box_min, box_max = center - box_scale, center + box_scale
        box_min, box_max = [np.where(p > 0, p, 0) for p in [box_min, box_max]]
        box_min, box_max = [np.where(p < img.shape[1::-1], p, img.shape[1::-1]) for p in [box_min, box_max]]
        crop_img = img[box_min[1]:box_max[1], box_min[0]:box_max[0]]
        crop_img = cv2.resize(crop_img, [self.img_res, self.img_res], interpolation=cv2.INTER_LINEAR)
        trans_img = self.process_image(crop_img)
        result[self.args.joints_2d_loss] = trans_img
        crop_img = np.transpose(crop_img.astype('float32'), (2,0,1))/255.0
        crop_img = torch.from_numpy(crop_img).float()
        orig_imgs[self.args.joints_2d_loss] = crop_img
        
        # joints_2d[:,:2] = 2. * (joints_2d[:,:2] - box_min) / (box_max - box_min) - 1
        # joints_2d = torch.from_numpy(joints_2d).float()
        joints_2d = torch.zeros(joints_2d.shape).float()
        joints_2d_dict[self.args.joints_2d_loss] = joints_2d
        # process image
        for image in self.args.input_dict['image']:
            trans_mat[image] = torch.eye(4,4)
            if image != self.args.joints_2d_loss:
                result[image] = torch.zeros_like(trans_img)
                orig_imgs[image] = torch.zeros_like(crop_img)
                joints_2d_dict[image] = torch.zeros_like(joints_2d)
        joints_3d = np.hstack((joints_3d, np.ones([joints_3d.shape[0], 1])))
        
        # return result
        result['joints_3d'] = torch.from_numpy(joints_3d).float()
        result['root_pelvis'] = torch.from_numpy(joints_3d[pelvis_idx]).float()
        result['vertices'] = torch.from_numpy(verts).float()
        result['joints_2d'] = joints_2d_dict
        result['trans_mat'] = trans_mat
        result['orig_img'] = orig_imgs

        result['joint_mask'] = joint_mask
        result['vert_mask'] = vert_mask
        
        return result
