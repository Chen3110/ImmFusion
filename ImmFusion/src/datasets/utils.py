import json
import cv2
import os
import torch
import numpy as np
import plyfile

INTRINSIC = {
    'image0': np.asarray([
        [969.48345947265625,    0,                  1024.9678955078125],
        [0,                     968.99578857421875, 781.4013671875],
        [0,                     0,                  1]]),
    'image1': np.asarray([
        [972.07073974609375,    0,                  1021.4869384765625  ],
        [0,                     971.651123046875,   780.25439453125     ],
        [0,                     0,                  1                   ]
    ])
}


class mmBodySequenceLoader(object):
    def __init__(self, seq_path: str, skip_head: int = 0, skip_tail: int = 0, 
                 resource=['image0','image1','depth0','depth1','radar0']) -> None:
        self.seq_path = seq_path
        self.skip_head = skip_head
        self.skip_tail = skip_tail
        self.resource = resource
        # load transformation matrix
        with open(os.path.join(seq_path, 'calib.txt')) as f:
            calib = eval(f.readline())
        self.calib = {
            'image0':calib['kinect_master'],
            'image1':calib['kinect_sub'],
            'depth0':calib['kinect_master'],
            'depth1':calib['kinect_sub'],
        }

    def __len__(self):
        return len(os.listdir(os.path.join(self.seq_path, 'mesh'))) - self.skip_head - self.skip_tail

    def __getitem__(self, idx: int):
        result = {}
        if 'radar0' in self.resource:
            result['radar0'] = np.load(os.path.join(
                self.seq_path, 'radar', 'frame_{}.npy'.format(idx+self.skip_head)))
        if 'image0' in self.resource:
            result['image0'] = cv2.imread(os.path.join(
                self.seq_path, 'image', 'master', 'frame_{}.png'.format(idx+self.skip_head)))
            result['bbox0'] = np.load(os.path.join(
                self.seq_path, 'bounding_box', 'master', 'frame_{}.npy'.format(idx+self.skip_head)))
        if 'image1' in self.resource:
            result['image1'] = cv2.imread(os.path.join(
                self.seq_path, 'image', 'sub', 'frame_{}.png'.format(idx+self.skip_head)))
            result['bbox1'] = np.load(os.path.join(
                self.seq_path, 'bounding_box', 'sub', 'frame_{}.npy'.format(idx+self.skip_head)))
        if 'depth0' in self.resource:
            result['depth0'] = np.load(os.path.join(
                self.seq_path, 'depth_pcl', 'master', 'frame_{}.npy'.format(idx+self.skip_head)))
        if 'depth1' in self.resource:
            result['depth1'] = np.load(os.path.join(
                self.seq_path, 'depth_pcl', 'sub', 'frame_{}.npy'.format(idx+self.skip_head)))
        result['mesh'] = np.load(os.path.join(
            self.seq_path, 'mesh', 'frame_{}.npz'.format(idx+self.skip_head)))
            
        return result


class HuMManSequenceLoader(object):
    def __init__(self, seq_path: str, skip_head: int = 0, skip_tail: int = 0, 
                 resource=['image0','image1','depth0','depth1'], ori_data=False) -> None:
        self.seq_path = seq_path
        self.skip_head = skip_head
        self.skip_tail = skip_tail
        self.resource = resource
        self.ori_data = ori_data
        # load transformation matrix
        with open(os.path.join(seq_path, 'cameras.json'), 'r', encoding='utf8') as f:
            calib = json.load(f)
        for k, v in calib.items():
            calib[k]['t'] = -np.array(v.pop('T')) @ np.array(v['R'])
            calib[k]['R'] = np.array(v['R']).T
            calib[k]['K'] = np.array(v['K'])
        self.calib = {}
        for i in range(10):
            self.calib.update({
                'image{}'.format(i):calib['kinect_color_00{}'.format(i)],
                'depth{}'.format(i):calib['kinect_depth_00{}'.format(i)],
            })
    def __len__(self):
        return len(os.listdir(os.path.join(self.seq_path, 'smpl_params'))) - self.skip_head - self.skip_tail

    def __getitem__(self, idx: int):
        result = {}
        for i in [0,3,6,9]:
            result.update({
                'image{}'.format(i):cv2.imread(os.path.join(self.seq_path, 'crop_image', 'kinect_00{}'.format(i), 
                                                            '%06d.png' % ((idx+self.skip_head)*6))),
                'depth{}'.format(i):np.load(os.path.join(self.seq_path, 'depth_pcl', 'kinect_00{}'.format(i), 
                                                            '%06d.npy' % ((idx+self.skip_head)*6)))
            })
        result['mesh'] = np.load(os.path.join(
            self.seq_path, 'smpl_params', '%06d.npz' % ((idx+self.skip_head)*6)))
            
        return result

class BEHAVESequenceLoader(object):
    def __init__(self, seq_path: str, skip_head: int = 0, skip_tail: int = 0, 
                 resource=['image0','image1','depth0']) -> None:
        self.seq_path = seq_path
        self.skip_head = skip_head
        self.skip_tail = skip_tail
        self.resource = resource
        # load transformation matrix
        with open(os.path.join(seq_path, 'cameras.json'), 'r', encoding='utf8') as f:
            calib = json.load(f)
        self.calib = {}
        for i in range(4):
            if 'image{}'.format(i) in self.resource:
                self.calib.update({
                    'image{}'.format(i):calib['kinect{}'.format(i)],
                    'depth{}'.format(i):calib['kinect{}'.format(i)],
                })

    def __len__(self):
        return len(os.listdir(os.path.join(self.seq_path, 'smpl_params'))) - self.skip_head - self.skip_tail
    
    def __getitem__(self, idx: int):
        result = {}
        for i in range(4):
            if 'image{}'.format(i) in self.resource:
                result.update({
                    'image{}'.format(i):cv2.imread(os.path.join(self.seq_path, 'crop_image', 'kinect{}'.format(i), 
                                                                '{}.png'.format(idx+self.skip_head))),
                })
            if 'depth{}'.format(i) in self.resource:
                result.update({
                    'depth{}'.format(i):np.load(os.path.join(self.seq_path, 'depth_pcl', 'kinect{}'.format(i), 
                                                                '{}.npy'.format(idx+self.skip_head)))
                })
        result['mesh'] = np.load(os.path.join(
            self.seq_path, 'smpl_params', '{}.npz'.format(idx+self.skip_head)))
            
        return result

class Human36MSequenceLoader(object):
    def __init__(self, seq_path: str, skip_head: int = 0, skip_tail: int = 0, 
                 resource=['image1','image2','image3','image4']) -> None:
        self.seq_path = seq_path
        self.skip_head = skip_head
        self.skip_tail = skip_tail
        self.resource = resource
        # load transformation matrix
        with open(os.path.join(seq_path, 'cameras.json'), 'r', encoding='utf8') as f:
            calib = json.load(f)
        self.calib = {}
        for i in range(1,5):
            if 'image{}'.format(i) in self.resource:
                self.calib.update({
                    'image{}'.format(i):calib['camera{}'.format(i)],
                })
    def __len__(self):
        return len(os.listdir(os.path.join(self.seq_path, 'smpl_params'))) - self.skip_head - self.skip_tail

    def __getitem__(self, idx: int):
        result = {}
        for i in range(1,5):
            if 'image{}'.format(i) in self.resource:
                result.update({
                    'image{}'.format(i):cv2.imread(os.path.join(self.seq_path, 'orig_image', 'camera{}'.format(i), 
                                                                '{}.png'.format(idx+self.skip_head))),
                })
        result['mesh'] = np.load(os.path.join(self.seq_path, 'smpl_params', '{}.npz'.format(idx+self.skip_head)))
        result['joints_h36m'] = np.load(os.path.join(self.seq_path, 'joints_h36m', '{}.npy'.format(idx+self.skip_head)))
            
        return result

class Human36MDatasetLoader(object):
    def __init__(self, seq_path: str, calib, smpl_para=None, joints_3d=None, skip_head=0, skip_tail=0):
        self.seq_path = seq_path
        self.skip_head = skip_head
        self.skip_tail = skip_tail
        self.seq_name = str.split(seq_path, '/')[-1]
        self.calib = calib
        self.smpl_frames = list(smpl_para.keys())
        self.smpl_frames.sort(key=lambda x:int(x))
        self.smpl_para = smpl_para
        self.joints_3d = joints_3d

    def __len__(self):
        return len(self.joints_3d) - self.skip_head - self.skip_tail
    
    def __getitem__(self, idx: int):
        result = {}
        result['joints_3d'] = self.joints_3d[idx]
        for cam in range(1,5):
            result['image{}'.format(cam)] = cv2.imread(os.path.join(
                '{}{}'.format(self.seq_path, cam), '{}{}_{:0>6d}.jpg'.format(self.seq_name, cam, idx+1)))
        if str(idx) in self.smpl_frames:
            result['smpl_para'] = self.smpl_para[str(idx)]
            result['has_mesh'] = 1
        else:
            result['has_mesh'] = 1
            
        return result
    
def conv_ply_2_arr(ply_path, return_rgb=True):
    plydata = plyfile.PlyData.read(ply_path)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    pcl = np.column_stack((x, y, z))
    if return_rgb:
        r = plydata['vertex']['red'] / 255
        g = plydata['vertex']['green'] / 255
        b = plydata['vertex']['blue'] / 255
        pcl = np.column_stack((x, y, z, r, g, b))
    return pcl
    
    
def copy2cpu(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def project_pcl(pcl, trans_mat=None, intrinsic=None, image_size=[1536,2048], return_int=True):
    if trans_mat is not None:
        pcl = (pcl - trans_mat['t']) @ np.array(trans_mat['R']).reshape(3, 3)
        intrinsic = trans_mat.get('K', None)
    intrinsic = np.array(intrinsic) if intrinsic is not None else INTRINSIC['image0']
    pcl_2d = ((pcl/pcl[:,2:3]) @ intrinsic.T)
    if return_int:
        pcl_2d = np.floor(pcl_2d).astype(int)[:,:2]
        pcl_2d[:, [0, 1]] = pcl_2d[:, [1, 0]]
        image_size = image_size[::-1]
    # filter out the points exceeding the image size
    image_size = np.array(image_size)
    pcl_2d[:,:2] = np.where(pcl_2d[:,:2]<image_size-1, pcl_2d[:,:2], image_size-1)
    pcl_2d = np.where(pcl_2d>0, pcl_2d, 0)
    
    return pcl_2d


def project_pcl_torch(pcl, trans_mat=None, intrinsic=None, image_size=[1536,2048], return_int=True):
    """
    Project pcl to the image plane
    """
    if trans_mat is not None:
        pcl = (pcl - trans_mat[:,None,:3,3]) @ trans_mat[:,:3,:3]
    intrinsic = intrinsic if intrinsic is not None else INTRINSIC['image0']
    pcl_2d = ((pcl/pcl[:,:,2:3]) @ torch.tensor(intrinsic).T.float().cuda())[:,:,:2]
    if return_int:
        pcl_2d = torch.floor(pcl_2d).long()
    pcl_2d[:,:,[0,1]] = pcl_2d[:,:,[1,0]]
    image_size = torch.tensor(image_size).cuda()
    pcl_2d = torch.where(pcl_2d<image_size-1, pcl_2d, image_size-1)
    pcl_2d = torch.where(pcl_2d>0, pcl_2d, 0)
    return pcl_2d

def depth_2_pcl(depth_image, intrinsic):
    fx, fy = intrinsic[0][0], intrinsic[1][1]
    cx, cy = intrinsic[0][2], intrinsic[1][2]
    
    depth = depth_image.astype(np.float32) / 1000.0
    height, width = depth.shape
    pixel_x, pixel_y = np.meshgrid(np.arange(width), np.arange(height))
    
    camera_x = (pixel_x - cx) * depth / fx
    camera_y = (pixel_y - cy) * depth / fy
    camera_z = depth

    point_cloud = np.stack([camera_x, camera_y, camera_z], axis=-1).reshape(-1, 3)
    
    return point_cloud

def filter_pcl(bounding_pcl: np.ndarray, target_pcl: np.ndarray, bound: float = 0.2, 
               offset: float = 0, height_axis: int = 2, facing_up=True):
    """
    Filter out the pcls of pcl_b that is not in the bounding_box of pcl_a
    """
    upper_bound = np.max(bounding_pcl[:, :3], axis=0) + bound
    lower_bound = np.min(bounding_pcl[:, :3], axis=0) - bound
    if facing_up:
        lower_bound[height_axis] += offset
    else:
        upper_bound[height_axis] -= offset
    
    # Compute the mask
    index = np.all((lower_bound <= target_pcl[:,:3]) & (target_pcl[:,:3] <= upper_bound), axis=-1)
    
    return target_pcl[index]

def pad_pcl(pcl, num_points=0, return_choices=False, ratio=None):
    if ratio is not None:
        num_points = int(pcl.shape[0] * ratio)
    if pcl.shape[0] > num_points:
        r = np.random.choice(pcl.shape[0], size=num_points, replace=False)
    elif not pcl.shape[0]:
        return np.empty(shape=pcl.shape)
    else:
        repeat, residue = num_points // pcl.shape[0], num_points % pcl.shape[0]
        r = np.random.choice(pcl.shape[0], size=residue, replace=False)
        r = np.concatenate([np.arange(pcl.shape[0]) for _ in range(repeat)] + [r], axis=0)
    if return_choices:
        return pcl[r, :], r
    return pcl[r, :]


def crop_image(joints:np.ndarray, image:np.ndarray, trans_mat:dict=None, visual:bool=False, 
               margin:float=0.2, square:bool=False, intrinsic=None, return_box:bool=False):
    """
    Crop the person area of image
    """
    # transform the joints to camera coordinate
    if trans_mat is not None:
        joints = (joints - trans_mat['t']) @ trans_mat['R']
        intrinsic = trans_mat.get('K', None)
    intrinsic = np.array(intrinsic) if intrinsic is not None else INTRINSIC['image0']
    joint_max = joints.max(axis=0) + margin
    joint_min = joints.min(axis=0) - margin
    # get 3d bounding box from joints
    box_3d = np.array([
        [joint_min[0], joint_min[1], joint_min[2]],
        [joint_min[0], joint_min[1], joint_max[2]],
        [joint_min[0], joint_max[1], joint_max[2]],
        [joint_min[0], joint_max[1], joint_min[2]],
        [joint_max[0], joint_max[1], joint_max[2]],
        [joint_max[0], joint_min[1], joint_max[2]],
        [joint_max[0], joint_max[1], joint_min[2]],
        [joint_max[0], joint_min[1], joint_min[2]],
    ])
    # project 3d bounding box to 2d image plane
    box_2d = project_pcl(box_3d, intrinsic=intrinsic)
    box_min = box_2d.min(0)
    box_max = box_2d.max(0)
    if square:
        size = box_max - box_min
        diff = abs(size[0] - size[1])//2
        if size[0] > size[1]:
            box_max[1] += diff
            box_min[1] -= diff
        elif size[0] < size[1]:
            box_max[0] += diff
            box_min[0] -= diff
    box_max = np.where(box_max<image.shape[:2], box_max, image.shape[:2])
    box_min = np.where(box_min>0, box_min, 0)
    # crop image
    crop_img = image[box_min[0]:box_max[0], box_min[1]:box_max[1]]

    if visual:
        cv2.namedWindow('img', 0)
        cv2.resizeWindow("img", 640, 480)
        cv2.rectangle(image, box_min[::-1], box_max[::-1], (0, 0, 255), 2)
        cv2.imshow('img', image)
        cv2.waitKey(0)

    if return_box:
        return crop_img, box_min, box_max
    return crop_img

def trans_mat_2_tensor(trans_mat):
    trans_mat_array = np.hstack((trans_mat['R'], np.array([trans_mat['t']]).T))
    trans_mat_array = np.vstack((trans_mat_array, [0,0,0,1]))
    return torch.tensor(trans_mat_array)

def trans_mat_2_dict(trans_mat):
    if isinstance(trans_mat, torch.Tensor):
        trans_mat = copy2cpu(trans_mat)
        trans_mat_dict = {
            'R': trans_mat[:3, :3],
            't': trans_mat[:3, 3]
        }
    return trans_mat_dict

def get_rgb_value(pcl, image, visual=False, ret_image=False):
    pcl_2d = project_pcl(pcl)
    pcl_color = image[pcl_2d[:, 0], pcl_2d[:, 1]]
    pcl_with_feature = np.hstack((pcl, pcl_color/255))

    if visual:
        image[pcl_2d[:, 0], pcl_2d[:, 1]] = [0, 255, 0]
        cv2.namedWindow('img', 0)
        cv2.resizeWindow("img", 640, 480)
        cv2.imshow('img', image)
        cv2.waitKey(0)

    return pcl_with_feature

def convert_square_image(image):
    """
     convert to square with slice
    """
    img_h, img_w, img_c = image.shape
    if img_h != img_w:
        long_side = max(img_w, img_h)
        short_side = min(img_w, img_h)
        loc = int(abs(img_w - img_h)/2)
        image = image.transpose((1, 0, 2)).copy() if img_w < img_h else image
        background = np.full((long_side, long_side, img_c), 255, np.uint8)
        background[loc: loc + short_side] = image[...]
        image = background.transpose((1, 0, 2)).copy() if img_w < img_h else background
    return image

def rodrigues_2_rot_mat(rvecs):
    batch_size = rvecs.shape[0]
    r_vecs = rvecs.reshape(-1, 3)
    total_size = r_vecs.shape[0]
    thetas = torch.norm(r_vecs, dim=1, keepdim=True)
    is_zero = torch.eq(torch.squeeze(thetas), torch.tensor(0.0))
    u = r_vecs / thetas

    # Each K is the cross product matrix of unit axis vectors
    # pyformat: disable
    zero = torch.autograd.Variable(torch.zeros(
        [total_size], device="cuda"))  # for broadcasting
    Ks_1 = torch.stack([zero, -u[:, 2],  u[:, 1]], axis=1)  # row 1
    Ks_2 = torch.stack([u[:, 2],  zero, -u[:, 0]], axis=1)  # row 2
    Ks_3 = torch.stack([-u[:, 1],  u[:, 0],  zero], axis=1)  # row 3
    # pyformat: enable
    Ks = torch.stack([Ks_1, Ks_2, Ks_3], axis=1)                  # stack rows

    identity_mat = torch.autograd.Variable(
        torch.eye(3, device="cuda").repeat(total_size, 1, 1))
    Rs = identity_mat + torch.sin(thetas).unsqueeze(-1) * Ks + \
        (1 - torch.cos(thetas).unsqueeze(-1)) * torch.matmul(Ks, Ks)
    # Avoid returning NaNs where division by zero happened
    R = torch.where(is_zero[:, None, None], identity_mat, Rs)

    return R.reshape(batch_size, -1)

def gen_random_indices(num_indices, size=None, min_size=None, max_size=None, min_ratio=0., max_ratio=1., includ_right=False):
    # generate random indices
    if num_indices <= 0:
        return []
    if size is None:
        if min_size is None:
            min_size = min_ratio*num_indices
        else:
            min_size = max(0, min_size)
        if max_size is None:
            max_size = max_ratio*num_indices
        else:
            max_size = min(num_indices, max_size)
        max_size_ = np.ceil(max_size) + 1 if includ_right else np.ceil(max_size)
        size = np.random.randint(min_size, max_size_)
    indices = np.random.choice(np.arange(num_indices), replace=False, size=int(size))
    return np.sort(indices)

def mosh_pose_transform(trans, root_orient, root_joint, trans_mat):
    mosh_offset = root_joint - trans
    new_trans = trans_mat['R'] @ (trans + mosh_offset) + trans_mat['t'] - mosh_offset
    orient_mat = trans_mat['R'] @ cv2.Rodrigues(root_orient)[0]
    new_orient = cv2.Rodrigues(orient_mat)[0]
    return new_trans, new_orient

def find_points_within_circle(pcl, center, dist, return_ind=False):
    dists = np.linalg.norm(pcl[:,:3] - center, axis=1)
    sele_indices = dists <= dist
    sel_pcl = pcl[sele_indices]
    if return_ind:
        return sel_pcl, sele_indices
    return sel_pcl

def find_points_within_cuboid(pcl, center, size, return_ind=False):
    mins = center - size / 2
    maxs = center + size / 2
    mask = np.all((mins <= pcl[:,:3]) & (pcl[:,:3] <= maxs), axis=-1)
    sel_pcl = pcl[mask]
    if return_ind:
        return sel_pcl, mask
    return sel_pcl