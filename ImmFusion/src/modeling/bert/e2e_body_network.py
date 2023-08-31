"""
"""

import copy
import torch
import torch
import yaml
from easydict import EasyDict
import numpy as np
import src.modeling.data.config as cfg
from src.modeling.pointnet2.pointnet2_modules import Pointnet2Backbone, PointnetSAModule
from src.modeling.bert.transformer import Transformer

class Graphormer_Body_Network(torch.nn.Module):
    '''
    End-to-end Graphormer network for human pose and mesh reconstruction from a single image.
    '''
    def __init__(self, args, config, backbone, trans_encoder, mesh_sampler):
        super(Graphormer_Body_Network, self).__init__()
        self.config = config
        self.config.device = args.device
        self.backbone = backbone
        self.trans_encoder = trans_encoder
        self.upsampling = torch.nn.Linear(655, 2619)
        self.upsampling2 = torch.nn.Linear(2619, 10475)
        self.cam_param_fc = torch.nn.Linear(3, 1)
        self.cam_param_fc2 = torch.nn.Linear(655, 250)
        self.cam_param_fc3 = torch.nn.Linear(250, 3)
        self.grid_feat_dim = torch.nn.Linear(1024, 2051)


    def forward(self, images, smpl, mesh_sampler, meta_masks=None, is_train=False):
        batch_size = images.size(0)
        # Generate T-pose template mesh
        template_pose = torch.zeros((1,69))
        template_pose[:,0] = 3.1416 # Rectify "upside down" reference mesh in global coord
        template_pose = template_pose.cuda(self.config.device)
        template_betas = torch.zeros((1,16)).cuda(self.config.device)
        template_res = smpl(template_pose, template_betas)
        template_vertices = template_res['vertices']

        # template mesh simplification
        template_vertices_sub = mesh_sampler.downsample(template_vertices)
        template_vertices_sub2 = mesh_sampler.downsample(template_vertices_sub, n1=1, n2=2)

        # template mesh-to-joint regression 
        template_3d_joints = template_res['joints'][:, :22]
        template_pelvis = template_3d_joints[:,0,:]
        # template_3d_joints = template_3d_joints[:,cfg.H36M_J17_TO_J14,:]
        num_joints = template_3d_joints.shape[1]

        # normalize
        template_3d_joints = template_3d_joints - template_pelvis[:, None, :]
        template_vertices_sub2 = template_vertices_sub2 - template_pelvis[:, None, :]

        # concatinate template joints and template vertices, and then duplicate to batch size
        ref_vertices = torch.cat([template_3d_joints, template_vertices_sub2],dim=1)
        ref_vertices = ref_vertices.expand(batch_size, -1, -1)

        # extract grid features and global image features using a CNN backbone
        image_feat, grid_feat = self.backbone(images)
        # concatinate image feat and 3d mesh template
        image_feat = image_feat.view(batch_size, 1, 2048).expand(-1, ref_vertices.shape[-2], -1)
        # process grid features
        grid_feat = torch.flatten(grid_feat, start_dim=2)
        grid_feat = grid_feat.transpose(1,2)
        grid_feat = self.grid_feat_dim(grid_feat)
        # concatinate image feat and template mesh to form the joint/vertex queries
        features = torch.cat([ref_vertices, image_feat], dim=2)
        # prepare input tokens including joint/vertex queries and grid features
        features = torch.cat([features, grid_feat],dim=1)

        if is_train==True:
            # apply mask vertex/joint modeling
            # meta_masks is a tensor of all the masks, randomly generated in dataloader
            # we pre-define a [MASK] token, which is a floating-value vector with 0.01s
            special_token = torch.ones_like(features[:,:-49,:]).cuda()*0.01
            features[:,:-49,:] = features[:,:-49,:]*meta_masks + special_token*(1-meta_masks)          

        # forward pass
        if self.config.output_attentions==True:
            features, hidden_states, att = self.trans_encoder(features)
        else:
            features = self.trans_encoder(features)

        pred_3d_joints = features[:,:num_joints,:]
        pred_vertices_sub2 = features[:,num_joints:-49,:]

        # learn camera parameters
        x = self.cam_param_fc(pred_vertices_sub2)
        x = x.transpose(1,2)
        x = self.cam_param_fc2(x)
        x = self.cam_param_fc3(x)
        cam_param = x.transpose(1,2)
        cam_param = cam_param.squeeze()

        temp_transpose = pred_vertices_sub2.transpose(1,2)
        pred_vertices_sub = self.upsampling(temp_transpose)
        pred_vertices_full = self.upsampling2(pred_vertices_sub)
        pred_vertices_sub = pred_vertices_sub.transpose(1,2)
        pred_vertices_full = pred_vertices_full.transpose(1,2)

        if self.config.output_attentions==True:
            return cam_param, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices_full, hidden_states, att
        else:
            return cam_param, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices_full


class ImmFusion(Graphormer_Body_Network):
    '''
    End-to-end Graphormer network for human pose and mesh reconstruction from a single image.
    '''
    def __init__(self, args, config, backbone, trans_encoder, mesh_sampler):
        super().__init__(args, config, backbone, trans_encoder, mesh_sampler)
        self.upsampling = torch.nn.Linear(655, 2619)
        self.upsampling2 = torch.nn.Linear(2619, 10475)
        self.cam_param_fc = torch.nn.Linear(3, 1)
        self.cam_param_fc2 = torch.nn.Linear(655, 250)
        self.cam_param_fc3 = torch.nn.Linear(250, 3)
        self.grid_feat_dim = torch.nn.Linear(1024, 2051)
        self.image_feat_dim = torch.nn.Linear(2048, 1024)
        
        mlp = [6,128,128,1024,1024,2048] if args.add_rgb else [3,128,128,1024,1024,2048]
        self.pointnet2 = PointnetSAModule(npoint=32, radius=0.4, nsample=32, mlp=mlp, use_xyz=True)
        self.point_feat_num_dim = torch.nn.Linear(32, 1)
        
        if args.points_w_image_feat:
            self.pointnet2 = PointnetSAModule(npoint=32, radius=0.4, nsample=32, mlp=[2051,4096,4096,2048], use_xyz=True)
        
        # self.pointnet2 = PointnetSAModule(npoint=49, radius=0.4, nsample=128, mlp=[3,128,128,1024,1024,2048], use_xyz=True)
        # self.point_feat_num_dim = torch.nn.Linear(49, 1)

        self.point_feat_dim = torch.nn.Linear(2048, 1024)
            
        self.point_embedding = torch.nn.Linear(3, 2051)
        
        self.transformer = Transformer(2048, 3, 8, 256, 4096)
        
        self.mask_ratio = args.mask_ratio

    def forward(self, args, images, smpl, mesh_sampler, meta_masks=None, is_train=False, pcls=None):
        batch_size = images.size(0)
        # Generate T-pose template mesh
        template_pose = torch.zeros((1,69))
        template_pose[:,0] = 3.1416 # Rectify "upside down" reference mesh in global coord
        template_pose = template_pose.cuda(self.config.device)
        template_betas = torch.zeros((1,16)).cuda(self.config.device)
        template_res = smpl(template_pose, template_betas)
        template_vertices = template_res['vertices']

        # template mesh simplification
        template_vertices_sub = mesh_sampler.downsample(template_vertices)
        template_vertices_sub2 = mesh_sampler.downsample(template_vertices_sub, n1=1, n2=2)

        # template mesh-to-joint regression 
        template_3d_joints = template_res['joints'][:, :22]
        template_pelvis = template_3d_joints[:,0,:]
        num_joints = template_3d_joints.shape[1]

        # normalize
        template_3d_joints = template_3d_joints - template_pelvis[:, None, :]
        template_vertices_sub2 = template_vertices_sub2 - template_pelvis[:, None, :]

        # concatinate template joints and template vertices, and then duplicate to batch size
        ref_vertices = torch.cat([template_3d_joints, template_vertices_sub2],dim=1)
        ref_vertices = ref_vertices.expand(batch_size, -1, -1)

        # image and point mask
        if is_train:
            image_mask = np.ones((batch_size, 1, 1, 1))
            point_mask = np.ones((batch_size, 1, 1))
            pb = np.random.random_sample(2)
            masked_num = np.floor(pb * self.mask_ratio * batch_size,) # at most x% could be masked
            image_indices = np.random.choice(np.arange(batch_size),replace=False,size=int(masked_num[0]))
            point_indices = list(np.random.choice(np.arange(batch_size),replace=False,size=int(masked_num[1])))
            for idx in image_indices:
                if idx in point_indices:
                    point_indices.remove(idx)
            image_mask[image_indices] = 0.0
            point_mask[point_indices] = 0.0
            image_mask = torch.from_numpy(image_mask).float().to(images.device)
            point_mask = torch.from_numpy(point_mask).float().to(images.device) 
            image_mask = image_mask.expand(-1, 3, 224, 224)
            point_mask = point_mask.expand(-1, 1024, 6)
            images = image_mask * images
            # pcls = point_mask * pcls
        
        # extract grid features and global image features using a CNN backbone
        image_feat, grid_feat = self.backbone(images)
        # process grid features
        grid_feat = torch.flatten(grid_feat, start_dim=2)
        grid_feat = grid_feat.transpose(1,2)
        grid_feat = self.grid_feat_dim(grid_feat)

        if args.points_w_image_feat:
            xyz, cluster_feat = self.pointnet2(xyz=pcls[:,:,:3].contiguous(), features=torch.cat((pcls[:,:,3:], image_feat[:,None,:].repeat(1,1024,1)), -1).transpose(1,2).contiguous())
        # elif self.use_pointnext:
        #     xyz, cluster_feat = self.pointnext(pcls[:,:,:3].contiguous(), pcls[:,:,3:].transpose(1,2).contiguous())
        #     cluster_feat = self.point_feat_dim(cluster_feat)
        else:
            xyz, cluster_feat = self.pointnet2(xyz=pcls[:,:,:3].contiguous(), features=pcls[:,:,3:].transpose(1,2).contiguous())
        
        xyz_embedding = self.point_embedding(xyz)
        point_feat = self.point_feat_num_dim(cluster_feat).squeeze()
        
        # concatinate image/point feat and 3d mesh template
        if args.wo_GIM:
            fusion_feat = (image_feat + point_feat).view(batch_size,1,2048).expand(-1,ref_vertices.shape[-2],-1)
        elif args.images_only:
            fusion_feat = image_feat.view(batch_size, 1, 2048).expand(-1, ref_vertices.shape[-2], -1)
        elif args.points_only or args.points_w_image_feat:
            fusion_feat = point_feat.view(batch_size,1,2048).expand(-1, ref_vertices.shape[-2], -1)
        elif args.cat_global_feat:
            image_feat = self.image_feat_dim(image_feat)
            point_feat = self.point_feat_dim(point_feat)
            fusion_feat = torch.cat((image_feat,point_feat),dim=-1).view(batch_size,1,2048).expand(-1,ref_vertices.shape[-2],-1)
        else:
            fusion_feat = self.transformer(point_feat.view(batch_size,1,2048),image_feat.view(batch_size,1,2048)).expand(-1,ref_vertices.shape[-2],-1)

        cluster_feat = torch.cat([xyz, cluster_feat.transpose(1,2).contiguous()], dim=2)
        cluster_feat = xyz_embedding + cluster_feat
        
        # concatinate image feat and template mesh to form the joint/vertex queries
        features = torch.cat([ref_vertices, fusion_feat], dim=2)

        # prepare input tokens including joint/vertex queries and grid features
        if args.only_grid_feat:
            features = torch.cat([features, grid_feat], dim=1)
        elif args.only_cluster_feat:
            features = torch.cat([features, cluster_feat], dim=1)
        else:
            features = torch.cat([features, grid_feat, cluster_feat], dim=1)

        if is_train==True:
            # apply mask vertex/joint modeling
            # meta_masks is a tensor of all the masks, randomly generated in dataloader
            # we pre-define a [MASK] token, which is a floating-value vector with 0.01s
            # TODO 49 -> 81
            num_mask = 677
            # num_mask = 726
            # num_mask = 758
            special_token = torch.ones_like(features[:,:num_mask,:]).cuda()*0.01
            features[:,:num_mask,:] = features[:,:num_mask,:]*meta_masks[:,:num_mask,:] + special_token*(1-meta_masks[:,:num_mask,:])

        # forward pass
        if self.config.output_attentions==True:
            features, hidden_states, att = self.trans_encoder(features)
        else:
            features = self.trans_encoder(features)

        pred_3d_joints = features[:,:num_joints,:]
        pred_vertices_sub2 = features[:,num_joints:677,:]

        pred_dict = {}

        temp_transpose = pred_vertices_sub2.transpose(1,2)
        pred_vertices_sub = self.upsampling(temp_transpose)
        pred_vertices_full = self.upsampling2(pred_vertices_sub)
        pred_vertices_sub = pred_vertices_sub.transpose(1,2)
        pred_vertices_full = pred_vertices_full.transpose(1,2)
        
        if self.config.output_attentions==True:
            pred_dict['attention'] = att
            pred_dict['seed_points'] = xyz
        return pred_dict, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices_full


class EndFusionGraphormer(Graphormer_Body_Network):
    '''
    End-to-end Graphormer network for human pose and mesh reconstruction from a single image.
    '''
    def __init__(self, args, config, backbone, trans_encoder, mesh_sampler):
        super().__init__(args, config, backbone, trans_encoder, mesh_sampler)
        self.config = config
        self.backbone = backbone
        self.image_trans_encoder = trans_encoder
        self.point_trans_encoder = copy.deepcopy(trans_encoder)
        
        self.upsampling = torch.nn.Linear(655, 2619)
        self.upsampling2 = torch.nn.Linear(2619, 10475)
        self.grid_feat_dim = torch.nn.Linear(1024, 2051)
        mlp = [6,128,128,1024,1024,2048] if args.add_rgb else [3,128,128,1024,1024,2048]
        self.pointnet2 = PointnetSAModule(npoint=49, radius=0.4, nsample=32, mlp=mlp, use_xyz=True)
        self.point_feat_num_dim = torch.nn.Linear(49, 1)
        self.point_feat_dim = torch.nn.Linear(2048, 3)
        self.point_obj_feat_dim = torch.nn.Linear(3, 2051)
        self.point_embedding = torch.nn.Linear(3, 2051)
        self.img_obj_feat_dim = torch.nn.Linear(2, 2051)
        self.output_dim = torch.nn.Linear(3, 3)
        self.res_output_dim = torch.nn.Linear(2054, 3)
        

    def forward(self, images, smpl, mesh_sampler, meta_masks=None, is_train=False, pcls=None):
        batch_size = images.size(0)
        # Generate T-pose template mesh
        template_pose = torch.zeros((1,69))
        template_pose[:,0] = 3.1416 # Rectify "upside down" reference mesh in global coord
        template_pose = template_pose.cuda(self.config.device)
        template_betas = torch.zeros((1,16)).cuda(self.config.device)
        template_res = smpl(template_pose, template_betas)
        template_vertices = template_res['vertices']

        # template mesh simplification
        template_vertices_sub = mesh_sampler.downsample(template_vertices)
        template_vertices_sub2 = mesh_sampler.downsample(template_vertices_sub, n1=1, n2=2)

        # template mesh-to-joint regression 
        template_3d_joints = template_res['joints'][:, :22]
        template_pelvis = template_3d_joints[:,0,:]
        # template_3d_joints = template_3d_joints[:,cfg.H36M_J17_TO_J14,:]
        num_joints = template_3d_joints.shape[1]

        # normalize
        template_3d_joints = template_3d_joints - template_pelvis[:, None, :]
        template_vertices_sub2 = template_vertices_sub2 - template_pelvis[:, None, :]

        # concatinate template joints and template vertices, and then duplicate to batch size
        ref_vertices = torch.cat([template_3d_joints, template_vertices_sub2],dim=1)
        ref_vertices = ref_vertices.expand(batch_size, -1, -1)

        # extract grid features and global image features using a CNN backbone
        image_feat, grid_feat = self.backbone(images)
        # process grid features
        grid_feat = torch.flatten(grid_feat, start_dim=2)
        grid_feat = grid_feat.transpose(1,2)
        grid_feat = self.grid_feat_dim(grid_feat)

        # TODO add point features
        xyz, cluster_feat = self.pointnet2(xyz=pcls[:,:,:3].contiguous(), features=pcls[:,:,3:].transpose(1,2).contiguous())
        xyz_embedding = self.point_embedding(xyz)
        global_point_feat = self.point_feat_num_dim(cluster_feat)
        cluster_feat = torch.cat([xyz, cluster_feat.transpose(1,2)], dim=2)
        cluster_feat = xyz_embedding + cluster_feat
        
        # concatinate image/point feat and 3d mesh template
        # fusion_feat = (image_feat.view(batch_size, 1, 2048) + global_point_feat.transpose(1,2)).expand(-1, ref_vertices.shape[-2], -1)
        image_feat = image_feat.view(batch_size, 1, 2048).expand(-1, ref_vertices.shape[-2], -1)
        point_feat = global_point_feat.transpose(1,2).expand(-1, ref_vertices.shape[-2], -1)

        # concatinate image feat and template mesh to form the joint/vertex queries
        image_features = torch.cat([ref_vertices, image_feat], dim=2)
        point_features = torch.cat([ref_vertices, point_feat], dim=2)

        # prepare input tokens including joint/vertex queries and grid features
        image_features = torch.cat([image_features, grid_feat], dim=1)
        point_features = torch.cat([point_features, cluster_feat], dim=1)

        if is_train==True:
            # apply mask vertex/joint modeling
            # meta_masks is a tensor of all the masks, randomly generated in dataloader
            # we pre-define a [MASK] token, which is a floating-value vector with 0.01s
            # TODO 49 -> 81
            special_token = torch.ones_like(image_features[:,:677,:]).cuda()*0.01
            image_features[:,:677,:] = image_features[:,:677,:]*meta_masks + special_token*(1-meta_masks)
            point_features[:,:677,:] = point_features[:,:677,:]*meta_masks + special_token*(1-meta_masks)

        # forward pass
        _image_features = self.image_trans_encoder(image_features)
        _point_features = self.point_trans_encoder(point_features)
        
        image_feat_score = _image_features[:,:,2:3]
        point_feat_score = _point_features[:,:,2:3]
        
        # fusion_features = image_feat_score * image_features[:,:,:3] + point_feat_score * point_features[:,:,:3]
        # mask = torch.abs(image_feat_score) >= torch.abs(point_feat_score) # h
        mask = torch.abs(image_feat_score) <= torch.abs(point_feat_score) # l
        # fusion_features = mask * _image_features[:,:,:3] + ~mask * _point_features[:,:,:3]
        # features = self.output_dim(fusion_features)
        fusion_features = mask * torch.cat((_image_features[:,:,:3], image_features), -1) + ~mask * torch.cat((_point_features[:,:,:3], point_features), -1)
        features = self.res_output_dim(fusion_features)

        pred_3d_joints = features[:,:num_joints,:]
        pred_vertices_sub2 = features[:,num_joints:677,:]

        pred_score = torch.cat((image_feat_score, point_feat_score), dim=1).squeeze()
        pred_dict = {'pred_score': pred_score}
        pred_dict.update({'pred_image_joints': _image_features[:,:22,:3]})
        pred_dict.update({'pred_image_verts': _image_features[:,22:677,:3]})
        pred_dict.update({'pred_point_joints': _point_features[:,:22,:3]})
        pred_dict.update({'pred_point_verts': _point_features[:,22:677,:3]})

        temp_transpose = pred_vertices_sub2.transpose(1,2)
        pred_vertices_sub = self.upsampling(temp_transpose)
        pred_vertices_full = self.upsampling2(pred_vertices_sub)
        pred_vertices_sub = pred_vertices_sub.transpose(1,2)
        pred_vertices_full = pred_vertices_full.transpose(1,2)

        return pred_dict, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices_full


class MidFusionGraphormer(Graphormer_Body_Network):
    '''
    End-to-end Graphormer network for human pose and mesh reconstruction from a single image.
    '''
    def __init__(self, args, config, backbone, trans_encoder, mesh_sampler):
        super().__init__(args, config, backbone, trans_encoder, mesh_sampler)
        self.config = config
        self.backbone = backbone
        self.image_trans_encoder = trans_encoder
        self.point_trans_encoder = copy.deepcopy(trans_encoder)
        self.image_score_net = torch.nn.Sequential(torch.nn.Linear(512, 1), torch.nn.Linear(128, 1))
        self.point_score_net = torch.nn.Sequential(torch.nn.Linear(512, 1), torch.nn.Linear(128, 1))
        
        self.upsampling = torch.nn.Linear(655, 2619)
        self.upsampling2 = torch.nn.Linear(2619, 10475)
        self.grid_feat_dim = torch.nn.Linear(1024, 2051)
        self.pointnet2 = PointnetSAModule(npoint=49, radius=0.4, nsample=32, mlp=[3,128,128,1024,1024,2048], use_xyz=True)
        self.point_feat_num_dim = torch.nn.Linear(49, 1)
        self.point_feat_dim = torch.nn.Linear(2048, 3)
        self.point_obj_feat_dim = torch.nn.Linear(3, 2051)
        self.point_embedding = torch.nn.Linear(3, 2051)
        self.img_obj_feat_dim = torch.nn.Linear(2, 2051)
        self.output_dim = torch.nn.Linear(3, 3)
        

    def forward(self, args, images, smpl, mesh_sampler, meta_masks=None, is_train=False, pcls=None):
        batch_size = images.size(0)
        # Generate T-pose template mesh
        template_pose = torch.zeros((1,69))
        template_pose[:,0] = 3.1416 # Rectify "upside down" reference mesh in global coord
        template_pose = template_pose.cuda(self.config.device)
        template_betas = torch.zeros((1,16)).cuda(self.config.device)
        template_res = smpl(template_pose, template_betas)
        template_vertices = template_res['vertices']

        # template mesh simplification
        template_vertices_sub = mesh_sampler.downsample(template_vertices)
        template_vertices_sub2 = mesh_sampler.downsample(template_vertices_sub, n1=1, n2=2)

        # template mesh-to-joint regression 
        template_3d_joints = template_res['joints'][:, :22]
        template_pelvis = template_3d_joints[:,0,:]
        # template_3d_joints = template_3d_joints[:,cfg.H36M_J17_TO_J14,:]
        num_joints = template_3d_joints.shape[1]

        # normalize
        template_3d_joints = template_3d_joints - template_pelvis[:, None, :]
        template_vertices_sub2 = template_vertices_sub2 - template_pelvis[:, None, :]

        # concatinate template joints and template vertices, and then duplicate to batch size
        ref_vertices = torch.cat([template_3d_joints, template_vertices_sub2],dim=1)
        ref_vertices = ref_vertices.expand(batch_size, -1, -1)
        
        # image and point mask
        if is_train:
            image_mask = np.ones((batch_size, 1, 1, 1))
            point_mask = np.ones((batch_size, 1, 1))
            pb = np.random.random_sample(2)
            masked_num = np.floor(pb*0.3*batch_size, ) # at most x% of the vertices could be masked
            image_indices = np.random.choice(np.arange(batch_size),replace=False,size=int(masked_num[0]))
            point_indices = list(np.random.choice(np.arange(batch_size),replace=False,size=int(masked_num[1])))
            for idx in image_indices:
                if idx in point_indices:
                    point_indices.remove(idx)
            image_mask[image_indices] = 0.0
            point_mask[point_indices] = 0.0
            image_mask = torch.from_numpy(image_mask).float().to(images.device)
            point_mask = torch.from_numpy(point_mask).float().to(images.device) 
            image_mask = image_mask.expand(-1, 3, 224, 224)
            point_mask = point_mask.expand(-1, 1024, 6)
            # images = image_mask * images
            # pcls = point_mask * pcls
            
        # extract grid features and global image features using a CNN backbone
        image_feat, grid_feat = self.backbone(images)
        # process grid features
        grid_feat = torch.flatten(grid_feat, start_dim=2)
        grid_feat = grid_feat.transpose(1,2)
        grid_feat = self.grid_feat_dim(grid_feat)

        # TODO add point features
        xyz, cluster_feat = self.pointnet2(xyz=pcls[:,:,:3].contiguous(), features=pcls[:,:,3:].transpose(1,2).contiguous())
        xyz_embedding = self.point_embedding(xyz)
        global_point_feat = self.point_feat_num_dim(cluster_feat)
        cluster_feat = torch.cat([xyz, cluster_feat.transpose(1,2)], dim=2)
        cluster_feat = xyz_embedding + cluster_feat
        
        # concatinate image/point feat and 3d mesh template
        # fusion_feat = (image_feat.view(batch_size, 1, 2048) + global_point_feat.transpose(1,2)).expand(-1, ref_vertices.shape[-2], -1)
        image_feat = image_feat.view(batch_size, 1, 2048).expand(-1, ref_vertices.shape[-2], -1)
        point_feat = global_point_feat.transpose(1,2).expand(-1, ref_vertices.shape[-2], -1)

        # concatinate image feat and template mesh to form the joint/vertex queries
        image_features = torch.cat([ref_vertices, image_feat], dim=2)
        point_features = torch.cat([ref_vertices, point_feat], dim=2)

        # prepare input tokens including joint/vertex queries and grid features
        image_features = torch.cat([image_features, grid_feat], dim=1)
        point_features = torch.cat([point_features, cluster_feat], dim=1)

        if is_train==True:
            # apply mask vertex/joint modeling
            # meta_masks is a tensor of all the masks, randomly generated in dataloader
            # we pre-define a [MASK] token, which is a floating-value vector with 0.01s
            # TODO 49 -> 81
            num_mask = 677
            # num_mask = 726
            # num_mask = 758
            special_token = torch.ones_like(image_features[:,:num_mask,:]).cuda()*0.01
            image_features[:,:num_mask,:] = image_features[:,:num_mask,:]*meta_masks[:,:num_mask,:] + special_token*(1-meta_masks[:,:num_mask,:])
            point_features[:,:num_mask,:] = point_features[:,:num_mask,:]*meta_masks[:,:num_mask,:] + special_token*(1-meta_masks[:,:num_mask,:])
            
        pred_score = []
        # forward pass
        for i in range(2):
            image_features = self.image_trans_encoder[i](image_features)
            point_features = self.point_trans_encoder[i](point_features)
            # score the tokens
            image_feat_score = self.image_score_net[i](image_features[:,:677,:])
            point_feat_score = self.point_score_net[i](point_features[:,:677,:])
            pred_score.append(image_feat_score)
            pred_score.append(point_feat_score)
            # replace unimportant tokens
            # image_mask = torch.abs(image_feat_score) <= torch.full_like(image_feat_score, 0.01)
            # point_mask = torch.abs(point_feat_score) <= torch.full_like(point_feat_score, 0.01)
            # _image_features = image_mask * image_features[:,:677,:] + ~image_mask * point_features[:,:677,:]
            # _point_features = point_mask * point_features[:,:677,:] + ~point_mask * image_features[:,:677,:]
            # image_features[:,:677,:] = _image_features
            # point_features[:,:677,:] = _point_features
            mask = torch.abs(image_feat_score) >= torch.abs(point_feat_score)
            _fusion_features = mask * image_features[:,:677,:] + ~mask * point_features[:,:677,:]
            image_features[:,:677,:] = _fusion_features
            point_features[:,:677,:] = _fusion_features
        
        image_features = self.image_trans_encoder[2](image_features)
        point_features = self.point_trans_encoder[2](point_features)
        fusion_features = image_features + point_features
        features = self.output_dim(fusion_features)
        # features = image_features
        pred_3d_joints = features[:,:num_joints,:]
        pred_vertices_sub2 = features[:,num_joints:677,:]

        pred_score = torch.cat(pred_score, dim=1).squeeze()
        pred_dict = {'pred_score': pred_score}
        pred_dict.update({'pred_image_joints': image_features[:,:22,:3]})
        pred_dict.update({'pred_image_verts': image_features[:,22:677,:3]})
        pred_dict.update({'pred_point_joints': point_features[:,:22,:3]})
        pred_dict.update({'pred_point_verts': point_features[:,22:677,:3]})

        temp_transpose = pred_vertices_sub2.transpose(1,2)
        pred_vertices_sub = self.upsampling(temp_transpose)
        pred_vertices_full = self.upsampling2(pred_vertices_sub)
        pred_vertices_sub = pred_vertices_sub.transpose(1,2)
        pred_vertices_full = pred_vertices_full.transpose(1,2)

        return pred_dict, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices_full