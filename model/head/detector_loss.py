from cv2 import phase
import torch
import math
import torch.distributed as dist
import pdb

from torch.nn import functional as F
from utils.comm import get_world_size

from model.anno_encoder import Anno_Encoder
from model.layers.utils import select_point_of_interest
from model.utils import Uncertainty_Reg_Loss, Laplace_Loss, RegL1Loss, Position_loss

from model.layers.focal_loss import *
from model.layers.iou_loss import *
from model.head.depth_losses import *
from model.layers.utils import Converter_key2channel

def make_loss_evaluator(cfg):
	loss_evaluator = Loss_Computation(cfg=cfg)
	return loss_evaluator

class Loss_Computation():
	def __init__(self, cfg):
		
		self.anno_encoder = Anno_Encoder(cfg)
		self.key2channel = Converter_key2channel(keys=cfg.MODEL.HEAD.REGRESSION_HEADS, channels=cfg.MODEL.HEAD.REGRESSION_CHANNELS)
		
		self.max_objs = cfg.DATASETS.MAX_OBJECTS
		self.center_sample = cfg.MODEL.HEAD.CENTER_SAMPLE
		self.regress_area = cfg.MODEL.HEAD.REGRESSION_AREA
		self.heatmap_type = cfg.MODEL.HEAD.HEATMAP_TYPE
		self.key_points_heat_map = cfg.MODEL.HEAD.HEATMAP_TYPE
		self.corner_depth_sp = cfg.MODEL.HEAD.SUPERVISE_CORNER_DEPTH
		self.loss_keys = cfg.MODEL.HEAD.LOSS_NAMES

		self.world_size = get_world_size()
		self.dim_weight = torch.as_tensor(cfg.MODEL.HEAD.DIMENSION_WEIGHT).view(1, 3)
		self.uncertainty_range = cfg.MODEL.HEAD.UNCERTAINTY_RANGE

		# loss functions
		loss_types = cfg.MODEL.HEAD.LOSS_TYPE
		self.cls_loss_fnc = FocalLoss(cfg.MODEL.HEAD.LOSS_PENALTY_ALPHA, cfg.MODEL.HEAD.LOSS_BETA) # penalty-reduced focal loss
		self.iou_loss = IOULoss(loss_type=loss_types[2]) # iou loss for 2D detection

		# depth loss
		if loss_types[3] == 'berhu': self.depth_loss = Berhu_Loss()
		elif loss_types[3] == 'inv_sig': self.depth_loss = Inverse_Sigmoid_Loss()
		elif loss_types[3] == 'log': self.depth_loss = Log_L1_Loss()
		elif loss_types[3] == 'L1': self.depth_loss = F.l1_loss
		else: raise ValueError

		# regular regression loss
		self.reg_loss = loss_types[1]
		self.reg_loss_fnc = F.l1_loss if loss_types[1] == 'L1' else F.smooth_l1_loss
		self.keypoint_loss_fnc = F.l1_loss
		self.keypoint_loc_loss_fnc = F.l1_loss
		# 
		self.keypoint_reg_loss = RegL1Loss()
		self.keypoint_position_loss = Position_loss()

		# multi-bin loss setting for orientation estimation
		self.multibin = (cfg.INPUT.ORIENTATION == 'multi-bin')
		self.orien_bin_size = cfg.INPUT.ORIENTATION_BIN_SIZE
		self.trunc_offset_loss_type = cfg.MODEL.HEAD.TRUNCATION_OFFSET_LOSS

		self.loss_weights = {}
		for key, weight in zip(cfg.MODEL.HEAD.LOSS_NAMES, cfg.MODEL.HEAD.INIT_LOSS_WEIGHT): self.loss_weights[key] = weight

		# whether to compute corner loss
		self.compute_direct_depth_loss = 'depth_loss' in self.loss_keys
		self.compute_svd_inverse_depth_loss = 'svd_inverse_loss' in self.loss_keys
		self.compute_keypoint_depth_loss = 'keypoint_depth_loss' in self.loss_keys
		self.compute_weighted_depth_loss = 'weighted_avg_depth_loss' in self.loss_keys
		self.compute_corner_loss = 'corner_loss' in self.loss_keys
		self.separate_trunc_offset = 'trunc_offset_loss' in self.loss_keys
		
		self.pred_direct_depth = 'depth' in self.key2channel.keys
		self.depth_with_uncertainty  = 'depth_uncertainty' in self.key2channel.keys
		self.compute_keypoint_corner = 'corner_offset' in self.key2channel.keys
		self.corner_with_uncertainty = 'corner_uncertainty' in self.key2channel.keys
		#====================================================================================#
		self.svd_inverse_with_uncertainty = 'svd_inverse_uncertainty' in self.key2channel.keys
		self.six_planes_location_uncertainty = 'six_planes_uncertainty' in self.key2channel.keys
		#====================================================================================#

		self.uncertainty_weight = cfg.MODEL.HEAD.UNCERTAINTY_WEIGHT # 1.0
		self.keypoint_xy_weights = cfg.MODEL.HEAD.KEYPOINT_XY_WEIGHT # [1, 1]
		self.keypoint_norm_factor = cfg.MODEL.HEAD.KEYPOINT_NORM_FACTOR # 1.0
		self.modify_invalid_keypoint_depths = cfg.MODEL.HEAD.MODIFY_INVALID_KEYPOINT_DEPTH
		
		self.down_ratio = cfg.MODEL.BACKBONE.DOWN_RATIO
		self.input_width = cfg.INPUT.WIDTH_TRAIN 
		self.output_width = self.input_width // cfg.MODEL.BACKBONE.DOWN_RATIO
		# depth used to compute 8 corners
		self.corner_loss_depth = cfg.MODEL.HEAD.CORNER_LOSS_DEPTH
		self.eps = 1e-5
		
		# 3D bounding box face index
		self.keypoints_num = cfg.DATASETS.KEYPOINTS_NUM
		self.compute_keypoints_local_offsets_loss = cfg.DATASETS.KEYPOINT_LOCAL_OFFSETS_LOSS
		self.use_area = cfg.DATASETS.USE_AREA
		self.keypoints_depth_mask_num = cfg.DATASETS.DEPTH_MASK_NUM

		self.compute_poisition_loss = cfg.DATASETS.POISITION_LOSS
		
		self.back_face_index  = torch.tensor([2,3,7,6])
		self.front_face_index = torch.tensor([0,1,5,4])

		self.bottom_face_index = torch.tensor([0,1,2,3])
		self.top_face_index    = torch.tensor([4,5,6,7])

		self.right_face_index = torch.tensor([3,0,4,7])
		self.left_face_index  = torch.tensor([1,2,6,5])
		
		if self.keypoints_num == 10:
			self.box_cpt_coef      = torch.Tensor([[[[ 1/2, 1/2, 1/2],
													 [ 1/2, 1/2,-1/2],
													 [-1/2, 1/2,-1/2],
													 [-1/2, 1/2, 1/2],

													 [ 1/2,-1/2, 1/2],
													 [ 1/2,-1/2,-1/2],
													 [-1/2,-1/2,-1/2],
													 [-1/2,-1/2, 1/2],

													 [   0, 1/2,   0],
													 [   0,-1/2,   0]]]])
		elif self.keypoints_num == 27:
			self.box_cpt_coef      = torch.Tensor([[[[ 1/2, 1/2, 1/2],
													 [ 1/2, 1/2,   0],
													 [ 1/2, 1/2,-1/2],
													 [   0, 1/2,-1/2],
													 [-1/2, 1/2,-1/2],
													 [-1/2, 1/2,   0],
													 [-1/2, 1/2, 1/2],
													 [   0, 1/2, 1/2],
													 [   0, 1/2,   0],
													
													 [ 1/2, 0, 1/2],
													 [ 1/2, 0,   0],
													 [ 1/2, 0,-1/2],
													 [   0, 0,-1/2],
													 [-1/2, 0,-1/2],
													 [-1/2, 0,   0],
													 [-1/2, 0, 1/2],
													 [   0, 0, 1/2],
													 [   0, 0,   0],

													 [ 1/2, -1/2, 1/2],
													 [ 1/2, -1/2,   0],
													 [ 1/2, -1/2,-1/2],
													 [   0, -1/2,-1/2],
													 [-1/2, -1/2,-1/2],
													 [-1/2, -1/2,   0],
													 [-1/2, -1/2, 1/2],
													 [   0, -1/2, 1/2],
													 [   0, -1/2,   0]]]])

		elif self.keypoints_num == 125: raise NotImplementedError
 
	def prepare_targets(self, targets):
		# clses
		heatmaps = torch.stack([t.get_field("hm") for t in targets])
		#===============================================================================#
		# New added
		key_points_heat_map = torch.stack([t.get_field("keypoints_hm") for t in targets]) # keypoints regression 
		center_point_indexs = torch.stack([t.get_field("cpts_index") for t in targets]) # center points index
		key_points_indexs = torch.stack([t.get_field("keypoints_index") for t in targets]) # keypoints index
		#===============================================================================#

		cls_ids = torch.stack([t.get_field("cls_ids") for t in targets])
		offset_3D = torch.stack([t.get_field("offset_3D") for t in targets])

		#===============================================================================#
		# New added
		keypoints_local_offset = torch.stack([t.get_field("keypoints_local_offset") for t in targets]) # keypoints local offset
		#===============================================================================#
		
		# 2d detection
		target_centers = torch.stack([t.get_field("target_centers") for t in targets])
		bboxes = torch.stack([t.get_field("2d_bboxes") for t in targets])
		# 3d detection
		keypoints = torch.stack([t.get_field("keypoints") for t in targets])
		keypoints_depth_mask = torch.stack([t.get_field("keypoints_depth_mask") for t in targets])
		dimensions = torch.stack([t.get_field("dimensions") for t in targets])
		locations = torch.stack([t.get_field("locations") for t in targets])
		plane_locations = torch.stack([t.get_field("plane_locations") for t in targets])
		rotys = torch.stack([t.get_field("rotys") for t in targets])
		alphas = torch.stack([t.get_field("alphas") for t in targets])
		orientations = torch.stack([t.get_field("orientations") for t in targets])
		# utils
		pad_size = torch.stack([t.get_field("pad_size") for t in targets])
		calibs = [t.get_field("calib") for t in targets]
		reg_mask = torch.stack([t.get_field("reg_mask") for t in targets])
		reg_weight = torch.stack([t.get_field("reg_weight") for t in targets])
		ori_imgs = torch.stack([t.get_field("ori_img") for t in targets])
		trunc_mask = torch.stack([t.get_field("trunc_mask") for t in targets])

		# ==================================================================== #
		occlusions = torch.stack([t.get_field("occlusions") for t in targets]) 
		# ==================================================================== #

		return_dict = dict(cls_ids=cls_ids, target_centers=target_centers, bboxes=bboxes, keypoints=keypoints, dimensions=dimensions, occlusions=occlusions,
			locations=locations, plane_locations=plane_locations,rotys=rotys, alphas=alphas, calibs=calibs, pad_size=pad_size, reg_mask=reg_mask, reg_weight=reg_weight,
			offset_3D=offset_3D, ori_imgs=ori_imgs, trunc_mask=trunc_mask, orientations=orientations, keypoints_depth_mask=keypoints_depth_mask,keypoints_local_offset=keypoints_local_offset,key_points_indexs=key_points_indexs,center_point_indexs=center_point_indexs
		)

		return heatmaps, return_dict, key_points_heat_map

	def prepare_predictions(self, targets_variables, predictions):
		# regression head 
		pred_regression = predictions['reg'] # torch.Size([2, 50, 96, 320])  torch.Size([2, 58, 96, 320])
		batch, channel, feat_h, feat_w = pred_regression.shape
		# 1. get the representative points
		targets_bbox_points = targets_variables["target_centers"] # representative points [2, 40, 2]   40 means max objects [[114,  60],[135,  51],[160,  48],[149,  47],]
		
		reg_mask_gt = targets_variables["reg_mask"] # (2,40) 40 objects (1,0)
		flatten_reg_mask_gt = reg_mask_gt.view(-1).bool() #  (80) (True, False)
		
		# the corresponding image_index for each object, used for finding pad_size, calib and so on
		batch_idxs = torch.arange(batch).view(-1, 1).expand_as(reg_mask_gt).reshape(-1).to(reg_mask_gt.device) 
		batch_idxs = batch_idxs[flatten_reg_mask_gt].to(reg_mask_gt.device) # 12 , 10 , 8 ...

		valid_targets_bbox_points = targets_bbox_points.view(-1, 2)[flatten_reg_mask_gt]

		# fcos-style targets for 2D
		target_bboxes_2D = targets_variables['bboxes'].view(-1, 4)[flatten_reg_mask_gt]
		target_bboxes_height = target_bboxes_2D[:, 3] - target_bboxes_2D[:, 1]
		target_bboxes_width  = target_bboxes_2D[:, 2] - target_bboxes_2D[:, 0]

		target_regression_2D = torch.cat((valid_targets_bbox_points - target_bboxes_2D[:, :2], target_bboxes_2D[:, 2:] - valid_targets_bbox_points), dim=1)
		mask_regression_2D = (target_bboxes_height > 0) & (target_bboxes_width > 0)
		target_regression_2D = target_regression_2D[mask_regression_2D]

		# targets for 3D
		target_clses = targets_variables["cls_ids"].view(-1)[flatten_reg_mask_gt]
		target_depths_3D = targets_variables['locations'][..., -1].view(-1)[flatten_reg_mask_gt] # ([ 6.2700, 17.1200, 46.1000,  7.1400]
		#==============================================================================================================================#
		# New added
		target_occlusions = targets_variables["occlusions"].view(-1)[flatten_reg_mask_gt] # (valid_objs,)
		target_3D_center = targets_variables['locations'].view(-1,3)[flatten_reg_mask_gt] # (valid_objs,3)
		target_plane_center = targets_variables['plane_locations'].view(-1,6,3)[flatten_reg_mask_gt] # (valid_objs,6,3)
		#==============================================================================================================================#
		target_rotys_3D = targets_variables['rotys'].view(-1)[flatten_reg_mask_gt]
		target_alphas_3D = targets_variables['alphas'].view(-1)[flatten_reg_mask_gt]
		target_offset_3D = targets_variables["offset_3D"].view(-1, 2)[flatten_reg_mask_gt]

		#==========================================================================================================================-===#
		# New added
		target_keypoints_local_offset = targets_variables["keypoints_local_offset"].view(-1, self.keypoints_num, 2)[flatten_reg_mask_gt] # torch.Size([2, 40, 12, 2]) ->[*,12,2]
		target_keypoints_index = targets_variables['key_points_indexs'].view(-1, self.keypoints_num)[flatten_reg_mask_gt]
		target_center_points_index = targets_variables['center_point_indexs'].view(-1)[flatten_reg_mask_gt]

		target_calib = torch.tensor([targets_variables['calibs'][i].P for i in range(batch)]) # class 
		target_calib = target_calib.unsqueeze(1).repeat(1,reg_mask_gt.shape[1],1,1).to(reg_mask_gt.device)  # torch.Size([2, 3, 4]) -> torch.Size([2, 1, 3, 4]) -> torch.Size([2, 40, 3, 4])
		target_calib = target_calib.view(-1,3,4)[flatten_reg_mask_gt] # torch.Size([2, 40, 3, 4]) -> torch.Size([valid_objs, 3, 4])
		
		target_image_pad_size = targets_variables['pad_size'].unsqueeze(1).unsqueeze(1).repeat(1,reg_mask_gt.shape[1],self.keypoints_num,1) # torch.Size([2, 2]) -> torch.Size([2, 1, 1, 2]) -> torch.Size([2, 40, 10, 2])
		target_image_pad_size = target_image_pad_size.view(-1,self.keypoints_num,2)[flatten_reg_mask_gt] # torch.Size([2, 40, 10, 2]) -> torch.Size([valid_objs, 10, 2])
		#==================================================================================#
		target_dimensions_3D = targets_variables['dimensions'].view(-1, 3)[flatten_reg_mask_gt]
		
		target_orientation_3D = targets_variables['orientations'].view(-1, targets_variables['orientations'].shape[-1])[flatten_reg_mask_gt]
		target_locations_3D = self.anno_encoder.decode_location_flatten(valid_targets_bbox_points, target_offset_3D, target_depths_3D, 
										targets_variables['calibs'], targets_variables['pad_size'], batch_idxs)

		target_corners_3D = self.anno_encoder.encode_box3d(target_rotys_3D, target_dimensions_3D, target_locations_3D) # 3D corner编码
		target_bboxes_3D = torch.cat((target_locations_3D, target_dimensions_3D, target_rotys_3D[:, None]), dim=1)

		target_trunc_mask = targets_variables['trunc_mask'].view(-1)[flatten_reg_mask_gt]
		obj_weights = targets_variables["reg_weight"].view(-1)[flatten_reg_mask_gt]

		# 2. extract corresponding predictions
		# pred_regression_pois_3D size = (valid_obj, 80)
		pred_regression_pois_3D = select_point_of_interest(batch, targets_bbox_points, pred_regression).view(-1, channel)[flatten_reg_mask_gt]
		
		pred_regression_2D = F.relu(pred_regression_pois_3D[mask_regression_2D, self.key2channel('2d_dim')])
		pred_offset_3D = pred_regression_pois_3D[:, self.key2channel('3d_offset')]
		
		#=========================================================================================================#
		# New added
		pred_keypoint_local_offset = pred_regression_pois_3D[:, self.key2channel('keypoint_local_offset')] #(*, 20)
		#=========================================================================================================#
		pred_dimensions_offsets_3D = pred_regression_pois_3D[:, self.key2channel('3d_dim')]
		pred_orientation_3D = torch.cat((pred_regression_pois_3D[:, self.key2channel('ori_cls')], 
										 pred_regression_pois_3D[:, self.key2channel('ori_offset')]), dim=1)
		
		# decode the pred residual dimensions to real dimensions
		pred_dimensions_3D = self.anno_encoder.decode_dimension(target_clses, pred_dimensions_offsets_3D)

		# preparing outputs
		targets = { 'reg_2D': target_regression_2D, 'offset_3D': target_offset_3D, 'depth_3D': target_depths_3D, 'orien_3D': target_orientation_3D, 'occlusions':target_occlusions,
					'dims_3D': target_dimensions_3D, 'corners_3D': target_corners_3D, 'keypoints_local_offset':target_keypoints_local_offset,'width_2D': target_bboxes_width, 'rotys_3D': target_rotys_3D,
					'cat_3D': target_bboxes_3D, 'trunc_mask_3D': target_trunc_mask, 'height_2D': target_bboxes_height, 'valid_targets_center_points':valid_targets_bbox_points, 
					'keypoints_index':target_keypoints_index,'center_points_index':target_center_points_index,'calib':target_calib,'image_pad_size':target_image_pad_size,'3D_center':target_3D_center,'3D_plane_center':target_plane_center
				  }

		preds = {'reg_2D': pred_regression_2D, 'offset_3D': pred_offset_3D,'orien_3D': pred_orientation_3D, 'dims_3D': pred_dimensions_3D, 'keypoints_local_offset': pred_keypoint_local_offset}
		
		reg_nums = {'reg_2D': mask_regression_2D.sum(), 'reg_3D': flatten_reg_mask_gt.sum(), 'reg_obj': flatten_reg_mask_gt.sum()}
		weights  = {'object_weights': obj_weights}

		# predict the depth with direct regression
		if self.pred_direct_depth:
    		#  回归深度信息  size = 1
			pred_depths_offset_3D = pred_regression_pois_3D[:, self.key2channel('depth')].squeeze(-1)
			pred_direct_depths_3D = self.anno_encoder.decode_depth(pred_depths_offset_3D) # direct depth
			preds['depth_3D'] = pred_direct_depths_3D #(objs,)

		# predict the uncertainty of depth regression
		if self.depth_with_uncertainty:
    		# 回归深度值的不确定度 size = 1
			preds['depth_uncertainty'] = pred_regression_pois_3D[:, self.key2channel('depth_uncertainty')].squeeze(-1)
			
			if self.uncertainty_range is not None:
				preds['depth_uncertainty'] = torch.clamp(preds['depth_uncertainty'], min=self.uncertainty_range[0], max=self.uncertainty_range[1])

			# else:
			# 	print('depth_uncertainty: {:.2f} +/- {:.2f}'.format(
			# 		preds['depth_uncertainty'].mean().item(), preds['depth_uncertainty'].std().item()))

		# predict the keypoints
		if self.compute_keypoint_corner:
			# targets for keypoints
			# [x, y, mask]
			target_corner_keypoints = targets_variables["keypoints"].view(flatten_reg_mask_gt.shape[0], -1, 3)[flatten_reg_mask_gt] # 2D project points and mask
			targets['keypoints'] = target_corner_keypoints[..., :2] # 前两列投影点
			targets['keypoints_mask'] = target_corner_keypoints[..., -1] # 后一列mask
			targets['coef'] = self.box_cpt_coef.to(flatten_reg_mask_gt.device).repeat(targets_variables["keypoints"].shape[0], targets_variables["keypoints"].shape[1],1,1).view(flatten_reg_mask_gt.shape[0],-1,3)[flatten_reg_mask_gt]

			reg_nums['keypoints'] = targets['keypoints_mask'].sum()

			# mask for whether depth should be computed from certain group of keypoints
			# 根据回归深度方式的个数设置 mask数量
			target_corner_depth_mask = targets_variables["keypoints_depth_mask"].view(-1, self.keypoints_depth_mask_num)[flatten_reg_mask_gt]
			targets['keypoints_depth_mask'] = target_corner_depth_mask

			# predictions for keypoints
			pred_keypoints_3D = pred_regression_pois_3D[:, self.key2channel('corner_offset')] # (*, 12) # offset
			pred_keypoints_3D = pred_keypoints_3D.view(flatten_reg_mask_gt.sum(), -1, 2)
			pred_keypoints_depths_3D = self.anno_encoder.decode_depth_from_keypoints_batch(pred_keypoints_3D, pred_dimensions_3D,
														targets_variables['calibs'], batch_idxs)

			preds['keypoints'] = pred_keypoints_3D # [objs, 12, 2] objs is variation
			preds['keypoints_depths'] = pred_keypoints_depths_3D # [objs, N个方式]

			#=========================================================================================================#
			# svd inverse depth
			cys = (target_center_points_index / self.output_width).int().float() # (valid_objs)
			cxs = (target_center_points_index % self.output_width).int().float() # (valid_objs)

			center_point = torch.stack([cxs,cys], dim=1) # (valid_objs,2)
			center_point = center_point.unsqueeze(1).repeat(1,self.keypoints_num,1) # (valid_objs, 10, 2)

			# center_point = valid_targets_bbox_points.unsqueeze(1).repeat(1,self.keypoints_num,1)
			pred_2D_keypoint = pred_keypoints_3D + center_point
			
			
			svd_3d_location, pred_depth_by_points_svd_inv, pred_rotys_3D, _, pred_plane_center_points = self.anno_encoder.decode_location_by_keypoints_svd_inverse(pred_orientation_3D, pred_2D_keypoint, pred_dimensions_3D, target_calib, target_image_pad_size, occlusions = target_occlusions, phase_flag='Training') # (objs)
			preds['pred_2D_keypoint'] = pred_2D_keypoint
			preds['svd_3d_location'] = svd_3d_location
			preds['depth_by_svd_inverse'] = pred_depth_by_points_svd_inv  #(objs,)
			preds['plane_center_points'] = pred_plane_center_points
			# preds['keypoints_depths'] = torch.cat((pred_keypoints_depths_3D, pred_depth_by_points_svd_inv.unsqueeze(1)),dim=1)
			#========================================================================================================# 

		# predict the uncertainties of the solved depths from groups of keypoints
		if self.corner_with_uncertainty:
    		# 回归关键点的不确定度 size = 5（根据深度方式确定个数）
			preds['corner_offset_uncertainty'] = pred_regression_pois_3D[:, self.key2channel('corner_uncertainty')]

			if self.uncertainty_range is not None:
				preds['corner_offset_uncertainty'] = torch.clamp(preds['corner_offset_uncertainty'], min=self.uncertainty_range[0], max=self.uncertainty_range[1])

			# else:
			# 	print('keypoint depth uncertainty: {:.2f} +/- {:.2f}'.format(
			# 		preds['corner_offset_uncertainty'].mean().item(), preds['corner_offset_uncertainty'].std().item()))

		if self.svd_inverse_with_uncertainty:
			preds['svd_inverse_uncertainty'] = pred_regression_pois_3D[:, self.key2channel('svd_inverse_uncertainty')].squeeze(-1)

			if self.uncertainty_range is not None:
				preds['svd_inverse_uncertainty'] = torch.clamp(preds['svd_inverse_uncertainty'], min=self.uncertainty_range[0], max=self.uncertainty_range[1])

		if self.six_planes_location_uncertainty:
			# 6个平面的不确定度
			preds['six_planes_uncertainty'] = pred_regression_pois_3D[:, self.key2channel('six_planes_uncertainty')] # (v,6)
		# compute the corners of the predicted 3D bounding boxes for the corner loss
		if self.corner_loss_depth == 'direct':
    		# 直接回归的深度值	
			pred_corner_depth_3D = pred_direct_depths_3D

		elif self.corner_loss_depth == 'keypoint_mean':
    		# 关键点预测的深度值方式的均值
			pred_corner_depth_3D = preds['keypoints_depths'].mean(dim=1)
		
		# Note: New added
		elif self.corner_loss_depth == 'svd_inverse':
    		# 关键点预测的深度值方式的均值
			pred_corner_depth_3D = pred_depth_by_points_svd_inv
		else:
			assert self.corner_loss_depth in ['soft_combine', 'hard_combine']
			# make sure all depths and their uncertainties are predicted
			# 将直接回归的深度值 与 关键点预测深度值 综合预测
			pred_combined_uncertainty = torch.cat((preds['depth_uncertainty'].unsqueeze(-1), preds['corner_offset_uncertainty'], preds['svd_inverse_uncertainty'].unsqueeze(-1)), dim=1).exp() # (valid_objects, 1+3+1)
			pred_combined_depths = torch.cat((pred_direct_depths_3D.unsqueeze(-1), preds['keypoints_depths'], pred_depth_by_points_svd_inv.unsqueeze(-1)), dim=1) # (valid_objects, 1+3+1)

			# pred_combined_uncertainty = torch.cat((preds['depth_uncertainty'].unsqueeze(-1), preds['corner_offset_uncertainty']), dim=1).exp() #(valid_objects, 1+N)
			# pred_combined_depths = torch.cat((pred_direct_depths_3D.unsqueeze(-1), preds['keypoints_depths']), dim=1)

			if self.corner_loss_depth == 'soft_combine':
				pred_uncertainty_weights = 1 / pred_combined_uncertainty
				pred_uncertainty_weights = pred_uncertainty_weights / pred_uncertainty_weights.sum(dim=1, keepdim=True)
				pred_corner_depth_3D = torch.sum(pred_combined_depths * pred_uncertainty_weights, dim=1)
				preds['weighted_depths'] = pred_corner_depth_3D
			
			elif self.corner_loss_depth == 'hard_combine':
				pred_corner_depth_3D = pred_combined_depths[torch.arange(pred_combined_depths.shape[0]), pred_combined_uncertainty.argmin(dim=1)]

		# compute the corners (center location)
		pred_locations_3D = self.anno_encoder.decode_location_flatten(valid_targets_bbox_points, pred_offset_3D, pred_corner_depth_3D, 
										targets_variables['calibs'], targets_variables['pad_size'], batch_idxs)

		# decode rotys and alphas
		# pred_rotys_3D, _ = self.anno_encoder.decode_axes_orientation(pred_orientation_3D, pred_locations_3D)
		# encode corners
		pred_corners_3D = self.anno_encoder.encode_box3d(pred_rotys_3D, pred_dimensions_3D, pred_locations_3D)

		# ============================================================================================================================ #
		# use svd-inv location calculate 3d iou
		# svd_pred_rotys_3D, _ = self.anno_encoder.decode_axes_orientation(pred_orientation_3D, svd_3d_location)
		svd_pred_rotys_3D = pred_rotys_3D
		svd_pred_corners_3D  = self.anno_encoder.encode_box3d(svd_pred_rotys_3D, pred_dimensions_3D, svd_3d_location)
		# ============================================================================================================================ #

		# concatenate all predictions
		pred_bboxes_3D = torch.cat((pred_locations_3D, pred_dimensions_3D, pred_rotys_3D[:, None]), dim=1)

		# ============================================================================================================================ #
		# new added
		svd_pred_bboxes_3D = torch.cat((svd_3d_location, pred_dimensions_3D, svd_pred_rotys_3D[:, None]), dim=1)
		# ============================================================================================================================ # 

		preds.update({'corners_3D': pred_corners_3D, 'svd_corners_3D':svd_pred_corners_3D,'rotys_3D': pred_rotys_3D, 'svd_rotys_3D':svd_pred_rotys_3D, 'cat_3D': pred_bboxes_3D, 'svd_cat_3d':svd_pred_bboxes_3D})

		return targets, preds, reg_nums, weights

	def __call__(self, predictions, targets):
		# predictions keys = ['cls', 'reg']
		targets_heatmap, targets_variables, targets_keypoint_heatmap = self.prepare_targets(targets)

		pred_heatmap = predictions['cls'] # 分类的heatmap torch.Size([2, 3, 96, 320])
		#=====================================================#
		pred_keypoint_heatmap = predictions['keypoint_heatmap']
		#=====================================================#
		pred_targets, preds, reg_nums, weights = self.prepare_predictions(targets_variables, predictions)
		# heatmap loss
		if self.heatmap_type == 'centernet':
			hm_loss, num_hm_pos = self.cls_loss_fnc(pred_heatmap, targets_heatmap)
			hm_loss = self.loss_weights['hm_loss'] * hm_loss / torch.clamp(num_hm_pos, 1)
		else: raise ValueError
		
		#================================================================================#
		if self.key_points_heat_map == 'centernet':
			# pred_keypoint_heatmap:torch.Size([2, 12, 96, 320])
			# targets_keypoint_heatmap: torch.Size([2, 12, 96, 320])
			kp_loss, num_kp_pos = self.cls_loss_fnc(pred_keypoint_heatmap, targets_keypoint_heatmap)
			kp_loss = self.loss_weights['keypoint_heatmap_loss'] * kp_loss / torch.clamp(num_kp_pos, 1)
			
		else: raise ValueError
		#================================================================================#
		
		# synthesize normal factors
		num_reg_2D = reg_nums['reg_2D']
		num_reg_3D = reg_nums['reg_3D']
		num_reg_obj = reg_nums['reg_obj']
		
		trunc_mask = pred_targets['trunc_mask_3D'].bool()
		num_trunc = trunc_mask.sum()
		num_nontrunc = num_reg_obj - num_trunc

		# IoU loss for 2D detection
		if num_reg_2D > 0:
			reg_2D_loss, iou_2D = self.iou_loss(preds['reg_2D'], pred_targets['reg_2D'])
			reg_2D_loss = self.loss_weights['bbox_loss'] * reg_2D_loss.mean()
			iou_2D = iou_2D.mean()
			depth_MAE = (preds['depth_3D'] - pred_targets['depth_3D']).abs() / pred_targets['depth_3D']
			# ====================================================================================================== #
			svd_depth_MAE = (preds['depth_by_svd_inverse'] - pred_targets['depth_3D']).abs() / pred_targets['depth_3D'] # New added
			# ====================================================================================================== #
			
		if num_reg_3D > 0:
			# direct depth loss
			if self.compute_direct_depth_loss:
				depth_3D_loss = self.loss_weights['depth_loss'] * self.depth_loss(preds['depth_3D'], pred_targets['depth_3D'], reduction='none')
				real_depth_3D_loss = depth_3D_loss.detach().mean()
				
				if self.depth_with_uncertainty:
					depth_3D_loss = depth_3D_loss * torch.exp(- preds['depth_uncertainty']) + \
							preds['depth_uncertainty'] * self.loss_weights['depth_loss']
				
				depth_3D_loss = depth_3D_loss.mean()

			# ======================================================================================================================================= #
			# svd depth loss 
			if self.compute_svd_inverse_depth_loss: 
				svd_inverse_depth_3D_loss = self.loss_weights['svd_inverse_loss'] * self.depth_loss(preds['depth_by_svd_inverse'], pred_targets['depth_3D'], reduction='none')
				real_svd_inverse_depth_3D_loss = svd_inverse_depth_3D_loss.detach().mean()
				
				if self.svd_inverse_with_uncertainty:
					svd_inverse_depth_3D_loss = svd_inverse_depth_3D_loss * torch.exp(- preds['svd_inverse_uncertainty']) + \
							preds['svd_inverse_uncertainty'] * self.loss_weights['svd_inverse_loss']
				
				svd_inverse_depth_3D_loss = svd_inverse_depth_3D_loss.mean()
			# ======================================================================================================================================= #

			# offset_3D loss
			offset_3D_loss = self.reg_loss_fnc(preds['offset_3D'], pred_targets['offset_3D'], reduction='none').sum(dim=1)
			
			#========================================================================================================================================#
			# offset_local_keypoints loss
			# pred_targets['keypoints_local_offset'] =  [13, 10, 2]
			# preds['keypoints_local_offset'] = [13, 20]
			# preds['keypoints'] = [13,10,2]
			keypoints_offset_local_loss = self.reg_loss_fnc(preds['keypoints_local_offset'], pred_targets['keypoints_local_offset'].view(-1,self.keypoints_num*2), reduction='none').sum(dim=1)
			#========================================================================================================================================#
			
			# use different loss functions for inside and outside objects
			if self.separate_trunc_offset:
				if self.trunc_offset_loss_type == 'L1':
					trunc_offset_loss = offset_3D_loss[trunc_mask]
				
				elif self.trunc_offset_loss_type == 'log':
					trunc_offset_loss = torch.log(1 + offset_3D_loss[trunc_mask])

				trunc_offset_loss = self.loss_weights['trunc_offset_loss'] * trunc_offset_loss.sum() / torch.clamp(trunc_mask.sum(), min=1)
				offset_3D_loss = self.loss_weights['offset_loss'] * offset_3D_loss[~trunc_mask].mean()
				# New added
				keypoints_offset_local_loss = self.loss_weights['keypoints_offset_local_loss'] * keypoints_offset_local_loss[~trunc_mask].mean()
			else:
				offset_3D_loss = self.loss_weights['offset_loss'] * offset_3D_loss.mean()
				# New added
				keypoints_offset_local_loss = self.loss_weights['keypoints_offset_local_loss'] * keypoints_offset_local_loss.mean()

			# orientation loss
			if self.multibin:
				orien_3D_loss = self.loss_weights['orien_loss'] * \
								Real_MultiBin_loss(preds['orien_3D'], pred_targets['orien_3D'], num_bin=self.orien_bin_size)

			# dimension loss
			dims_3D_loss = self.reg_loss_fnc(preds['dims_3D'], pred_targets['dims_3D'], reduction='none') * self.dim_weight.type_as(preds['dims_3D'])
			dims_3D_loss = self.loss_weights['dims_loss'] * dims_3D_loss.sum(dim=1).mean()

			with torch.no_grad(): pred_IoU_3D = get_iou_3d(preds['corners_3D'].cpu(), pred_targets['corners_3D'].cpu()).mean()

			# NEW Added
			with torch.no_grad(): svd_pred_IoU_3D = get_iou_3d(preds['svd_corners_3D'].cpu(), pred_targets['corners_3D'].cpu()).mean()

			# corner loss
			if self.compute_corner_loss:
				# N x 8 x 3
				corner_3D_loss = self.loss_weights['corner_loss'] * \
							self.reg_loss_fnc(preds['corners_3D'], pred_targets['corners_3D'], reduction='none').sum(dim=2).mean()

				# ================================================================================================================== #
				# add new svd_corner_3D_loss
				svd_corner_3D_loss = self.loss_weights['svd_corner_loss'] * \
							self.reg_loss_fnc(preds['svd_corners_3D'], pred_targets['corners_3D'], reduction='none').sum(dim=2).mean()
				# ================================================================================================================== #			
				'''
				four_heghts_edges = preds['corners_3D'][:,self.bottom_face_index,:] - preds['corners_3D'][:,self.top_face_index,:]
				four_lengths_edges = preds['corners_3D'][:,self.right_face_index,:]  - preds['corners_3D'][:,self.left_face_index,:] 
				four_width_edges = preds['corners_3D'][:,self.back_face_index,:]   - preds['corners_3D'][:,self.front_face_index,:]  
				
				vertical_3D_loss = (torch.abs(torch.sum(four_heghts_edges * four_lengths_edges, dim=2)) + \
							        torch.abs(torch.sum(four_width_edges * four_heghts_edges, dim=2))+ \
							        torch.abs(torch.sum(four_lengths_edges * four_width_edges, dim=2)))

				horizon_heght_loss =    (torch.abs(four_heghts_edges[:,0,0] * four_heghts_edges[:,1,1] - four_heghts_edges[:,1,0] * four_heghts_edges[:,0,1]) + \
										 torch.abs(four_heghts_edges[:,0,1] * four_heghts_edges[:,1,2] - four_heghts_edges[:,1,1] * four_heghts_edges[:,0,2]) + \
										 torch.abs(four_heghts_edges[:,0,0] * four_heghts_edges[:,1,2] - four_heghts_edges[:,1,0] * four_heghts_edges[:,0,2]) + \
										 torch.abs(four_heghts_edges[:,1,0] * four_heghts_edges[:,2,1] - four_heghts_edges[:,2,0] * four_heghts_edges[:,1,1]) + \
										 torch.abs(four_heghts_edges[:,1,1] * four_heghts_edges[:,2,2] - four_heghts_edges[:,2,1] * four_heghts_edges[:,1,2]) + \
										 torch.abs(four_heghts_edges[:,1,0] * four_heghts_edges[:,2,2] - four_heghts_edges[:,2,0] * four_heghts_edges[:,1,2]) + \
										 torch.abs(four_heghts_edges[:,2,0] * four_heghts_edges[:,3,1] - four_heghts_edges[:,3,0] * four_heghts_edges[:,2,1]) + \
										 torch.abs(four_heghts_edges[:,2,1] * four_heghts_edges[:,3,2] - four_heghts_edges[:,3,1] * four_heghts_edges[:,2,2]) + \
										 torch.abs(four_heghts_edges[:,2,0] * four_heghts_edges[:,3,2] - four_heghts_edges[:,3,0] * four_heghts_edges[:,2,2]) + \
										 torch.abs(four_heghts_edges[:,3,0] * four_heghts_edges[:,0,1] - four_heghts_edges[:,0,0] * four_heghts_edges[:,3,1]) + \
										 torch.abs(four_heghts_edges[:,3,1] * four_heghts_edges[:,0,2] - four_heghts_edges[:,0,1] * four_heghts_edges[:,3,2]) + \
										 torch.abs(four_heghts_edges[:,3,0] * four_heghts_edges[:,0,2] - four_heghts_edges[:,0,0] * four_heghts_edges[:,3,2])).sum()
				
				horizon_length_loss =  (torch.abs(four_lengths_edges[:,0,0] * four_lengths_edges[:,1,1] - four_lengths_edges[:,1,0] * four_lengths_edges[:,0,1]) + \
									   torch.abs(four_lengths_edges[:,0,1] * four_lengths_edges[:,1,2] - four_lengths_edges[:,1,1] * four_lengths_edges[:,0,2]) + \
									   torch.abs(four_lengths_edges[:,0,0] * four_lengths_edges[:,1,2] - four_lengths_edges[:,1,0] * four_lengths_edges[:,0,2]) + \
									   torch.abs(four_lengths_edges[:,1,0] * four_lengths_edges[:,2,1] - four_lengths_edges[:,2,0] * four_lengths_edges[:,1,1]) + \
									   torch.abs(four_lengths_edges[:,1,1] * four_lengths_edges[:,2,2] - four_lengths_edges[:,2,1] * four_lengths_edges[:,1,2]) + \
									   torch.abs(four_lengths_edges[:,1,0] * four_lengths_edges[:,2,2] - four_lengths_edges[:,2,0] * four_lengths_edges[:,1,2]) + \
									   torch.abs(four_lengths_edges[:,2,0] * four_lengths_edges[:,3,1] - four_lengths_edges[:,3,0] * four_lengths_edges[:,2,1]) + \
									   torch.abs(four_lengths_edges[:,2,1] * four_lengths_edges[:,3,2] - four_lengths_edges[:,3,1] * four_lengths_edges[:,2,2]) + \
									   torch.abs(four_lengths_edges[:,2,0] * four_lengths_edges[:,3,2] - four_lengths_edges[:,3,0] * four_lengths_edges[:,2,2]) + \
									   torch.abs(four_lengths_edges[:,3,0] * four_lengths_edges[:,0,1] - four_lengths_edges[:,0,0] * four_lengths_edges[:,3,1]) + \
									   torch.abs(four_lengths_edges[:,3,1] * four_lengths_edges[:,0,2] - four_lengths_edges[:,0,1] * four_lengths_edges[:,3,2]) + \
									   torch.abs(four_lengths_edges[:,3,0] * four_lengths_edges[:,0,2] - four_lengths_edges[:,0,0] * four_lengths_edges[:,3,2])).sum()

				horizon_width_loss =   (torch.abs(four_width_edges[:,0,0] * four_width_edges[:,1,1] - four_width_edges[:,1,0] * four_width_edges[:,0,1]) + \
									   torch.abs(four_width_edges[:,0,1] * four_width_edges[:,1,2] - four_width_edges[:,1,1] * four_width_edges[:,0,2]) + \
									   torch.abs(four_width_edges[:,0,0] * four_width_edges[:,1,2] - four_width_edges[:,1,0] * four_width_edges[:,0,2]) + \
									   torch.abs(four_width_edges[:,1,0] * four_width_edges[:,2,1] - four_width_edges[:,2,0] * four_width_edges[:,1,1]) + \
									   torch.abs(four_width_edges[:,1,1] * four_width_edges[:,2,2] - four_width_edges[:,2,1] * four_width_edges[:,1,2]) + \
									   torch.abs(four_width_edges[:,1,0] * four_width_edges[:,2,2] - four_width_edges[:,2,0] * four_width_edges[:,1,2]) + \
									   torch.abs(four_width_edges[:,2,0] * four_width_edges[:,3,1] - four_width_edges[:,3,0] * four_width_edges[:,2,1]) + \
									   torch.abs(four_width_edges[:,2,1] * four_width_edges[:,3,2] - four_width_edges[:,3,1] * four_width_edges[:,2,2]) + \
									   torch.abs(four_width_edges[:,2,0] * four_width_edges[:,3,2] - four_width_edges[:,3,0] * four_width_edges[:,2,2]) + \
									   torch.abs(four_width_edges[:,3,0] * four_width_edges[:,0,1] - four_width_edges[:,0,0] * four_width_edges[:,3,1]) + \
									   torch.abs(four_width_edges[:,3,1] * four_width_edges[:,0,2] - four_width_edges[:,0,1] * four_width_edges[:,3,2]) + \
									   torch.abs(four_width_edges[:,3,0] * four_width_edges[:,0,2] - four_width_edges[:,0,0] * four_width_edges[:,3,2])).sum()
				
				horizon_3D_loss = horizon_heght_loss + horizon_length_loss + horizon_width_loss
				'''

			if self.compute_keypoint_corner:
				# N x K x 3

				keypoint_loss = self.loss_weights['keypoint_loss'] * self.keypoint_loss_fnc(preds['keypoints'],
									 pred_targets['keypoints'], reduction='none').sum(dim=2) * pred_targets['keypoints_mask']
				keypoint_loss = keypoint_loss.sum() / torch.clamp(pred_targets['keypoints_mask'].sum(), min=1)
				# ========================================================================================================= #
				if self.compute_poisition_loss:
    				# svd-inverse calculate loc
					# keypoint_poisition_loss = self.keypoint_position_loss(preds, pred_targets, self.output_width, self.down_ratio, self.six_planes_location_uncertainty)
					
					svd_location_loss = preds['svd_3d_location'] - pred_targets['3D_center']
					svd_location_loss_norm = torch.norm(svd_location_loss, p=2, dim=1)
					svd_location_mask_num = (svd_location_loss_norm != 0).sum()
					svd_location_loss = svd_location_loss_norm.sum() / (svd_location_mask_num + 1)

					plane_cpt_loss = preds['plane_center_points'] - pred_targets['3D_plane_center']
					plane_cpt_loss_norm = torch.norm(plane_cpt_loss, p=2, dim=2) * preds['six_planes_uncertainty']
					plane_cpt_loss_norm = torch.norm(plane_cpt_loss_norm, p=1, dim=1)
					plane_mask_num = (plane_cpt_loss_norm!= 0).sum() #  mask_num =1
					plane_cpt_loss =  plane_cpt_loss_norm.sum() / (plane_mask_num + 1)

					keypoint_poisition_loss = svd_location_loss + plane_cpt_loss*self.loss_weights['plane_weight_loss']
				# ======================================================================================================== #
				if self.compute_keypoints_local_offsets_loss:
					kys = (pred_targets['keypoints_index'] / self.output_width).int().float() # (valid_objs, 10)
					kxs = (pred_targets['keypoints_index'] % self.output_width).int().float() # (valid_objs, 10)
					
					key_points_location = torch.stack([kxs,kys], dim=2) # (valid_objs, 10, 2)
					
					# key_points_location_gt = key_points_location + preds['keypoints_local_offset'].view(-1,self.keypoints_num,2)
					key_points_location_gt = key_points_location + pred_targets['keypoints_local_offset']
					key_points_location_pred = preds['pred_2D_keypoint']

					keypoint_location_loss = self.loss_weights['keypoint_loc_loss'] * self.keypoint_loc_loss_fnc(key_points_location_pred,
											 key_points_location_gt, reduction='none').sum(dim=[2]) * pred_targets['keypoints_mask']
					keypoint_location_loss = keypoint_location_loss.sum() / torch.clamp(pred_targets['keypoints_mask'].sum(), min=1)
					

				if self.compute_keypoint_depth_loss:
					pred_keypoints_depth, keypoints_depth_mask = preds['keypoints_depths'], pred_targets['keypoints_depth_mask'].bool()
					target_keypoints_depth = pred_targets['depth_3D'].unsqueeze(-1).repeat(1, self.keypoints_depth_mask_num)
					
					
					valid_pred_keypoints_depth = pred_keypoints_depth[keypoints_depth_mask]
					invalid_pred_keypoints_depth = pred_keypoints_depth[~keypoints_depth_mask].detach()
					
					# valid and non-valid
					valid_keypoint_depth_loss = self.loss_weights['keypoint_depth_loss'] * self.reg_loss_fnc(valid_pred_keypoints_depth, 
															target_keypoints_depth[keypoints_depth_mask], reduction='none')
					
					invalid_keypoint_depth_loss = self.loss_weights['keypoint_depth_loss'] * self.reg_loss_fnc(invalid_pred_keypoints_depth, 
															target_keypoints_depth[~keypoints_depth_mask], reduction='none')
					
					# for logging
					log_valid_keypoint_depth_loss = valid_keypoint_depth_loss.detach().mean()

					if self.corner_with_uncertainty:
						# center depth, corner 0246 depth, corner 1357 depth
						pred_keypoint_depth_uncertainty = preds['corner_offset_uncertainty']

						valid_uncertainty = pred_keypoint_depth_uncertainty[keypoints_depth_mask]
						invalid_uncertainty = pred_keypoint_depth_uncertainty[~keypoints_depth_mask]

						valid_keypoint_depth_loss = valid_keypoint_depth_loss * torch.exp(- valid_uncertainty) + \
												self.loss_weights['keypoint_depth_loss'] * valid_uncertainty

						invalid_keypoint_depth_loss = invalid_keypoint_depth_loss * torch.exp(- invalid_uncertainty)

					# average
					valid_keypoint_depth_loss = valid_keypoint_depth_loss.sum() / torch.clamp(keypoints_depth_mask.sum(), 1)
					invalid_keypoint_depth_loss = invalid_keypoint_depth_loss.sum() / torch.clamp((~keypoints_depth_mask).sum(), 1)

					# the gradients of invalid depths are not back-propagated
					if self.modify_invalid_keypoint_depths:
						keypoint_depth_loss = valid_keypoint_depth_loss + invalid_keypoint_depth_loss
					else:
						keypoint_depth_loss = valid_keypoint_depth_loss
				
				# compute the average error for each method of depth estimation
				# print(preds['keypoints_depths'])
				keypoint_MAE = (preds['keypoints_depths'] - pred_targets['depth_3D'].unsqueeze(-1)).abs() \
									/ pred_targets['depth_3D'].unsqueeze(-1)
				
				if self.keypoints_num == 10:
					center_MAE = keypoint_MAE[:, 0].mean()
					keypoint_02_or_05_MAE = keypoint_MAE[:, 1].mean()
					keypoint_13_or_14_MAE = keypoint_MAE[:, 2].mean()

					# ============================================ #
					# svd_depth_MAE = keypoint_MAE[:, 3].mean()
					# ============================================ # 
				elif self.keypoints_num == 12 and not self.use_area:
					# ============================================ #
					center_bt_MAE = keypoint_MAE[:, 0].mean()
					keypoint_02_MAE = keypoint_MAE[:, 1].mean()
					keypoint_13_MAE = keypoint_MAE[:, 2].mean()
					
					center_rl_MAE = keypoint_MAE[:, 3].mean()
					keypoint_05_MAE = keypoint_MAE[:, 4].mean()
					keypoint_14_MAE = keypoint_MAE[:, 5].mean()
					#========================================#
				elif self.keypoints_num == 12 and self.use_area:
					area_center_MAE = keypoint_MAE[:, 0].mean()
					front_face_MAE  = keypoint_MAE[:, 1].mean()
					back_face_MAE   = keypoint_MAE[:, 2].mean()
				elif self.keypoints_num == 27:
					center_MAE = keypoint_MAE[:, 0].mean()
					keypoint_04_or_020_MAE = keypoint_MAE[:, 1].mean()
					keypoint_26_or_218_MAE = keypoint_MAE[:, 2].mean()
					keypoint_15_or_119_MAE = keypoint_MAE[:, 3].mean()
					keypoint_37_or_321_MAE = keypoint_MAE[:, 4].mean()
				elif self.keypoints_num == 125:
					center_MAE = keypoint_MAE[:, 0].mean()
					keypoint_08_or_020_MAE  = keypoint_MAE[:, 1].mean()
					keypoint_412_or_218_MAE = keypoint_MAE[:, 2].mean()
					keypoint_19_or_321_MAE  = keypoint_MAE[:, 3].mean()
					keypoint_311_or_321_MAE = keypoint_MAE[:, 4].mean()
					keypoint_210_or_321_MAE = keypoint_MAE[:, 5].mean()
					keypoint_513_or_119_MAE = keypoint_MAE[:, 6].mean()
					keypoint_715_or_321_MAE = keypoint_MAE[:, 7].mean()
					keypoint_614_or_321_MAE = keypoint_MAE[:, 8].mean()
				else:
					raise ValueError
				if self.corner_with_uncertainty:
					if self.pred_direct_depth and self.depth_with_uncertainty and self.svd_inverse_with_uncertainty:
						# preds['depth_3D'] = (objs)
						# preds['keypoints_depths'] = (objs, 3)
						# preds['depth_by_svd_inverse'] = (objs)

						combined_depth = torch.cat((preds['depth_3D'].unsqueeze(1), preds['keypoints_depths'], preds['depth_by_svd_inverse'].unsqueeze(1)), dim=1)
						combined_uncertainty = torch.cat((preds['depth_uncertainty'].unsqueeze(1), preds['corner_offset_uncertainty'], preds['svd_inverse_uncertainty'].unsqueeze(-1)), dim=1).exp()
						combined_MAE = torch.cat((depth_MAE.unsqueeze(1), keypoint_MAE, svd_depth_MAE.unsqueeze(1)), dim=1)
						# combined_depth = torch.cat((preds['depth_3D'].unsqueeze(1), preds['keypoints_depths']), dim=1)
						# combined_uncertainty = torch.cat((preds['depth_uncertainty'].unsqueeze(1), preds['corner_offset_uncertainty']), dim=1).exp()
						# combined_MAE = torch.cat((depth_MAE.unsqueeze(1), keypoint_MAE), dim=1)
					else:
						combined_depth = preds['keypoints_depths']
						combined_uncertainty = preds['corner_offset_uncertainty'].exp()
						combined_MAE = keypoint_MAE

					# the oracle MAE
					lower_MAE = torch.min(combined_MAE, dim=1)[0]
					# the hard ensemble
					hard_MAE = combined_MAE[torch.arange(combined_MAE.shape[0]), combined_uncertainty.argmin(dim=1)]
					# the soft ensemble
					combined_weights = 1 / combined_uncertainty
					combined_weights = combined_weights / combined_weights.sum(dim=1, keepdim=True)
					soft_depths = torch.sum(combined_depth * combined_weights, dim=1)
					soft_MAE = (soft_depths - pred_targets['depth_3D']).abs() / pred_targets['depth_3D']
					# the average ensemble
					mean_depths = combined_depth.mean(dim=1)
					mean_MAE = (mean_depths - pred_targets['depth_3D']).abs() / pred_targets['depth_3D']

					# average
					lower_MAE, hard_MAE, soft_MAE, mean_MAE = lower_MAE.mean(), hard_MAE.mean(), soft_MAE.mean(), mean_MAE.mean()
				
					if self.compute_weighted_depth_loss:
						soft_depth_loss = self.loss_weights['weighted_avg_depth_loss'] * \
										  self.reg_loss_fnc(soft_depths, pred_targets['depth_3D'], reduction='mean')
				
			depth_MAE = depth_MAE.mean()		
			svd_depth_MAE = svd_depth_MAE.mean() # new added

		loss_dict = {
			'hm_loss':  hm_loss,
			'kp_loss':  kp_loss,
			'bbox_loss': reg_2D_loss,
			'dims_loss': dims_3D_loss,
			'orien_loss': orien_3D_loss,
		}

		log_loss_dict = {
			'2D_IoU': iou_2D.item(),
			'3D_IoU': pred_IoU_3D.item(),
			'SVD_3D_IoU':svd_pred_IoU_3D.item(), 
		}

		MAE_dict = {}

		if self.separate_trunc_offset:
			loss_dict['offset_loss'] = offset_3D_loss
			loss_dict['keypoints_offset_local_loss'] = keypoints_offset_local_loss # New added
			loss_dict['trunc_offset_loss'] = trunc_offset_loss
		else:
			loss_dict['keypoints_offset_local_loss'] = keypoints_offset_local_loss # New added
			loss_dict['offset_loss'] = offset_3D_loss

		if self.compute_corner_loss:
			loss_dict['corner_loss'] = corner_3D_loss
			loss_dict['svd_corner_loss'] = svd_corner_3D_loss # new added
			#loss_dict['vertical_3D_loss'] = vertical_3D_loss
			#loss_dict['horizon_3D_loss'] = horizon_3D_loss
			

		if self.pred_direct_depth:
			loss_dict['depth_loss'] = depth_3D_loss
			log_loss_dict['depth_loss'] = real_depth_3D_loss.item()
			MAE_dict['depth_MAE'] = depth_MAE.item()
		
		# ======================================================================= # 
		# New added
		if self.svd_inverse_with_uncertainty:
			loss_dict['svd_depth_loss'] = svd_inverse_depth_3D_loss
			log_loss_dict['svd_depth_loss'] = real_svd_inverse_depth_3D_loss.item()
			MAE_dict['svd_depth_MAE'] = svd_depth_MAE.item()	
		# ======================================================================= # 

		if self.compute_keypoint_corner:
			loss_dict['keypoint_loss'] = keypoint_loss
			if self.compute_keypoints_local_offsets_loss:
				loss_dict['keypoints_loc_loss'] = keypoint_location_loss
			if self.compute_poisition_loss:
				loss_dict['keypoint_poisition_loss'] = keypoint_poisition_loss
			
			if self.keypoints_num == 10:
				MAE_dict.update({
					'center_MAE': center_MAE.item(),
					'02_or_05_MAE': keypoint_02_or_05_MAE.item(),
					'13_or_14_MAE': keypoint_13_or_14_MAE.item(),
					
				})#'svd_depth_MAE':svd_depth_MAE.item()
			elif self.keypoints_num == 12 and not self.use_area:
				MAE_dict.update({
					'center_bt_MAE': center_bt_MAE.item(),
					'02_MAE': keypoint_02_MAE.item(),
					'13_MAE': keypoint_13_MAE.item(),
					
					'center_rl_MAE': center_rl_MAE.item(),
					'05_MAE': keypoint_05_MAE.item(),
					'14_MAE': keypoint_14_MAE.item(),
				})
			elif self.keypoints_num == 12 and self.use_area:
				MAE_dict.update({
					'area_center_MAE': area_center_MAE.item(),
					'1256_MAE': front_face_MAE.item(),
					'0347_MAE': back_face_MAE.item(),
				})
			elif self.keypoints_num == 27:
				MAE_dict.update({
					'center_MAE': center_MAE.item(),
					'04_or_020_MAE': keypoint_04_or_020_MAE.item(),
					'26_or_218_MAE': keypoint_26_or_218_MAE.item(),
					'15_or_119_MAE': keypoint_15_or_119_MAE.item(),
					'37_or_321_MAE': keypoint_37_or_321_MAE.item(),
				})
			elif self.keypoints_num == 125:
					MAE_dict.update({
					'center_MAE': center_MAE.item(),
					'04_or_020_MAE': keypoint_04_or_020_MAE.item(),
					'26_or_218_MAE': keypoint_26_or_218_MAE.item(),
					'15_or_119_MAE': keypoint_15_or_119_MAE.item(),
					'37_or_321_MAE': keypoint_37_or_321_MAE.item(),
				})

			if self.corner_with_uncertainty:
				MAE_dict.update({
					'lower_MAE': lower_MAE.item(),
					'hard_MAE': hard_MAE.item(),
					'soft_MAE': soft_MAE.item(),
					'mean_MAE': mean_MAE.item(),
				})

		if self.compute_keypoint_depth_loss:
			loss_dict['keypoint_depth_loss'] = keypoint_depth_loss
			log_loss_dict['keypoint_depth_loss'] = log_valid_keypoint_depth_loss.item()

		if self.compute_weighted_depth_loss:
			loss_dict['weighted_avg_depth_loss'] = soft_depth_loss

		# loss_dict ===> log_loss_dict
		for key, value in loss_dict.items():
			if key not in log_loss_dict:
				log_loss_dict[key] = value.item()

		# stop when the loss has NaN or Inf
		for v in loss_dict.values():		
			if torch.isnan(v).sum() > 0:
				pdb.set_trace()
			if torch.isinf(v).sum() > 0:
				pdb.set_trace()

		log_loss_dict.update(MAE_dict)

		return loss_dict, log_loss_dict

def Real_MultiBin_loss(vector_ori, gt_ori, num_bin=4):
	gt_ori = gt_ori.view(-1, gt_ori.shape[-1]) # bin1 cls, bin1 offset, bin2 cls, bin2 offst (valid_objs, 8)

	cls_losses = 0
	reg_losses = 0
	reg_cnt = 0
	for i in range(num_bin):
		# bin cls loss
		cls_ce_loss = F.cross_entropy(vector_ori[:, (i * 2) : (i * 2 + 2)], gt_ori[:, i].long(), reduction='none')
		# regression loss
		valid_mask_i = (gt_ori[:, i] == 1)
		cls_losses += cls_ce_loss.mean()
		if valid_mask_i.sum() > 0:
			s = num_bin * 2 + i * 2
			e = s + 2
			pred_offset = F.normalize(vector_ori[valid_mask_i, s : e])
			reg_loss =  F.l1_loss(pred_offset[:, 0], torch.sin(gt_ori[valid_mask_i, num_bin + i]), reduction='none') + \
						F.l1_loss(pred_offset[:, 1], torch.cos(gt_ori[valid_mask_i, num_bin + i]), reduction='none')

			reg_losses += reg_loss.sum()
			reg_cnt += valid_mask_i.sum()

	return cls_losses / num_bin + reg_losses / reg_cnt
