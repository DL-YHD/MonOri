from cv2 import phase
import numpy as np
import pdb
import torch
import torch.nn.functional as F

import torchvision.ops.roi_align as roi_align
from data.datasets.kitti_utils import convertAlpha2Rot
PI = np.pi


def generate_with_probability():
    random_num = np.random.rand()  # 生成0到1之间的随机数
    if random_num <= 0.5:
        return True
    else:
        return False
	
class Anno_Encoder():
		def __init__(self, cfg):
			device = cfg.MODEL.DEVICE
			self.INF = 100000000
			self.EPS = 1e-3
			
			self.keypoints_num = cfg.DATASETS.KEYPOINTS_NUM
			self.use_area = cfg.DATASETS.USE_AREA
			
			self.use_bot_top_centers = cfg.DATASETS.USE_BOT_TOP_CENTERS
			self.use_right_left_centers = cfg.DATASETS.USE_RIGHT_LEFT_CENTERS
			
			# center related
			self.num_cls = len(cfg.DATASETS.DETECT_CLASSES)
			self.min_radius = cfg.DATASETS.MIN_RADIUS
			self.max_radius = cfg.DATASETS.MAX_RADIUS
			self.center_ratio = cfg.DATASETS.CENTER_RADIUS_RATIO
			self.target_center_mode = cfg.INPUT.HEATMAP_CENTER
			# if mode == 'max', centerness is the larger value, if mode == 'area', assigned to the smaller bbox
			self.center_mode = cfg.MODEL.HEAD.CENTER_MODE
			
			# depth related
			self.depth_mode = cfg.MODEL.HEAD.DEPTH_MODE
			self.depth_range = cfg.MODEL.HEAD.DEPTH_RANGE
			self.depth_ref = torch.as_tensor(cfg.MODEL.HEAD.DEPTH_REFERENCE).to(device=device)

			# dimension related
			self.dim_mean = torch.as_tensor(cfg.MODEL.HEAD.DIMENSION_MEAN).to(device=device)
			self.dim_std = torch.as_tensor(cfg.MODEL.HEAD.DIMENSION_STD).to(device=device)
			self.dim_modes = cfg.MODEL.HEAD.DIMENSION_REG

			# orientation related
			self.alpha_centers = torch.tensor([0, PI / 2, PI, - PI / 2]).to(device=device)
			self.multibin = (cfg.INPUT.ORIENTATION == 'multi-bin')
			self.orien_bin_size = cfg.INPUT.ORIENTATION_BIN_SIZE

			# offset related
			self.offset_mean = cfg.MODEL.HEAD.REGRESSION_OFFSET_STAT[0]
			self.offset_std = cfg.MODEL.HEAD.REGRESSION_OFFSET_STAT[1]

			# output info
			self.down_ratio = cfg.MODEL.BACKBONE.DOWN_RATIO
			self.output_height = cfg.INPUT.HEIGHT_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
			self.output_width = cfg.INPUT.WIDTH_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
			self.K = self.output_width * self.output_height

		@staticmethod
		def rad_to_matrix(rotys, N):
			device = rotys.device

			cos, sin = rotys.cos(), rotys.sin()

			i_temp = torch.tensor([[ 1, 0, 1],
								   [ 0, 1, 0],
								   [-1, 0, 1]]).to(dtype=torch.float32, device=device)

			ry = i_temp.repeat(N, 1).view(N, -1, 3)

			ry[:, 0, 0] *= cos
			ry[:, 0, 2] *= sin
			ry[:, 2, 0] *= sin
			ry[:, 2, 2] *= cos

			return ry

		def decode_box2d_fcos(self, centers, pred_offset, pad_size=None, out_size=None):
			box2d_center = centers.view(-1, 2)
			box2d = box2d_center.new(box2d_center.shape[0], 4).zero_()
			# left, top, right, bottom
			box2d[:, :2] = box2d_center - pred_offset[:, :2]
			box2d[:, 2:] = box2d_center + pred_offset[:, 2:]

			# for inference
			if pad_size is not None:
				N = box2d.shape[0]
				out_size = out_size[0]
				# upscale and subtract the padding
				box2d = box2d * self.down_ratio - pad_size.repeat(1, 2)
				# clamp to the image bound
				box2d[:, 0::2].clamp_(min=0, max=out_size[0].item() - 1)
				box2d[:, 1::2].clamp_(min=0, max=out_size[1].item() - 1)

			return box2d

		def encode_box3d(self, rotys, dims, locs):
			'''
			construct 3d bounding box for each object.
			Args:
					rotys: rotation in shape N
					dims: dimensions of objects
					locs: locations of objects

			Returns:

			'''
			if len(rotys.shape) == 2:
					rotys = rotys.flatten()
			if len(dims.shape) == 3:
					dims = dims.view(-1, 3)
			if len(locs.shape) == 3:
					locs = locs.view(-1, 3)

			device = rotys.device
			N = rotys.shape[0]
			
			ry = self.rad_to_matrix(rotys, N)
			# l, h, w
			dims_corners = dims.view(-1, 1).repeat(1, 8)
			dims_corners = dims_corners * 0.5
			dims_corners[:, 4:] = -dims_corners[:, 4:] # 前四个取反
			index = torch.tensor([[4, 5, 0, 1, 6, 7, 2, 3],
								  [0, 1, 2, 3, 4, 5, 6, 7],
								  [4, 0, 1, 5, 6, 2, 3, 7]]).repeat(N, 1).to(device=device)
			
			box_3d_object = torch.gather(dims_corners, 1, index)
			box_3d = torch.matmul(ry, box_3d_object.view(N, 3, -1))
			box_3d += locs.unsqueeze(-1).repeat(1, 1, 8)

			return box_3d.permute(0, 2, 1)

		def decode_depth(self, depths_offset):
			'''
			Transform depth offset to depth
			return depth (objs,)
			'''
			if self.depth_mode == 'exp':
				depth = depths_offset.exp()
			elif self.depth_mode == 'linear':
				depth = depths_offset * self.depth_ref[1] + self.depth_ref[0]
			elif self.depth_mode == 'inv_sigmoid':
				depth = 1 / torch.sigmoid(depths_offset) - 1
			else:
				raise ValueError

			if self.depth_range is not None:
				depth = torch.clamp(depth, min=self.depth_range[0], max=self.depth_range[1])

			return depth

		def decode_location_flatten(self, points, offsets, depths, calibs, pad_size, batch_idxs):
			batch_size = len(calibs)
			gts = torch.unique(batch_idxs, sorted=True).tolist()
			locations = points.new_zeros(points.shape[0], 3).float()
			points = (points + offsets) * self.down_ratio - pad_size[batch_idxs]

			for idx, gt in enumerate(gts):
				corr_pts_idx = torch.nonzero(batch_idxs == gt).squeeze(-1)
				calib = calibs[gt]
				# concatenate uv with depth
				corr_pts_depth = torch.cat((points[corr_pts_idx], depths[corr_pts_idx, None]), dim=1).float()
				locations[corr_pts_idx] = calib.project_image_to_rect(corr_pts_depth)

			return locations

		def decode_depth_from_keypoints(self, pred_offsets, pred_keypoints, pred_dimensions, calibs, avg_center=False):
			# pred_keypoints: K x 10 x 2, 8 vertices, bottom center and top center
			assert len(calibs) == 1 # for inference, batch size is always 1
			calib = calibs[0]
			# we only need the values of y
			pred_height_3D = pred_dimensions[:, 1]
			pred_keypoints = pred_keypoints.view(-1, 10, 2)
			# center height -> depth
			if avg_center:
				updated_pred_keypoints = pred_keypoints - pred_offsets.view(-1, 1, 2)
				center_height = updated_pred_keypoints[:, 8:10, 1]
				center_depth = calib.f_u * pred_height_3D.unsqueeze(-1) / (center_height.abs() * self.down_ratio * 2)
				center_depth = center_depth.mean(dim=1)
			else:
				center_height = pred_keypoints[:, 8, 1] - pred_keypoints[:, 9, 1]
				center_depth = calib.f_u * pred_height_3D / (center_height.abs() * self.down_ratio)
			
			# corner height -> depth
			corner_02_height = pred_keypoints[:, [0, 2], 1] - pred_keypoints[:, [4, 6], 1]
			corner_13_height = pred_keypoints[:, [1, 3], 1] - pred_keypoints[:, [5, 7], 1]
			corner_02_depth = calib.f_u * pred_height_3D.unsqueeze(-1) / (corner_02_height * self.down_ratio)
			corner_13_depth = calib.f_u * pred_height_3D.unsqueeze(-1) / (corner_13_height * self.down_ratio)
			corner_02_depth = corner_02_depth.mean(dim=1)
			corner_13_depth = corner_13_depth.mean(dim=1)
			# K x 3
			pred_depths = torch.stack((center_depth, corner_02_depth, corner_13_depth), dim=1)

			return pred_depths

		def decode_depth_from_keypoints_batch(self, pred_keypoints, pred_dimensions, calibs, batch_idxs=None):
			# [198.23093523,  76.97646943],
			# [200.49017233,  78.74539486],
			# [174.49576232,  78.69863992],
			# [173.68051766,  76.93475746],
			# [198.23093523,  38.34185761],
			# [200.49017233,  37.84027947],
			# [174.49576232,  37.85353682],
			# [173.68051766,  38.35368502],
			# [186.69341031,  77.81360616],
			# [186.69341031,  38.10448781],
			# [199.32830822,  57.96695607],
			# [174.07652078,  57.95114917],
			# [185.94721513,  57.6516872 ],
			# [187.48342617,  58.28445662]]
			# pred_keypoints: K x 10 x 2, 8 vertices, bottom center and top center
			# [-0.2024, -0.3030],
			# [ 0.1719,  0.1034],
			# [ 0.1609,  0.5092],
			# [-0.1166,  0.5831],
			# [-0.1428, -0.0278],
			# [-0.3770, -0.1751],
			# [-0.2594,  0.1520],
			# [-0.1528,  0.3224],
			# [ 0.1256, -0.0619],
			# [ 0.0888, -0.3054],
			# [-0.1528,  0.2454],
			# [ 0.3370, -0.1053],
			# [-0.2279,  0.5220],
			# [ 0.2889,  0.0119],
			# [ 0.1223, -0.8117],
			# [ 0.2498,  0.3048],
			# [-0.0413, -0.2140],
			# [-0.0244, -0.4424],
			# [-0.2411,  0.1416],
			# [-0.9579, -0.7468],
			# [-0.3438,  0.0570],
			# [-0.0416,  0.3368],
			# [-0.1532,  0.0464],
			# [ 0.1407, -0.4024],
			# [-0.0835, -0.1022],
			# [-1.0294, -0.5221],
			# [ 0.5932, -0.5209]
			# pred_keypoints: bias offset

			pred_height_3D = pred_dimensions[:, 1].clone()
			pred_length_3D = pred_dimensions[:, 0].clone()
			#pred_width_3D  = pred_dimensions[:, 2].clone()
			
			
			batch_size = len(calibs)
			if batch_size == 1:
				batch_idxs = pred_dimensions.new_zeros(pred_dimensions.shape[0])
			
			if self.keypoints_num == 10 and self.use_bot_top_centers and not self.use_right_left_centers:
				center_height = pred_keypoints[:, 8, 1] - pred_keypoints[:, 9, 1] # bottom - top
				corner_02_height = pred_keypoints[:, [0, 2], 1] - pred_keypoints[:, [4, 6], 1]
				corner_13_height = pred_keypoints[:, [1, 3], 1] - pred_keypoints[:, [5, 7], 1]
				face_1256_height = pred_keypoints[:, [1, 2], 1] - pred_keypoints[:, [5, 6], 1]
				face_0347_height = pred_keypoints[:, [0, 3], 1] - pred_keypoints[:, [4, 7], 1]
				pred_keypoint_depths = {'center_bt': [], 'corner_02': [], 'corner_13': []}
			elif self.keypoints_num == 10 and self.use_right_left_centers and not self.use_bot_top_centers:
				center_length = pred_keypoints[:, 8, 0] - pred_keypoints[:, 9, 1] # right - left
				corner_05_length = pred_keypoints[:, [0, 2], 0] - pred_keypoints[:, [4, 6], 0]
				corner_14_length = pred_keypoints[:, [1, 3], 0] - pred_keypoints[:, [5, 7], 0]
				face_1256_length = pred_keypoints[:, [1, 2], 0] - pred_keypoints[:, [5, 6], 0]
				face_0347_length = pred_keypoints[:, [0, 3], 0] - pred_keypoints[:, [4, 7], 0]
				pred_keypoint_depths = {'center_rl': [],'corner_05': [], 'corner_14': []}
			elif self.keypoints_num == 12 and not self.use_area:
				center_height = pred_keypoints[:, 8, 1] - pred_keypoints[:, 9, 1] # bottom - top
				corner_02_height = pred_keypoints[:, [0, 2], 1] - pred_keypoints[:, [4, 6], 1]
				corner_13_height = pred_keypoints[:, [1, 3], 1] - pred_keypoints[:, [5, 7], 1]
				
				center_length = pred_keypoints[:, 10, 0] - pred_keypoints[:, 11, 0] # right - left
				corner_05_length = pred_keypoints[:, [0, 5], 0] - pred_keypoints[:, [3, 6], 0]
				corner_14_length = pred_keypoints[:, [1, 4], 0] - pred_keypoints[:, [2, 7], 0]
								
				pred_keypoint_depths = {'center_bt': [], 'corner_02': [], 'corner_13': [], 'center_rl': [],'corner_05': [], 'corner_14': []}
			elif self.keypoints_num == 12 and self.use_area:
				center_height = pred_keypoints[:, 8, 1] - pred_keypoints[:, 9, 1] # bottom - top
				face_1256_height = pred_keypoints[:, [1, 2], 1] - pred_keypoints[:, [5, 6], 1]
				face_0347_height = pred_keypoints[:, [0, 3], 1] - pred_keypoints[:, [4, 7], 1]
				
				center_length = pred_keypoints[:, 10, 0] - pred_keypoints[:, 11, 0] # right - left
				face_1256_length = pred_keypoints[:, [1, 2], 0] - pred_keypoints[:, [5, 6], 0]
				face_0347_length = pred_keypoints[:, [0, 3], 0] - pred_keypoints[:, [4, 7], 0]
				
				pred_keypoint_depths = {'area_ct_depth': [], 'area_two_depth': [], 'area_three_depth': []}
			elif self.keypoints_num == 27:
				center_height = pred_keypoints[:, 8, 1] - pred_keypoints[:, 26, 1] # bottom - top

				corner_04_height  = pred_keypoints[:, [0, 4], 1] - pred_keypoints[:, [18, 22], 1] #(obj, 2)
				corner_26_height  = pred_keypoints[:, [2, 6], 1] - pred_keypoints[:, [20, 24], 1] #(obj, 2)

				# edge center point hight
				center_15_height = pred_keypoints[:, [1, 5], 1] - pred_keypoints[:, [19, 23], 1] # bottom - top
				center_37_height = pred_keypoints[:, [3, 7], 1] - pred_keypoints[:, [21, 25], 1] # bottom - top


				face_back_height  = pred_keypoints[:, [0, 7, 6], 1] - pred_keypoints[:, [18, 25, 24], 1] #(obj, 3)
				face_front_height = pred_keypoints[:, [2, 3, 4], 1] - pred_keypoints[:, [20, 21, 22], 1] #(obj, 3)

				face_right_height  = pred_keypoints[:, [0, 1, 2], 1] - pred_keypoints[:, [18, 19, 20], 1] #(obj, 3)
				face_left_height   = pred_keypoints[:, [6, 5, 4], 1] - pred_keypoints[:, [24, 23, 22], 1] #(obj, 3)


				pred_keypoint_depths = {'center_bt': [], 'corner_04': [], 'corner_26': [], 'corner_15': [], 'corner_37': []}

			elif self.keypoints_num == 125:
				center_height = pred_keypoints[:, 24, 1] - pred_keypoints[:, 124, 1] # bottom - top
				corner_08_height  = pred_keypoints[:, [0,  8], 1] - pred_keypoints[:, [100, 108], 1]
				corner_412_height  = pred_keypoints[:,[4, 12], 1] - pred_keypoints[:, [104, 112], 1]
				face_back_height  = pred_keypoints[:, [0, 15, 14, 13, 12], 1] - pred_keypoints[:, [100, 115, 114, 113, 112], 1]
				face_front_height = pred_keypoints[:, [4,  5,  6,  7,  8], 1] - pred_keypoints[:, [104, 105, 106, 107, 108], 1]
				pred_keypoint_depths = {'center_bt': [], 'corner_08': [], 'corner_412': []}

			else:
				raise ValueError
			
			#center_width = pred_keypoints[:, 12, 2] - pred_keypoints[:, 13, 2] # back - front
			#corner_05_length = pred_keypoints[:, [0, 7], 2] - pred_keypoints[:, [1, 6], 2]
			#corner_14_length = pred_keypoints[:, [3, 4], 2] - pred_keypoints[:, [2, 5], 2]

			for idx, gt_idx in enumerate(torch.unique(batch_idxs, sorted=True).tolist()):
				calib = calibs[idx]
				corr_pts_idx = torch.nonzero(batch_idxs == gt_idx).squeeze(-1)
				if self.keypoints_num == 10 and self.use_bot_top_centers and not self.use_right_left_centers:
					center_bt_depth = calib.f_u * pred_height_3D[corr_pts_idx] / (F.relu(center_height[corr_pts_idx]) * self.down_ratio + self.EPS)
					corner_02_depth = calib.f_u * pred_height_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(corner_02_height[corr_pts_idx]) * self.down_ratio + self.EPS)
					corner_13_depth = calib.f_u * pred_height_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(corner_13_height[corr_pts_idx]) * self.down_ratio + self.EPS)
	
					corner_02_depth = corner_02_depth.mean(dim=1)
					corner_13_depth = corner_13_depth.mean(dim=1)
					
					pred_keypoint_depths['center_bt'].append(center_bt_depth)
					pred_keypoint_depths['corner_02'].append(corner_02_depth)
					pred_keypoint_depths['corner_13'].append(corner_13_depth)
					
				elif self.keypoints_num == 10 and self.use_right_left_centers and not self.use_bot_top_centers:
					center_rl_depth = calib.f_v * pred_length_3D[corr_pts_idx] / (F.relu(center_length[corr_pts_idx]) * self.down_ratio + self.EPS)
					corner_05_depth = calib.f_v * pred_length_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(corner_05_length[corr_pts_idx]) * self.down_ratio + self.EPS)
					corner_14_depth = calib.f_v * pred_length_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(corner_14_length[corr_pts_idx]) * self.down_ratio + self.EPS)
					
					corner_05_depth = corner_05_depth.mean(dim=1)
					corner_14_depth = corner_14_depth.mean(dim=1)
					
					pred_keypoint_depths['center_rl'].append(center_rl_depth)
					pred_keypoint_depths['corner_05'].append(corner_05_depth)
					pred_keypoint_depths['corner_14'].append(corner_14_depth)

				elif self.keypoints_num == 12 and not self.use_area:
					center_bt_depth = calib.f_u * pred_height_3D[corr_pts_idx] / (F.relu(center_height[corr_pts_idx]) * self.down_ratio + self.EPS)
					corner_02_depth = calib.f_u * pred_height_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(corner_02_height[corr_pts_idx]) * self.down_ratio + self.EPS)
					corner_13_depth = calib.f_u * pred_height_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(corner_13_height[corr_pts_idx]) * self.down_ratio + self.EPS)
	
					corner_02_depth = corner_02_depth.mean(dim=1)
					corner_13_depth = corner_13_depth.mean(dim=1)
					
					center_rl_depth = calib.f_v * pred_length_3D[corr_pts_idx] / (F.relu(center_length[corr_pts_idx]) * self.down_ratio + self.EPS)
					corner_05_depth = calib.f_v * pred_length_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(corner_05_length[corr_pts_idx]) * self.down_ratio + self.EPS)
					corner_14_depth = calib.f_v * pred_length_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(corner_14_length[corr_pts_idx]) * self.down_ratio + self.EPS)
					
					corner_05_depth = corner_05_depth.mean(dim=1)
					corner_14_depth = corner_14_depth.mean(dim=1)
					
					pred_keypoint_depths['center_bt'].append(center_bt_depth)
					pred_keypoint_depths['corner_02'].append(corner_02_depth)
					pred_keypoint_depths['corner_13'].append(corner_13_depth)
					
					pred_keypoint_depths['center_rl'].append(center_rl_depth)
					pred_keypoint_depths['corner_05'].append(corner_05_depth)
					pred_keypoint_depths['corner_14'].append(corner_14_depth)
				elif self.keypoints_num == 12 and self.use_area:
					center_bt_depth = calib.f_u * pred_height_3D[corr_pts_idx] / (F.relu(center_height[corr_pts_idx]) * self.down_ratio + self.EPS)
					# 正面高度（2个值）
					face_1256_height_depth = calib.f_u * pred_height_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(face_1256_height[corr_pts_idx]) * self.down_ratio + self.EPS)
					# 背面高度（2个值）
					face_0347_height_depth = calib.f_u * pred_height_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(face_0347_height[corr_pts_idx]) * self.down_ratio + self.EPS)
					
					center_rl_depth = calib.f_v * pred_length_3D[corr_pts_idx] / (F.relu(center_length[corr_pts_idx]) * self.down_ratio + self.EPS)
					# 正面长度（2个值）
					face_1256_length_depth = calib.f_v * pred_length_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(face_1256_length[corr_pts_idx]) * self.down_ratio + self.EPS)
					# 背面长度（2个值）
					face_0347_length_depth = calib.f_v * pred_length_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(face_0347_length[corr_pts_idx]) * self.down_ratio + self.EPS)
					
					area_ct_depth = torch.sqrt(center_bt_depth * center_rl_depth)
					
					area_front_depth = torch.sqrt(face_1256_height_depth*face_1256_length_depth).mean(dim=1) # 正面两个面积深度的均值
					area_back_depth = torch.sqrt(face_0347_height_depth*face_0347_length_depth).mean(dim=1)  # 背面两个面积深度的均值
					area_two_depth = (area_front_depth + area_back_depth) / 2
					area_three_depth = (area_ct_depth + area_back_depth + area_back_depth) / 3
					
					pred_keypoint_depths['area_ct_depth'].append(area_ct_depth)
					pred_keypoint_depths['area_two_depth'].append(area_two_depth)
					pred_keypoint_depths['area_three_depth'].append(area_three_depth)

				elif self.keypoints_num == 27:
					center_bt_depth = calib.f_u * pred_height_3D[corr_pts_idx] / (F.relu(center_height[corr_pts_idx]) * self.down_ratio + self.EPS)
					corner_04_depth = calib.f_u * pred_height_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(corner_04_height[corr_pts_idx]) * self.down_ratio + self.EPS)
					corner_26_depth = calib.f_u * pred_height_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(corner_26_height[corr_pts_idx]) * self.down_ratio + self.EPS)

					corner_15_depth = calib.f_u * pred_height_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(center_15_height[corr_pts_idx]) * self.down_ratio + self.EPS)
					corner_37_depth = calib.f_u * pred_height_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(center_37_height[corr_pts_idx]) * self.down_ratio + self.EPS)

					corner_04_depth = corner_04_depth.mean(dim=1) # (1,2)-> (1)
					corner_26_depth = corner_26_depth.mean(dim=1) # (1,2)-> (1)
					
					corner_15_depth = corner_15_depth.mean(dim=1)
					corner_37_depth = corner_37_depth.mean(dim=1)

					
					pred_keypoint_depths['center_bt'].append(center_bt_depth)
					pred_keypoint_depths['corner_04'].append(corner_04_depth)
					pred_keypoint_depths['corner_26'].append(corner_26_depth)
					pred_keypoint_depths['corner_15'].append(corner_15_depth)
					pred_keypoint_depths['corner_37'].append(corner_37_depth)

				else:
					raise ValueError
				
			for key, depths in pred_keypoint_depths.items():
				pred_keypoint_depths[key] = torch.clamp(torch.cat(depths), min=self.depth_range[0], max=self.depth_range[1])

			pred_depths = torch.stack([depth for depth in pred_keypoint_depths.values()], dim=1)
			
			return pred_depths

		

		def decode_dimension(self, cls_id, dims_offset):
			'''
			retrieve object dimensions
			Args:
					cls_id: each object id
					dims_offset: dimension offsets, shape = (N, 3)

			Returns:

			'''
			cls_id = cls_id.flatten().long()
			cls_dimension_mean = self.dim_mean[cls_id, :]

			if self.dim_modes[0] == 'exp':
				dims_offset = dims_offset.exp()

			if self.dim_modes[2]:
				cls_dimension_std = self.dim_std[cls_id, :]
				dimensions = dims_offset * cls_dimension_std + cls_dimension_mean
			else:
				dimensions = dims_offset * cls_dimension_mean
				
			return dimensions

		def decode_axes_orientation(self, vector_ori, locations):
			'''
			retrieve object orientation
			Args:
					vector_ori: local orientation in [axis_cls, head_cls, sin, cos] format
					locations: object location

			Returns: for training we only need roty
							 for testing we need both alpha and roty

			'''
			if self.multibin:
				pred_bin_cls = vector_ori[:, : self.orien_bin_size * 2].view(-1, self.orien_bin_size, 2)
				pred_bin_cls = torch.softmax(pred_bin_cls, dim=2)[..., 1]
				orientations = vector_ori.new_zeros(vector_ori.shape[0])
				for i in range(self.orien_bin_size):
					mask_i = (pred_bin_cls.argmax(dim=1) == i)
					s = self.orien_bin_size * 2 + i * 2
					e = s + 2
					pred_bin_offset = vector_ori[mask_i, s : e]
					orientations[mask_i] = torch.atan2(pred_bin_offset[:, 0], pred_bin_offset[:, 1]) + self.alpha_centers[i]
			else:
				axis_cls = torch.softmax(vector_ori[:, :2], dim=1)
				axis_cls = axis_cls[:, 0] < axis_cls[:, 1]
				head_cls = torch.softmax(vector_ori[:, 2:4], dim=1)
				head_cls = head_cls[:, 0] < head_cls[:, 1]
				# cls axis
				orientations = self.alpha_centers[axis_cls + head_cls * 2]
				sin_cos_offset = F.normalize(vector_ori[:, 4:])
				orientations += torch.atan(sin_cos_offset[:, 0] / sin_cos_offset[:, 1])

			locations = locations.view(-1, 3)
			rays = torch.atan2(locations[:, 0], locations[:, 2])
			
			alphas = orientations
			rotys = alphas + rays

			larger_idx = (rotys > PI).nonzero()
			small_idx = (rotys < -PI).nonzero()
			if len(larger_idx) != 0:
					rotys[larger_idx] -= 2 * PI
			if len(small_idx) != 0:
					rotys[small_idx] += 2 * PI

			larger_idx = (alphas > PI).nonzero()
			small_idx = (alphas < -PI).nonzero()
			if len(larger_idx) != 0:
					alphas[larger_idx] -= 2 * PI
			if len(small_idx) != 0:
					alphas[small_idx] += 2 * PI
			
			return rotys, alphas
		''' 
		def points_index_select(self, rays, rot_y, num_kp = 6):

			visible_point_index   = torch.zeros(rot_y.shape[0], num_kp, dtype=torch.int32) # (v, 6)
			occlusion_point_index = torch.zeros(rot_y.shape[0], num_kp, dtype=torch.int32) # (v, 6)
			
			# 找到每个目标对应的区间
			quadrant_1_small_idx = (rot_y >= -PI).nonzero().flatten()
			quadrant_1_large_idx = (rot_y < -(3/4) * PI).nonzero().flatten()
			quadrant_1_idx = set(np.array(quadrant_1_small_idx.cpu())) & set(np.array(quadrant_1_large_idx.cpu()))
   
			quadrant_2_small_idx = (rot_y >= -((3/4) * PI)).nonzero().flatten()
			quadrant_2_large_idx = (rot_y <  -((1/2) * PI)).nonzero().flatten()
			quadrant_2_idx = set(np.array(quadrant_2_small_idx.cpu())) & set(np.array(quadrant_2_large_idx.cpu()))
			
			quadrant_3_small_idx = (rot_y >= -((1/2) * PI)).nonzero().flatten()
			quadrant_3_large_idx = (rot_y <  -((1/4) * PI)).nonzero().flatten()
			quadrant_3_idx = set(np.array(quadrant_3_small_idx.cpu())) & set(np.array(quadrant_3_large_idx.cpu()))
   
			quadrant_4_small_idx = (rot_y >=- ((1/4) * PI)).nonzero().flatten()
			quadrant_4_large_idx = (rot_y < 0).nonzero().flatten()
			quadrant_4_idx = set(np.array(quadrant_4_small_idx.cpu())) & set(np.array(quadrant_4_large_idx.cpu()))
   
			quadrant_5_small_idx = (rot_y >= 0).nonzero().flatten()
			quadrant_5_large_idx = (rot_y < (1/4) * PI).nonzero().flatten()
			quadrant_5_idx = set(np.array(quadrant_5_small_idx.cpu())) & set(np.array(quadrant_5_large_idx.cpu()))
   
			quadrant_6_small_idx = (rot_y >= (1/4) * PI).nonzero().flatten()
			quadrant_6_large_idx = (rot_y < (1/2) * PI).nonzero().flatten()
			quadrant_6_idx = set(np.array(quadrant_6_small_idx.cpu())) & set(np.array(quadrant_6_large_idx.cpu()))

			quadrant_7_small_idx = (rot_y >= (1/2) * PI).nonzero().flatten()
			quadrant_7_large_idx = (rot_y <  (3/4) * PI).nonzero().flatten()
			quadrant_7_idx = set(np.array(quadrant_7_small_idx.cpu())) & set(np.array(quadrant_7_large_idx.cpu()))
			
			quadrant_8_small_idx = (rot_y >= (3/4) * PI).nonzero().flatten()
			quadrant_8_large_idx = (rot_y < PI).nonzero().flatten()
			quadrant_8_idx = set(np.array(quadrant_8_small_idx.cpu())) & set(np.array(quadrant_8_large_idx.cpu()))

			rays_left_idx = (rays <= 0).nonzero().flatten()
			rays_right_idx= (rays > 0).nonzero().flatten()
			
			rays_left_idx_set  = set(np.array(rays_left_idx.cpu()))  # 图像左边索引
			rays_right_idx_set = set(np.array(rays_right_idx.cpu())) # 图像右边索引
			

			if len(quadrant_1_idx & rays_left_idx_set) != 0: # 图像左边
				intersection_idx = torch.tensor(list(quadrant_1_idx & rays_left_idx_set)) 
				visible_point_index[intersection_idx]   = torch.tensor([2, 3, 6, 7, 8, 9], dtype=torch.int32) # Back  Face 
				occlusion_point_index[intersection_idx] = torch.tensor([0, 1, 4, 5, 8, 9], dtype=torch.int32) # Front Face
			if len(quadrant_1_idx & rays_right_idx_set) != 0: # 图像右边
				intersection_idx = torch.tensor(list(quadrant_1_idx & rays_right_idx_set))
				visible_point_index[intersection_idx]   = torch.tensor([0, 1, 4, 5, 8, 9], dtype=torch.int32) # Front Face
				occlusion_point_index[intersection_idx] = torch.tensor([2, 3, 6, 7, 8, 9], dtype=torch.int32) # Back  Face 
				# visible_point_index   = [1, 2, 5, 6, 8, 9] # Left  Face
				# occlusion_point_index = [0, 3, 4, 7, 8, 9] # Right Face
    
			if len(quadrant_2_idx & rays_left_idx_set) != 0:
				intersection_idx = torch.tensor(list(quadrant_2_idx & rays_left_idx_set)) 
				visible_point_index[intersection_idx]   = torch.tensor([1, 2, 5, 6, 8, 9], dtype=torch.int32) # Left  Face 
				occlusion_point_index[intersection_idx] = torch.tensor([0, 3, 4, 7, 8, 9], dtype=torch.int32) # Right Face
			if len(quadrant_2_idx & rays_right_idx_set) != 0:
				intersection_idx = torch.tensor(list(quadrant_2_idx & rays_right_idx_set))
				visible_point_index[intersection_idx]   = torch.tensor([0, 3, 4, 7, 8, 9], dtype=torch.int32) # Right Face
				occlusion_point_index[intersection_idx] = torch.tensor([1, 2, 5, 6, 8, 9], dtype=torch.int32) # Left  Face
				# visible_point_index   = [0, 1, 4, 5, 8, 9] # Front Face
				# occlusion_point_index = [2, 3, 6, 7, 8, 9] # Back  Face
    
			if len(quadrant_3_idx & rays_left_idx_set) != 0:
				intersection_idx = torch.tensor(list(quadrant_3_idx & rays_left_idx_set))
				visible_point_index[intersection_idx]   = torch.tensor([1, 2, 5, 6, 8, 9], dtype=torch.int32) # Left  Face 
				occlusion_point_index[intersection_idx] = torch.tensor([0, 3, 4, 7, 8, 9], dtype=torch.int32) # Right Face
			if len(quadrant_3_idx & rays_right_idx_set) != 0:
				intersection_idx = torch.tensor(list(quadrant_3_idx & rays_right_idx_set))
				visible_point_index[intersection_idx]   = torch.tensor([0, 3, 4, 7, 8, 9], dtype=torch.int32) # Right Face 
				occlusion_point_index[intersection_idx] = torch.tensor([1, 2, 5, 6, 8, 9], dtype=torch.int32) # Left  Face 
				# visible_point_index   = [0, 1, 4, 5, 8, 9] # Front Face
				# occlusion_point_index = [2, 3, 6, 7, 8, 9] # Back  Face
    
			if len(quadrant_4_idx & rays_left_idx_set) != 0:
				intersection_idx = torch.tensor(list(quadrant_4_idx & rays_left_idx_set))
				visible_point_index[intersection_idx]   = torch.tensor([0, 1, 4, 5, 8, 9], dtype=torch.int32) # Front Face
				occlusion_point_index[intersection_idx] = torch.tensor([2, 3, 6, 7, 8, 9], dtype=torch.int32) # Back  Face 
			if len(quadrant_4_idx & rays_right_idx_set) != 0:
				intersection_idx = torch.tensor(list(quadrant_4_idx & rays_right_idx_set))
				visible_point_index[intersection_idx]   = torch.tensor([2, 3, 6, 7, 8, 9], dtype=torch.int32) # Back  Face 
				occlusion_point_index[intersection_idx] = torch.tensor([0, 1, 4, 5, 8, 9], dtype=torch.int32) # Front Face
				# visible_point_index   = [0, 3, 4, 7, 8, 9] # Right Face
				# occlusion_point_index = [1, 2, 5, 6, 8, 9] # Left  Face
    
			if len(quadrant_5_idx & rays_left_idx_set) != 0:
				intersection_idx = torch.tensor(list(quadrant_5_idx & rays_left_idx_set))
				visible_point_index[intersection_idx]   = torch.tensor([0, 1, 4, 5, 8, 9], dtype=torch.int32) # Front Face
				occlusion_point_index[intersection_idx] = torch.tensor([2, 3, 6, 7, 8, 9], dtype=torch.int32) # Back  Face
			if len(quadrant_5_idx & rays_right_idx_set) != 0:
				intersection_idx = torch.tensor(list(quadrant_5_idx & rays_right_idx_set))
				visible_point_index[intersection_idx]   = torch.tensor([2, 3, 6, 7, 8, 9], dtype=torch.int32) # Back  Face
				occlusion_point_index[intersection_idx] = torch.tensor([0, 1, 4, 5, 8, 9], dtype=torch.int32) # Front Face
				# visible_point_index   = [0, 3, 4, 7, 8, 9] # Right Face
				# occlusion_point_index = [1, 2, 5, 6, 8, 9] # Left  Face
    
			if len(quadrant_6_idx & rays_left_idx_set) != 0:
				intersection_idx = torch.tensor(list(quadrant_6_idx & rays_left_idx_set)) 
				visible_point_index[intersection_idx]   = torch.tensor([0, 3, 4, 7, 8, 9], dtype=torch.int32) # Right Face
				occlusion_point_index[intersection_idx] = torch.tensor([1, 2, 5, 6, 8, 9], dtype=torch.int32) # Left  Face
			if len(quadrant_6_idx & rays_right_idx_set) != 0:
				intersection_idx = torch.tensor(list(quadrant_6_idx & rays_right_idx_set))
				visible_point_index[intersection_idx]   = torch.tensor([1, 2, 5, 6, 8, 9], dtype=torch.int32) # Left  Face
				occlusion_point_index[intersection_idx] = torch.tensor([0, 3, 4, 7, 8, 9], dtype=torch.int32) # Right Face
				# visible_point_index   = [2, 3, 6, 7, 8, 9] # Back  Face
				# occlusion_point_index = [0, 1, 4, 5, 8, 9] # Front Face
    
			if len(quadrant_7_idx & rays_left_idx_set) != 0:
				intersection_idx = torch.tensor(list(quadrant_7_idx & rays_left_idx_set))
				visible_point_index[intersection_idx]   = torch.tensor([0, 3, 4, 7, 8, 9], dtype=torch.int32) # Right Face
				occlusion_point_index[intersection_idx] = torch.tensor([1, 2, 5, 6, 8, 9], dtype=torch.int32) # Left  Face
			if len(quadrant_7_idx & rays_right_idx_set) != 0:
				intersection_idx = torch.tensor(list(quadrant_7_idx & rays_right_idx_set))
				visible_point_index[intersection_idx]   = torch.tensor([1, 2, 5, 6, 8, 9], dtype=torch.int32) # Left  Face
				occlusion_point_index[intersection_idx] = torch.tensor([0, 3, 4, 7, 8, 9], dtype=torch.int32) # Right Face
				# visible_point_index   = [2, 3, 6, 7, 8, 9] # Back  Face
				# occlusion_point_index = [0, 1, 4, 5, 8, 9] # Front Face
    
			if len(quadrant_8_idx & rays_left_idx_set)!= 0:
				intersection_idx = torch.tensor(list(quadrant_8_idx & rays_left_idx_set))
				visible_point_index[intersection_idx]   = torch.tensor([2, 3, 6, 7, 8, 9], dtype=torch.int32) # Back  Face 
				occlusion_point_index[intersection_idx] = torch.tensor([0, 1, 4, 5, 8, 9], dtype=torch.int32) # Front Face
			if len(quadrant_8_idx & rays_right_idx_set) != 0:
				intersection_idx = torch.tensor(list(quadrant_8_idx & rays_right_idx_set))
				visible_point_index[intersection_idx]   = torch.tensor([0, 1, 4, 5, 8, 9], dtype=torch.int32) # Front Face
				occlusion_point_index[intersection_idx] = torch.tensor([2, 3, 6, 7, 8, 9], dtype=torch.int32) # Back  Face  
				# visible_point_index   = [1, 2, 5, 6, 8, 9] # Left  Face
				# occlusion_point_index = [0, 3, 4, 7, 8, 9] # Right Face
    
			# if len(quadrant_8_idx):
				# for i in range(len(rot_y)):
				#	visible_point_index[i]   = torch.from_numpy(np.random.choice(a = [0,1,2,3,4,5,6,7,8,9], size=num_kp, replace = False))
				#	occlusion_point_index[i] = torch.from_numpy(np.random.choice(a = [0,1,2,3,4,5,6,7,8,9], size=num_kp, replace = False))
				# visible_point_index   = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # All Points
				# occlusion_point_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # All Points

			return visible_point_index.type(torch.long), occlusion_point_index.type(torch.long)
		'''

		def points_index_select(self, rays, rot_y, num_kp = 6):

			visible_point_index   = torch.zeros(rot_y.shape[0], num_kp, dtype=torch.int32) # (v, 6)
			occlusion_point_index = torch.zeros(rot_y.shape[0], num_kp, dtype=torch.int32) # (v, 6)
			
			# 找到每个目标对应的区间
			quadrant_1_small_idx = (rot_y >= -PI).nonzero().flatten()
			quadrant_1_large_idx = (rot_y < -(3/4) * PI).nonzero().flatten()
			quadrant_1_idx = set(np.array(quadrant_1_small_idx.cpu())) & set(np.array(quadrant_1_large_idx.cpu()))
   
			quadrant_2_small_idx = (rot_y >= -((3/4) * PI)).nonzero().flatten()
			quadrant_2_large_idx = (rot_y <  -((1/4) * PI)).nonzero().flatten()
			quadrant_2_idx = set(np.array(quadrant_2_small_idx.cpu())) & set(np.array(quadrant_2_large_idx.cpu()))
   
			quadrant_3_small_idx = (rot_y >=- ((1/4) * PI)).nonzero().flatten()
			quadrant_3_large_idx = (rot_y < (1/4) * PI).nonzero().flatten()
			quadrant_3_idx = set(np.array(quadrant_3_small_idx.cpu())) & set(np.array(quadrant_3_large_idx.cpu()))
   
			quadrant_4_small_idx = (rot_y >= (1/4) * PI).nonzero().flatten()
			quadrant_4_large_idx = (rot_y <  (3/4) * PI).nonzero().flatten()
			quadrant_4_idx = set(np.array(quadrant_4_small_idx.cpu())) & set(np.array(quadrant_4_large_idx.cpu()))
			
			quadrant_5_small_idx = (rot_y >= (3/4) * PI).nonzero().flatten()
			quadrant_5_large_idx = (rot_y < PI).nonzero().flatten()
			quadrant_5_idx = set(np.array(quadrant_5_small_idx.cpu())) & set(np.array(quadrant_5_large_idx.cpu()))

			rays_left_idx = (rays <= 0).nonzero().flatten()
			rays_right_idx= (rays > 0).nonzero().flatten()
			rays_left_idx_set  = set(np.array(rays_left_idx))  # 图像左边索引
			rays_right_idx_set = set(np.array(rays_right_idx)) # 图像右边索引
			
			if len(quadrant_1_idx & rays_left_idx_set) != 0:  # 图像左边
				intersection_idx = torch.tensor(list(quadrant_1_idx & rays_left_idx_set)) 
				visible_point_index[intersection_idx]   = torch.tensor([2, 3, 6, 7, 8, 9], dtype=torch.int32) # Back  Face 
				occlusion_point_index[intersection_idx] = torch.tensor([0, 1, 4, 5, 8, 9], dtype=torch.int32) # Front Face
			if len(quadrant_1_idx & rays_right_idx_set) != 0: # 图像右边
				intersection_idx = torch.tensor(list(quadrant_1_idx & rays_right_idx_set))
				visible_point_index[intersection_idx]   = torch.tensor([0, 1, 4, 5, 8, 9], dtype=torch.int32) # Front Face
				occlusion_point_index[intersection_idx] = torch.tensor([2, 3, 6, 7, 8, 9], dtype=torch.int32) # Back  Face 

			if len(quadrant_2_idx & rays_left_idx_set) != 0:  # 图像左边
				intersection_idx = torch.tensor(list(quadrant_2_idx & rays_left_idx_set))
				visible_point_index[intersection_idx]   = torch.tensor([0, 3, 4, 7, 8, 9], dtype=torch.int32) # Right Face
				occlusion_point_index[intersection_idx] = torch.tensor([1, 2, 5, 6, 8, 9], dtype=torch.int32) # Left  Face 	
			if len(quadrant_2_idx & rays_right_idx_set) != 0: # 图像右边
				intersection_idx = torch.tensor(list(quadrant_2_idx & rays_right_idx_set))
				visible_point_index[intersection_idx]   = torch.tensor([1, 2, 5, 6, 8, 9], dtype=torch.int32) # Left  Face 
				occlusion_point_index[intersection_idx] = torch.tensor([0, 3, 4, 7, 8, 9], dtype=torch.int32) # Right Face
    
			if len(quadrant_3_idx & rays_left_idx_set) != 0:  # 图像左边
				intersection_idx = torch.tensor(list(quadrant_3_idx & rays_left_idx_set))
				visible_point_index[intersection_idx]   = torch.tensor([0, 1, 4, 5, 8, 9], dtype=torch.int32) # Back Face 
				occlusion_point_index[intersection_idx] = torch.tensor([2, 3, 6, 7, 8, 9], dtype=torch.int32) # Front Face				
			if len(quadrant_3_idx & rays_right_idx_set) != 0: # 图像右边
				intersection_idx = torch.tensor(list(quadrant_3_idx & rays_right_idx_set))
				visible_point_index[intersection_idx]   = torch.tensor([2, 3, 6, 7, 8, 9], dtype=torch.int32) # Front Face 
				occlusion_point_index[intersection_idx] = torch.tensor([0, 1, 4, 5, 8, 9], dtype=torch.int32) # Back Face
    
			if len(quadrant_4_idx & rays_left_idx_set) != 0:  # 图像左边
				intersection_idx = torch.tensor(list(quadrant_4_idx & rays_left_idx_set))
				visible_point_index[intersection_idx]   = torch.tensor([1, 2, 5, 6, 8, 9], dtype=torch.int32) # Right Face 
				occlusion_point_index[intersection_idx] = torch.tensor([0, 3, 4, 7, 8, 9], dtype=torch.int32) # Left Face
			if len(quadrant_4_idx & rays_right_idx_set) != 0: # 图像右边
				intersection_idx = torch.tensor(list(quadrant_4_idx & rays_right_idx_set))
				visible_point_index[intersection_idx]   = torch.tensor([0, 3, 4, 7, 8, 9], dtype=torch.int32) # Left Face
				occlusion_point_index[intersection_idx] = torch.tensor([1, 2, 5, 6, 8, 9], dtype=torch.int32) # Right Face 
    
			if len(quadrant_5_idx & rays_left_idx_set) != 0:  # 图像左边
				intersection_idx = torch.tensor(list(quadrant_5_idx & rays_left_idx_set))
				visible_point_index[intersection_idx]   = torch.tensor([2, 3, 6, 7, 8, 9], dtype=torch.int32) # Back  Face
				occlusion_point_index[intersection_idx] = torch.tensor([0, 1, 4, 5, 8, 9], dtype=torch.int32) # Front Face
			if len(quadrant_5_idx & rays_right_idx_set) != 0: # 图像右边
				intersection_idx = torch.tensor(list(quadrant_5_idx & rays_right_idx_set))
				visible_point_index[intersection_idx]   = torch.tensor([0, 1, 4, 5, 8, 9], dtype=torch.int32) # Front Face
				occlusion_point_index[intersection_idx] = torch.tensor([2, 3, 6, 7, 8, 9], dtype=torch.int32) # Back  Face

			return visible_point_index.type(torch.long), occlusion_point_index.type(torch.long)
		
		def plane_position_inference(self, rays, rot_y, pred_dim, A_temp, valid_objs, keypoint, kp_norm, kps_num, box_cpt_coef, occlusions=None, rand_num=4, occ_level=1, rand_flag=True):
				cosori = torch.cos(rot_y) # (V)
				sinori = torch.sin(rot_y) # (V)

				l = pred_dim[:,0] # (1)
				h = pred_dim[:,1] # (1)
				w = pred_dim[:,2] # (1)
		
				A = torch.zeros_like(A_temp)   # (v, 20, 3)
				B = torch.zeros_like(keypoint) # (v, 10, 2)
				C = torch.zeros_like(keypoint) # (v, 10, 2)
				
				'''
				# 随机选取N个点求取svd伪逆
				if occlusions != None:
					A_mask = occlusions.unsqueeze(1).unsqueeze(2).repeat(1, A.size(1),A.size(2))
					B_mask = occlusions.unsqueeze(1).unsqueeze(2).repeat(1, B.size(1),B.size(2))
					C_mask = occlusions.unsqueeze(1).unsqueeze(2).repeat(1, C.size(1),C.size(2))

					rand_index = torch.from_numpy(np.random.choice(a = [0,1,2,3,4,5,6,7,8,9], size=rand_num, replace = False))

					A_OCC = torch.zeros_like(A_temp)   # (v, 20, 3)
					B_OCC = torch.zeros_like(keypoint) # (v, 10, 2)
					C_OCC = torch.zeros_like(keypoint) # (v, 10, 2)

					A_ALL = torch.zeros_like(A_temp)   # (v, 20, 3)
					B_ALL = torch.zeros_like(keypoint) # (v, 10, 2)
					C_ALL = torch.zeros_like(keypoint) # (v, 10, 2)

					for i in rand_index:
						A_OCC[:,i*2,:]   = A_temp[:,i*2,:]
						A_OCC[:,i*2+1,:] = A_temp[:,i*2+1,:]
					
					for i in rand_index: # 体心
						B_OCC[:,i,0] = l * box_cpt_coef[:,i,0] * cosori + w * box_cpt_coef[:,i,2] * sinori
						B_OCC[:,i,1] = h * box_cpt_coef[:,i,1]

					for i in rand_index:
						C_OCC[:,i,0] = l * box_cpt_coef[:,i,0] * (-sinori) + w * box_cpt_coef[:,i,2] * cosori
						C_OCC[:,i,1] = l * box_cpt_coef[:,i,0] * (-sinori) + w * box_cpt_coef[:,i,2] * cosori

					for i in range(kps_num):
						A_ALL[:,i*2,:]   = A_temp[:,i*2,:]
						A_ALL[:,i*2+1,:] = A_temp[:,i*2+1,:]
					
					for i in range(kps_num): # 体心
						B_ALL[:,i,0] = l * box_cpt_coef[:,i,0] * cosori + w * box_cpt_coef[:,i,2] * sinori
						B_ALL[:,i,1] = h * box_cpt_coef[:,i,1]

					for i in range(kps_num):
						C_ALL[:,i,0] = l * box_cpt_coef[:,i,0] * (-sinori) + w * box_cpt_coef[:,i,2] * cosori
						C_ALL[:,i,1] = l * box_cpt_coef[:,i,0] * (-sinori) + w * box_cpt_coef[:,i,2] * cosori

					# 遮挡标记大于1的目标使用随机点，遮挡标记小与1的使用所有点，分别处理。
					A = A_OCC * (A_mask >= occ_level) + A_ALL * (A_mask < occ_level) 
					B = B_OCC * (B_mask >= occ_level) + B_ALL * (B_mask < occ_level)	
					C = C_OCC * (C_mask >= occ_level) + C_ALL * (C_mask < occ_level)
					
				else:
					for i in range(kps_num):
						A[:,i*2,:]   = A_temp[:,i*2,:]
						A[:,i*2+1,:] = A_temp[:,i*2+1,:]
					
					for i in range(kps_num): # 体心
						B[:,i,0] = l * box_cpt_coef[:,i,0] * cosori + w * box_cpt_coef[:,i,2] * sinori
						B[:,i,1] = h * box_cpt_coef[:,i,1]

					for i in range(kps_num):
						C[:,i,0] = l * box_cpt_coef[:,i,0] * (-sinori) + w * box_cpt_coef[:,i,2] * cosori
						C[:,i,1] = l * box_cpt_coef[:,i,0] * (-sinori) + w * box_cpt_coef[:,i,2] * cosori
				'''
				if occlusions != None and not rand_flag:
					# alphas = rot_y - rays
					prob = generate_with_probability() # 0.5概率
					visible_point_index, occlusion_point_index = self.points_index_select(rays, rot_y)
					# visible_point_index, occlusion_point_index = self.points_index_select(rays, alphas)	
					# valid_objs_idx = np.arange(valid_objs) 建立索引
					if prob:
						for idx in range(len(visible_point_index)): # 体心
							select_key_points = visible_point_index[idx]
							A[idx,select_key_points*2,:]   = A_temp[idx,select_key_points*2,:]
							A[idx,select_key_points*2+1,:] = A_temp[idx,select_key_points*2+1,:]
					
							B[idx,select_key_points,0] = l[idx] * box_cpt_coef[idx,select_key_points,0] * cosori[idx] + w[idx] * box_cpt_coef[idx,select_key_points,2] * sinori[idx]
							B[idx,select_key_points,1] = h[idx] * box_cpt_coef[idx,select_key_points,1]

							C[idx,select_key_points,0] = l[idx] * box_cpt_coef[idx,select_key_points,0] * (-sinori[idx]) + w[idx] * box_cpt_coef[idx,select_key_points,2] * cosori[idx]
							C[idx,select_key_points,1] = l[idx] * box_cpt_coef[idx,select_key_points,0] * (-sinori[idx]) + w[idx] * box_cpt_coef[idx,select_key_points,2] * cosori[idx]
					else:
						for idx in range(len(occlusion_point_index)): # 体心
							select_key_points = occlusion_point_index[idx]
							A[idx,select_key_points*2,:]   = A_temp[idx,select_key_points*2,:]
							A[idx,select_key_points*2+1,:] = A_temp[idx,select_key_points*2+1,:]
					
							B[idx,select_key_points,0] = l[idx] * box_cpt_coef[idx,select_key_points,0] * cosori[idx] + w[idx] * box_cpt_coef[idx,select_key_points,2] * sinori[idx]
							B[idx,select_key_points,1] = h[idx] * box_cpt_coef[idx,select_key_points,1]

							C[idx,select_key_points,0] = l[idx] * box_cpt_coef[idx,select_key_points,0] * (-sinori[idx]) + w[idx] * box_cpt_coef[idx,select_key_points,2] * cosori[idx]
							C[idx,select_key_points,1] = l[idx] * box_cpt_coef[idx,select_key_points,0] * (-sinori[idx]) + w[idx] * box_cpt_coef[idx,select_key_points,2] * cosori[idx]
					
				# 随机选取N个点求取svd伪逆
				if occlusions != None and rand_flag:
					rand_index = torch.from_numpy(np.random.choice(a = [0,1,2,3,4,5,6,7,8,9], size=rand_num, replace = False))

					A_mask = occlusions.unsqueeze(1).unsqueeze(2).repeat(1, A.size(1),A.size(2))
					B_mask = occlusions.unsqueeze(1).unsqueeze(2).repeat(1, B.size(1),B.size(2))
					C_mask = occlusions.unsqueeze(1).unsqueeze(2).repeat(1, C.size(1),C.size(2))

					A_OCC = torch.zeros_like(A_temp)   # (v, 20, 3)
					B_OCC = torch.zeros_like(keypoint) # (v, 10, 2)
					C_OCC = torch.zeros_like(keypoint) # (v, 10, 2)

					A_ALL = torch.zeros_like(A_temp)   # (v, 20, 3)
					B_ALL = torch.zeros_like(keypoint) # (v, 10, 2)
					C_ALL = torch.zeros_like(keypoint) # (v, 10, 2)
				
					for i in rand_index: # 体心
						A_OCC[:,i*2,:]   = A_temp[:,i*2,:]
						A_OCC[:,i*2+1,:] = A_temp[:,i*2+1,:]
					
						B_OCC[:,i,0] = l * box_cpt_coef[:,i,0] * cosori + w * box_cpt_coef[:,i,2] * sinori
						B_OCC[:,i,1] = h * box_cpt_coef[:,i,1]

						C_OCC[:,i,0] = l * box_cpt_coef[:,i,0] * (-sinori) + w * box_cpt_coef[:,i,2] * cosori
						C_OCC[:,i,1] = l * box_cpt_coef[:,i,0] * (-sinori) + w * box_cpt_coef[:,i,2] * cosori
					
					for i in range(kps_num): # 体心
						A_ALL[:,i*2,:]   = A_temp[:,i*2,:]
						A_ALL[:,i*2+1,:] = A_temp[:,i*2+1,:]
					
						B_ALL[:,i,0] = l * box_cpt_coef[:,i,0] * cosori + w * box_cpt_coef[:,i,2] * sinori
						B_ALL[:,i,1] = h * box_cpt_coef[:,i,1]

						C_ALL[:,i,0] = l * box_cpt_coef[:,i,0] * (-sinori) + w * box_cpt_coef[:,i,2] * cosori
						C_ALL[:,i,1] = l * box_cpt_coef[:,i,0] * (-sinori) + w * box_cpt_coef[:,i,2] * cosori

					# 遮挡标记大于1的目标使用随机点，遮挡标记小与1的使用所有点，分别处理。
					A = A_OCC * (A_mask >= occ_level) + A_ALL * (A_mask < occ_level) 
					B = B_OCC * (B_mask >= occ_level) + B_ALL * (B_mask < occ_level)	
					C = C_OCC * (C_mask >= occ_level) + C_ALL * (C_mask < occ_level)
					
				if occlusions == None:
					for i in range(kps_num): # 体心
						A[:,i*2,:]   = A_temp[:,i*2,:]
						A[:,i*2+1,:] = A_temp[:,i*2+1,:]
					
						B[:,i,0] = l * box_cpt_coef[:,i,0] * cosori + w * box_cpt_coef[:,i,2] * sinori
						B[:,i,1] = h * box_cpt_coef[:,i,1]

						C[:,i,0] = l * box_cpt_coef[:,i,0] * (-sinori) + w * box_cpt_coef[:,i,2] * cosori
						C[:,i,1] = l * box_cpt_coef[:,i,0] * (-sinori) + w * box_cpt_coef[:,i,2] * cosori

				B  = B - kp_norm * C # (v, 10, 2)
				AT = A.permute(0, 2, 1) # (V, 3, 20)

				B  = B.contiguous().view(valid_objs, kps_num*2, 1)
				pinv = torch.bmm(AT, A)
				# pinv = torch.inverse(pinv) # b*c 3 3 
				pinv = torch.inverse(pinv.to('cpu')).to('cuda') # b*c 3 3 To fit RTX4090
				pinv = torch.bmm(pinv, AT)
				pinv = torch.bmm(pinv, B)
				pinv = pinv.squeeze(2)
				
				depth = pinv[:,2]
				# gt_locs[:,1] = gt_locs[:,1] + gt_dims[:,1] / 2
				return pinv, depth
		
		def decode_location_by_keypoints_svd_inverse(self, vector_ori, keypoint, pred_dim, calib, pad_size, occlusions=None, pred_rotys=None, locations=None, target=None, mask=None, phase_flag=None):

			self.const = torch.Tensor([[-1, 0], [0, -1]])
			if self.keypoints_num == 10:
				self.box_cpt_coef = torch.Tensor( [[[ 1/2, 1/2, 1/2],
                                            		[ 1/2, 1/2,-1/2],
                                                	[-1/2, 1/2,-1/2],
                                                	[-1/2, 1/2, 1/2],

													[ 1/2,-1/2, 1/2],
													[ 1/2,-1/2,-1/2],
													[-1/2,-1/2,-1/2],
													[-1/2,-1/2, 1/2],

													[   0, 1/2,   0],
													[   0,-1/2,   0]]])
				# =========================================================== #
				self.left_cpt_coef = self.box_cpt_coef.clone()
				self.left_cpt_coef[:,:,0]   = self.box_cpt_coef[:,:,0] + 1/2
				
				self.right_cpt_coef = self.box_cpt_coef.clone()
				self.right_cpt_coef[:,:,0]  = self.box_cpt_coef[:,:,0] - 1/2
				
				self.top_cpt_coef = self.box_cpt_coef.clone()
				self.top_cpt_coef[:,:,1]  = self.box_cpt_coef[:,:,1] + 1/2
					
				self.bottom_cpt_coef = self.box_cpt_coef.clone()
				self.bottom_cpt_coef[:,:,1] = self.box_cpt_coef[:,:,1] - 1/2
									
				self.back_cpt_coef = self.box_cpt_coef.clone()
				self.back_cpt_coef[:,:,2]   = self.box_cpt_coef[:,:,2] + 1/2
				
				self.front_cpt_coef = self.box_cpt_coef.clone()
				self.front_cpt_coef[:,:,2]  = self.box_cpt_coef[:,:,2] - 1/2
				# =========================================================== #
			elif self.keypoints_num == 27:
				self.box_cpt_coef = torch.Tensor([[ [ 1/2, 1/2, 1/2],
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
													[   0, -1/2,   0]]])
			valid_objs  =  keypoint.size(0)
			kps_num     =  keypoint.size(1)
			box_cpt_coef = self.box_cpt_coef.repeat(valid_objs,1,1).cuda() #(v,10,3)
			# ======================================================================== #
			front_cpt_coef = self.front_cpt_coef.repeat(valid_objs,1,1).cuda() #(v,10,3)
			back_cpt_coef = self.back_cpt_coef.repeat(valid_objs,1,1).cuda() #(v,10,3)
			top_cpt_coef = self.top_cpt_coef.repeat(valid_objs,1,1).cuda() #(v,10,3)
			bottom_cpt_coef = self.bottom_cpt_coef.repeat(valid_objs,1,1).cuda() #(v,10,3)
			left_cpt_coef = self.left_cpt_coef.repeat(valid_objs,1,1).cuda() #(v,10,3)
			right_cpt_coef = self.right_cpt_coef.repeat(valid_objs,1,1).cuda() #(v,10,3)
			# ======================================================================== #

			keypoint = keypoint * self.down_ratio - pad_size # [v, 10, 2]
			center_point = (keypoint[:, 8, :] + keypoint[:, 9, :]) / 2 # center_point = (bottom + top) / 2  (valid,2)

			# target = target[0]
			# valid_mask = mask[:40]

			# num_gt = valid_mask.sum()
			# gt_clses = target.get_field('cls_ids')[valid_mask]
			# gt_boxes = target.get_field('gt_bboxes')[valid_mask]
			# gt_locs = target.get_field('locations')[valid_mask]
			# gt_kps = target.get_field('keypoints')[valid_mask]
			# gt_centers = target.get_field('target_centers')[valid_mask]
			# gt_dims = target.get_field('dimensions')[valid_mask]
			# gt_rotys = target.get_field('rotys')[valid_mask]

			# gt_depths = gt_locs[:, -1]
			# gt_boxes_center = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
			
			# print('pred_keypoint',keypoint)
			# keypoint = gt_kps[:,:,:2] +  gt_centers.unsqueeze(1).repeat(1,10,1) # use ground truth 
			# print('gt_keypoint',keypoint)

			if self.multibin:
					pred_bin_cls = vector_ori[:, : self.orien_bin_size * 2].view(-1, self.orien_bin_size, 2)
					pred_bin_cls = torch.softmax(pred_bin_cls, dim=2)[..., 1]
					orientations = vector_ori.new_zeros(vector_ori.shape[0])
					for i in range(self.orien_bin_size):
						mask_i = (pred_bin_cls.argmax(dim=1) == i)
						s = self.orien_bin_size * 2 + i * 2
						e = s + 2
						pred_bin_offset = vector_ori[mask_i, s : e]
						orientations[mask_i] = torch.atan2(pred_bin_offset[:, 0], pred_bin_offset[:, 1]) + self.alpha_centers[i]
			else:
				axis_cls = torch.softmax(vector_ori[:, :2], dim=1)
				axis_cls = axis_cls[:, 0] < axis_cls[:, 1]
				head_cls = torch.softmax(vector_ori[:, 2:4], dim=1)
				head_cls = head_cls[:, 0] < head_cls[:, 1]
				# cls axis
				orientations = self.alpha_centers[axis_cls + head_cls * 2]
				sin_cos_offset = F.normalize(vector_ori[:, 4:])
				orientations += torch.atan(sin_cos_offset[:, 0] / sin_cos_offset[:, 1])

			if phase_flag == 'Training':
				calib = calib.cuda()
				if locations == None:
    				# use project point to estimate
					si = torch.zeros_like(keypoint[:, 0, 0]) + calib[:, 0, 0] # (v,1)
					rays = torch.atan2(center_point[:,0] - calib[:, 0, 2], si) # (v)
				else:
					# use predict location to estimate 
					locations = locations.view(-1, 3)
					rays = torch.atan2(locations[:, 0], locations[:, 2])

				f = calib[:, 0, 0].unsqueeze(1).unsqueeze(2) #(objs) ->(objs, 1, 1)
				f = f.expand_as(keypoint).cuda() #(1) -> (v, 10, 2)
				cx, cy = calib[:, 0, 2].unsqueeze(1), calib[:, 1, 2].unsqueeze(1)			
				cxy = torch.cat((cx, cy), dim=1) # (valid_objs, 2)
				cxy = cxy.unsqueeze(1).repeat(1, kps_num, 1)  # (valid_objs, 2) -> (valid_objs, 10, 2)

			elif phase_flag == 'Inference':
				calib = torch.tensor(calib[0].P)
				if locations == None:
					# use project point to estimate
					si = torch.zeros_like(keypoint[:, 0, 0]) + calib[0, 0] # (v,1)
					rays = torch.atan2(center_point[:,0] - calib[0, 2], si) # (v)
				else:
					# use predict location to estimate 
					locations = locations.view(-1, 3)
					rays = torch.atan2(locations[:, 0], locations[:, 2])

				f = calib[0, 0] #(3,4) ->(1)
				f = f.expand_as(keypoint).cuda() #(1) -> (v, 10, 2)
				cx, cy = calib[0, 2].unsqueeze(0), calib[1, 2].unsqueeze(0)
				cxy = torch.cat((cx, cy), dim=0) # (2)
				cxy = cxy.expand_as(keypoint).cuda()  # (2) -> (v, 10, 2)

			else:
				print("Error: You Must Specified the phase_flag Value !")
				raise
			alphas = orientations
			rot_y = alphas + rays

			larger_idx = (rot_y > PI).nonzero()
			small_idx  = (rot_y < -PI).nonzero()

			if len(larger_idx) != 0:
					rot_y[larger_idx] -= 2 * PI
			if len(small_idx) != 0:
					rot_y[small_idx]  += 2 * PI
			
			larger_idx = (alphas > PI).nonzero()
			small_idx = (alphas < -PI).nonzero()
			if len(larger_idx) != 0:
					alphas[larger_idx] -= 2 * PI
			if len(small_idx) != 0:
					alphas[small_idx] += 2 * PI

			# rot_y = gt_rotys # use ground truth rotys
			# rot_y = pred_rotys.squeeze(1) # use predict rot_y
			kp_norm = (keypoint - cxy) / f # (V, 10, 2)

			# l = gt_dims[:,0] # (1) use gt_dims
			# h = gt_dims[:,1] # (1) use gt_dims
			# w = gt_dims[:,2] # (1) use gt_dims

			kp = kp_norm.contiguous().view(-1, kps_num*2).unsqueeze(2)  # (v, 10, 2) -> (v, 20) -> (v, 20, 1)
			const = self.const.repeat(valid_objs, kps_num, 1).cuda() # (v, 2, 2) -> (v, 20, 2)

			A_temp = torch.cat([const, kp], dim=2) # (v, 20, 3)

			pinv, depth = self.plane_position_inference(rays, rot_y, pred_dim, A_temp, valid_objs, keypoint, kp_norm, kps_num, box_cpt_coef, occlusions, 5) # (v, 3), (v,)
			
			front_pinv, front_depth = self.plane_position_inference(rays, rot_y, pred_dim, A_temp, valid_objs, keypoint, kp_norm, kps_num, front_cpt_coef, occlusions, 5)
			back_pinv, back_depth = self.plane_position_inference(rays, rot_y, pred_dim, A_temp, valid_objs, keypoint, kp_norm, kps_num, back_cpt_coef, occlusions, 5)
			
			top_pinv, top_depth = self.plane_position_inference(rays, rot_y, pred_dim, A_temp, valid_objs, keypoint, kp_norm, kps_num, top_cpt_coef, occlusions, 5)
			bottom_pinv, bottom_depth = self.plane_position_inference(rays, rot_y, pred_dim, A_temp, valid_objs, keypoint, kp_norm, kps_num, bottom_cpt_coef, occlusions, 5)
			
			left_pinv, left_depth = self.plane_position_inference(rays, rot_y, pred_dim, A_temp, valid_objs, keypoint, kp_norm, kps_num, left_cpt_coef, occlusions, 5)
			right_pinv, right_depth = self.plane_position_inference(rays, rot_y, pred_dim, A_temp, valid_objs, keypoint, kp_norm, kps_num, right_cpt_coef, occlusions, 5)

			plane_center_points = torch.stack([front_pinv, back_pinv, top_pinv, bottom_pinv, left_pinv, right_pinv], dim=1) # [v,6,3]
			
			return pinv, depth, rot_y, alphas, plane_center_points
	
if __name__ == '__main__':
	pass