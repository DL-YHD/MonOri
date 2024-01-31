import torch
import torch.nn.functional as F
import numpy as np
import math
import torchvision

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2) # dim = 18
    
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim) # shape = (8,32,18)
    
    feat = feat.gather(1, ind) # 根据索引找到对应下标的值。
    # feat = [8, 32, 18]
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous() # 调换通道顺序,将(B,C,W,H) -> (B,W,H,C)
    feat = feat.view(feat.size(0), -1, feat.size(3)) # (B,96,320,18) ->(B,30720,18)
    feat = _gather_feat(feat, ind) #(B,32,18)
    return feat



class RegL1Loss(torch.nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        # output = [1, 2, 96, 320]
        # mask = [1, 288]
        # ind  = [1, 288]
        # pred = [1, 288, 2]
        pred = _transpose_and_gather_feat(output, ind) # 同样根据索引找到对应位置的预测值
        mask = mask.unsqueeze(2).expand_as(pred).float() # 将mask升维转换成和pred一样的形状
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        # loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class Position_loss(torch.nn.Module):
    def __init__(self):
        super(Position_loss, self).__init__()

        self.const = torch.Tensor([[-1, 0], [0, -1]])

    def svd_function(self, valid_objs, kps_num, rot_y, box_cpt_coef, A_temp, kp_norm, dim, occlusions=None, rand_num=4, occ_level=1):
        
        #===============================#
        # Note: Important 
        l = dim[:, 0] # (valid_objs)
        h = dim[:, 1] # (valid_objs)
        w = dim[:, 2] # (valid_objs)

        cosori = torch.cos(rot_y).cuda() # (valid_objs)
        sinori = torch.sin(rot_y).cuda() # (valid_objs)

        A = torch.zeros_like(A_temp)
        B = torch.zeros_like(kp_norm)
        C = torch.zeros_like(kp_norm)

        # 随机选取N个点求取svd伪逆
        if occlusions != None:
            A_mask = occlusions.unsqueeze(1).unsqueeze(2).repeat(1, A.size(1),A.size(2))
            B_mask = occlusions.unsqueeze(1).unsqueeze(2).repeat(1, B.size(1),B.size(2))
            C_mask = occlusions.unsqueeze(1).unsqueeze(2).repeat(1, C.size(1),C.size(2))

            rand_index = torch.from_numpy(np.random.choice(a = [0,1,2,3,4,5,6,7,8,9], size=rand_num, replace = False))

            A_OCC = torch.zeros_like(A_temp)   # (v, 20, 3)
            B_OCC = torch.zeros_like(kp_norm) # (v, 10, 2)
            C_OCC = torch.zeros_like(kp_norm) # (v, 10, 2)

            A_ALL = torch.zeros_like(A_temp)   # (v, 20, 3)
            B_ALL = torch.zeros_like(kp_norm) # (v, 10, 2)
            C_ALL = torch.zeros_like(kp_norm) # (v, 10, 2)

            for i in rand_index:
                A_OCC[:,i*2:i*2+1,:]   = A_temp[:,i*2:i*2+1,:]
                A_OCC[:,i*2+1:i*2+2,:] = A_temp[:,i*2+1:i*2+2,:]
					
            for i in rand_index: # 体心
                B_OCC[:,i,0] = l * box_cpt_coef[:,i,0] * cosori + w * box_cpt_coef[:,i,2] * sinori
                B_OCC[:,i,1] = h * box_cpt_coef[:,i,1]

            for i in rand_index:
                C_OCC[:,i,0] = l * box_cpt_coef[:,i,0] * (-sinori) + w * box_cpt_coef[:,i,2] * cosori
                C_OCC[:,i,1] = l * box_cpt_coef[:,i,0] * (-sinori) + w * box_cpt_coef[:,i,2] * cosori

            for i in range(kps_num):
                A_ALL[:,i*2:i*2+1,:]   = A_temp[:,i*2:i*2+1,:]
                A_ALL[:,i*2+1:i*2+2,:] = A_temp[:,i*2+1:i*2+2,:]
					
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
                A[:,i*2:i*2+1,:]   = A_temp[:,i*2:i*2+1,:]
                A[:,i*2+1:i*2+2,:] = A_temp[:,i*2+1:i*2+2,:]
					
            for i in range(kps_num): # 体心
                B[:,i,0] = l * box_cpt_coef[:,i,0] * cosori + w * box_cpt_coef[:,i,2] * sinori
                B[:,i,1] = h * box_cpt_coef[:,i,1]

            for i in range(kps_num):
                C[:,i,0] = l * box_cpt_coef[:,i,0] * (-sinori) + w * box_cpt_coef[:,i,2] * cosori
                C[:,i,1] = l * box_cpt_coef[:,i,0] * (-sinori) + w * box_cpt_coef[:,i,2] * cosori

        # rand_index = torch.from_numpy(np.random.choice(a = [0,1,2,3,4,5,6,7,8,9], size=rand_num, replace = False))                               
        # for i in rand_index:
        #     A[:,i*2:i*2+1,:]   = A_temp[:,i*2:i*2+1,:]
        #     A[:,i*2+1:i*2+2,:] = A_temp[:,i*2+1:i*2+2,:]
        
        # for i in rand_index: # 体心
        #     B[:,i,0] = l * box_cpt_coef[:,i,0] * cosori + w * box_cpt_coef[:,i,2] * sinori
        #     B[:,i,1] = h * box_cpt_coef[:,i,1]

        # for i in rand_index:
        #     C[:,i,0] = l * box_cpt_coef[:,i,0] * (-sinori) + w * box_cpt_coef[:,i,2] * cosori
        #     C[:,i,1] = l * box_cpt_coef[:,i,0] * (-sinori) + w * box_cpt_coef[:,i,2] * cosori

        B = B - kp_norm * C # (valid_objs, 10, 2)
        AT = A.permute(0, 2, 1) # (valid_objs, 3, 20)
        B  = B.contiguous().view(valid_objs, kps_num*2, 1).float()
        # mask = mask.unsqueeze(2)
        pinv = torch.bmm(AT, A)
        pinv = torch.inverse(pinv)  # b*c 3 3
        pinv = torch.bmm(pinv, AT)
        pinv = torch.bmm(pinv, B)
        pinv = pinv.contiguous().view(valid_objs, 3, 1).squeeze(2)
        return pinv

    def forward(self, pred, target, output_w, down_ratio, weight_flag=False):

        # pinv = pred['svd_3d_location']
        gt_3D_center = target['3D_center'] # (valid_objs,3)
        gt_plane_centers = target['3D_plane_center'] # (valid_objs, 6, 3)
        occlusions = target['occlusions'] # (valid_objs,)
        calib = target['calib'] # (valid_objs, 3, 4)
        pad_size = target['image_pad_size'] # (valid_objs, 10, 2)
        box_cpt_coef = target['coef'] # (valid_objs, 10, 3])
        box_cpt_coef = box_cpt_coef.cuda()

        front_cpt_coef = box_cpt_coef.clone()
        front_cpt_coef[:,:,0]  = box_cpt_coef[:,:,0] - 1/2
        
        back_cpt_coef = box_cpt_coef.clone()
        back_cpt_coef[:,:,0]   = box_cpt_coef[:,:,0] + 1/2
               
        bottom_cpt_coef = box_cpt_coef.clone()
        bottom_cpt_coef[:,:,1] = box_cpt_coef[:,:,1] - 1/2
        
        top_cpt_coef = box_cpt_coef.clone()
        top_cpt_coef[:,:,1]  = box_cpt_coef[:,:,1] + 1/2
               
        right_cpt_coef = box_cpt_coef.clone()
        right_cpt_coef[:,:,2]   = box_cpt_coef[:,:,2] - 1/2
        
        left_cpt_coef = box_cpt_coef.clone()
        left_cpt_coef[:,:,2]  = box_cpt_coef[:,:,2] + 1/2
       

        pred_dim = pred['dims_3D'] # (valid_objs,3) l,h,w
        pred_rot = pred['rotys_3D'] # (valid_objs)
        kps = pred['keypoints'] # keypoints (valid_objs, 10, 2)

        valid_objs  =  kps.size(0)
        kps_num     =  kps.size(1)
        
        cys = (target['center_points_index'] / output_w).int().float() # (valid_objs)
        cxs = (target['center_points_index'] % output_w).int().float() # (valid_objs)
        center_point = torch.stack([cxs,cys], dim=1) # (valid_objs,2)
        center_point = center_point.unsqueeze(1).repeat(1,kps_num,1) # (valid_objs,10, 2)

        # 还原原来的坐标
        kps = kps + center_point
        kps = kps * down_ratio - pad_size # [valid_objs, 10, 2]

        #calib = calib.unsqueeze(1)
        #calib = calib.expand(b, c, -1, -1).contiguous()
        f = calib[:, 0, 0].unsqueeze(1).unsqueeze(1) #(valid_objs) -> (valid_objs, 1, 1)
        f = f.expand_as(kps) #(valid_objs, 1, 1) -> (valid_objs, 10, 2)
        cx, cy = calib[:, 0, 2].unsqueeze(1), calib[:, 1, 2].unsqueeze(1)

        cxy = torch.cat((cx, cy), dim=1) # (valid_objs, 2)
        cxy = cxy.unsqueeze(1).repeat(1, kps_num, 1)  # (valid_objs, 2) -> (valid_objs, 10, 2)
        kp_norm = (kps.cuda() - cxy.cuda()) / f.cuda() # (valid_objs, 10, 2)
        
        kp = kp_norm.contiguous().view(valid_objs,-1).unsqueeze(2)  # (valid_objs, 10, 2) -> (valid_objs, 20) -> (valid_objs, 20, 1)
        const = self.const.repeat(kps_num,1).unsqueeze(0).cuda() # (2, 2) -> (20, 2) -> (1, 20, 2)
        const = const.expand(valid_objs, -1, -1)
        A_temp = torch.cat([const, kp.float()], dim=2) # (valid_objs, 20, 3)

        # 随机选取N个点求取svd伪逆
        # k = 4
        # index = torch.from_numpy(np.random.choice(a = [0,1,2,3,4,5,6,7,8,9], size=k, replace = False))

        pinv = self.svd_function(valid_objs, kps_num, pred_rot, box_cpt_coef, A_temp, kp_norm, pred_dim, occlusions=None, rand_num=4) # (valid_objs,3)

        # index = torch.from_numpy(np.random.choice(a = [0,1,2,3,4,5,6,7,8,9], size=k, replace = False))
        pinv_back = self.svd_function(valid_objs, kps_num, pred_rot, back_cpt_coef, A_temp, kp_norm, pred_dim, occlusions=None, rand_num=4)
        
        # index = torch.from_numpy(np.random.choice(a = [0,1,2,3,4,5,6,7,8,9], size=k, replace = False))
        pinv_front = self.svd_function(valid_objs, kps_num, pred_rot, front_cpt_coef, A_temp, kp_norm, pred_dim, occlusions=None, rand_num=4)
        
        # index = torch.from_numpy(np.random.choice(a = [0,1,2,3,4,5,6,7,8,9], size=k, replace = False))
        pinv_bottom = self.svd_function(valid_objs, kps_num, pred_rot, bottom_cpt_coef, A_temp, kp_norm, pred_dim, occlusions=None, rand_num=4)
        
        # index = torch.from_numpy(np.random.choice(a = [0,1,2,3,4,5,6,7,8,9], size=k, replace = False))
        pinv_top = self.svd_function(valid_objs, kps_num, pred_rot, top_cpt_coef, A_temp, kp_norm, pred_dim, occlusions=None, rand_num=4)
        
        # index = torch.from_numpy(np.random.choice(a = [0,1,2,3,4,5,6,7,8,9], size=k, replace = False))
        pinv_right = self.svd_function(valid_objs, kps_num, pred_rot, right_cpt_coef, A_temp, kp_norm, pred_dim, occlusions=None, rand_num=4)
        
        # index = torch.from_numpy(np.random.choice(a = [0,1,2,3,4,5,6,7,8,9], size=k, replace = False))
        pinv_left = self.svd_function(valid_objs, kps_num, pred_rot, left_cpt_coef, A_temp, kp_norm, pred_dim, occlusions=None, rand_num=4)

        # change the center to kitti center. Note that the pinv is the 3D center point in the camera coordinate system
        # pinv[:, :, 1] = pinv[:, :, 1] + dim[:, :, 0] / 2

        pinv_planes = torch.stack((pinv_back, pinv_front, pinv_bottom, pinv_top, pinv_right, pinv_left), dim=1) # (valid_objs,6,3)

        # print('pred_dim',pred_dim[0])
        # print('pinv',pinv[0])
        # print('gt_3D_center',gt_3D_center[0])
        # print('pinv_planes',pinv_planes[0])
        # print('gt_plane_center',gt_plane_centers[0])
        
        loss = pinv - gt_3D_center
        loss_norm = torch.norm(loss, p=2, dim=1)
        mask_num = (loss_norm != 0).sum()
        loss = loss_norm.sum() / (mask_num + 1)

        plane_cpt_loss = pinv_planes - gt_plane_centers
        plane_cpt_loss_norm = torch.norm(plane_cpt_loss, p=2, dim=2) # (valid,6)

        if weight_flag:
            six_plane_loss_weights = pred['six_planes_uncertainty'] # keypoints (valid_objs, 6)
            plane_cpt_loss_norm = six_plane_loss_weights * plane_cpt_loss_norm # (valid,6)
        
        plane_cpt_loss_norm = torch.norm(plane_cpt_loss_norm, p=1, dim=1)  # (valid,)
        plane_mask_num = (plane_cpt_loss_norm != 0).sum() #  mask_num =1
        plane_cpt_loss = plane_cpt_loss_norm.sum() / (plane_mask_num + 1) # loss = 29.2169
        total_loss = loss + plane_cpt_loss
        # ================================================================= #
        # process grad = 0 or grad = 1
        total_loss = torch.where(torch.isnan(total_loss), torch.full_like(total_loss, 0), total_loss)
        total_loss = torch.where(torch.isinf(total_loss), torch.full_like(total_loss, 1), total_loss)
        # ================================================================= #

        '''
        dim_gt = batch['dim'].clone()  # b,c,3
        # dim_gt[:, :, 0] = torch.exp(dim_gt[:, :, 0]) * 1.63
        # dim_gt[:, :, 1] = torch.exp(dim_gt[:, :, 1]) * 1.53
        # dim_gt[:, :, 2] = torch.exp(dim_gt[:, :, 2]) * 3.88
        location_gt = batch['location']
        ori_gt = batch['ori']
        dim_gt[dim_mask] = 0

        gt_box = torch.cat((location_gt, dim_gt, ori_gt), dim=2)
        box_pred = box_pred.view(b * c, -1)
        gt_box = gt_box.view(b * c, -1)

        box_score = boxes_iou3d_gpu(box_pred, gt_box)
        box_score = torch.diag(box_score).view(b, c)
        prob = prob.squeeze(2)
        box_score = box_score * loss_mask * dim_mask_score_mask
        loss_prob = F.binary_cross_entropy_with_logits(prob, box_score.detach(), reduce=False)
        loss_prob = loss_prob * loss_mask * dim_mask_score_mask
        loss_prob = torch.sum(loss_prob, dim=1)
        loss_prob = loss_prob.sum() / (mask_num + 1)
        box_score = box_score * loss_mask
        box_score = box_score.sum() / (mask_num + 1)
        '''
        return total_loss

class Uncertainty_Reg_Loss(torch.nn.Module):
    def __init__(self, reg_loss_fnc):
        super(Uncertainty_Reg_Loss, self).__init__()
        self.reg_loss_fnc = reg_loss_fnc

    def forward(self, pred, target, uncertainty):
        reg_loss = self.reg_loss_fnc(pred, target)
        reg_loss = reg_loss * torch.exp(- uncertainty) + 0.5 * uncertainty

        return loss

class Laplace_Loss(torch.nn.Module):
    def __init__(self):
        super(Laplace_Loss, self).__init__()

    def forward(self, pred, target, reduction='none'):
        # pred/target: K x ...
        loss = (1 - pred / target).abs() 
        return loss

def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = (p.grad ** 2).sum()
            totalnorm += modulenorm
    
    totalnorm = torch.sqrt(totalnorm)

    norm = clip_norm / torch.max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)

def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

class Wing_Loss(torch.nn.Module):
    def __init__(self, w=10, eps=2):
        super(Wing_Loss, self).__init__()
        self.w = w
        self.eps = eps
        self.C = w - w * np.log(1 + w / eps)

    def forward(self, prediction, target):
        differ = (prediction - target).abs()
        log_idxs = (differ < self.w).nonzero()
        l1_idxs = (differ >= self.w).nonzero()

        loss = prediction.new_zeros(prediction.shape[0])
        loss[log_idxs] = self.w * torch.log(differ[log_idxs] / self.eps + 1)
        loss[l1_idxs] = differ[l1_idxs] - self.C

        return loss


if __name__ == '__main__':
    num = 10000
    a = torch.zeros(num)
    b = (torch.arange(num).float() - (num / 2)) / (num / 20)

    wing_loss_fnc = Wing_Loss()
    loss = wing_loss_fnc(a, b)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(b, loss)
    plt.show()

