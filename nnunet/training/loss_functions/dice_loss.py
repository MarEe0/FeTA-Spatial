#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
from nnunet.training.loss_functions.TopK_loss import TopKLoss
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn
import numpy as np
np.set_printoptions(threshold=np.inf)

#class GDL(nn.Module):

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn

#class SoftDiceLoss(nn.Module): Now InmaDiceLoss
#class MCCLoss(nn.Module):

class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc

class SoftDiceLossSquared(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        squares the terms in the denominator as proposed by Milletari et al.
        """
        super(SoftDiceLossSquared, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(x.shape, y.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y = y.long()
                y_onehot = torch.zeros(shp_x)
                if x.device.type == "cuda":
                    y_onehot = y_onehot.cuda(x.device.index)
                y_onehot.scatter_(1, y, 1).float()

        intersect = x * y_onehot
        # values in the denominator get smoothed
        denominator = x ** 2 + y_onehot ** 2

        # aggregation was previously done in get_tp_fp_fn, but needs to be done here now (needs to be done after
        # squaring)
        intersect = sum_tensor(intersect, axes, False) + self.smooth
        denominator = sum_tensor(denominator, axes, False) + self.smooth

        dc = 2 * intersect / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_sr = 0.5
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        self.ignore_label = ignore_label
                
        
        self.spatial_rel = GraphSpatialLoss3D()

        if not square_dice:
            self.dc = InmaDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs) # InmaDiceLoss, SoftDiceLoss
            self.dc2 = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs) # InmaDiceLoss, SoftDiceLoss
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        #print('net_output shape: ', net_output.shape)
        #print('target shape: ', net_output.shape)
        #net_output shape:  torch.Size([2, 15, 128, 128, 128])
		#target shape:  torch.Size([2, 1, 128, 128, 128])

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss2 = self.dc2(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        
        
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()
        
        sr_loss = self.spatial_rel(net_output, target)   
        #maybe threshold weights: higher penalty if differences are large / lower if differences are small, assuming certain error. 
        
        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_sr * sr_loss
            """print('result is:', result)
            #print('self.weight_sr is:', self.weight_sr)
            #print('sr_loss is:', sr_loss)
  
            #print('self.weight_ce is:', self.weight_ce)
            print('ce_loss is:', ce_loss)
            
            #print('self.weight_dice is:', self.weight_dice)
            print('dc_loss is:', dc_loss)
            print('dc_loss2 is:', dc_loss2)"""
            
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result


#class DC_and_BCE_loss(nn.Module):
#class GDL_and_CE_loss(nn.Module):
#class DC_and_topk_loss(nn.Module):

        
class InmaDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=False, smooth=1.):
        super(InmaDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
                       
        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = tp + self.smooth                # intersection
        denominator = tp + fp + fn + self.smooth    # union
        volume = tp + fn
        
        num = nominator / (volume + 1e-8) # 1/volume == weight
        den = denominator / (volume + 1e-8)

        # dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                nume = torch.sum(num[1:]) # sum_tensor(num[1:], axes, keepdim=False)
                deno = torch.sum(den[1:]) # sum_tensor(den[1:], axes, keepdim=False)

            else:
                nume = torch.sum(num[:, 1:]) # sum_tensor(num[:, 1:], axes, keepdim=False)
                deno = torch.sum(den[:, 1:]) # sum_tensor(den[:, 1:], axes, keepdim=False)

        tc = nume/deno
        dc = 2*tc/(tc+1)

        return -dc


def weird_to_num(tensor_, replace=0):
    return_tensor = tensor_.clone()
    return_tensor[torch.isnan(tensor_)] = replace
    return_tensor[torch.isinf(tensor_)] = replace
    return return_tensor


class GraphSpatialLoss3D(nn.Module):
    def __init__(self, n_class=8, centroid='threshold'):  # n_class --> 14 labels + background
        
        super(GraphSpatialLoss3D, self).__init__()  
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_class = n_class
        self.centroid_type = centroid

        if self.centroid_type == 'threshold':
            self.threshold = torch.nn.Threshold(1. / float(n_class), 0)
        elif self.centroid_type == 'sigmoid':
            self.sigmoid = torch.nn.Sigmoid()
            self.sig_th = 1. / float(n_class)
        # change path to npy file with priors (priors_one.npy = priors of combined hemispheres; priors.npy = priors of separated hemispheres)
        GT_sr = np.load('/homedtic/malenya/nnUNet/DB_SpatialConstraints/priors_one.npy')
        #GT_sr = np.load('/homedtic/malenya/nnUNet/DB_SpatialConstraints/priors.npy')
        self.GT_sr = GT_sr.transpose()
        
    
    def forward(self, x, y, loss_mask=None):

        # outputs, predictions = x
        # ground truths = y  

        self.image_dimensions = x.shape
        _, _, h, w, d= self.image_dimensions      # h x w x d --> can't be in the init part because x is defined only in the forward. (maybe put a self there, and call x from init??)
        coordinates_map = torch.meshgrid([torch.arange(h), torch.arange(w), torch.arange(d)])            
        self.coords = coordinates_map
        
        sour, targ, dy_gt, dx_gt, dd_gt = self.GT_sr
        
        self.compute_centroids(x)         
        
        # self.compute_colors(data, x)   --> data must be accessed from here. Not done.
        
        sum_err = 0
        # for each prior relation, compute distance/color in the prediction and compare it with the ground truth.
        for rel in range(len(sour)):      
            
            sour0 = int(sour[rel])
            targ0 = int(targ[rel])
            
            dy = self.centroids_y[sour0] - self.centroids_y[targ0] 
            dx = self.centroids_x[sour0] - self.centroids_x[targ0]
            dd = self.centroids_d[sour0] - self.centroids_d[targ0]
            """print('rel is: ', rel, ' --------- source: ', sour0, '------- targ: ', targ0, '-----------')
            print('dy is:', dy)
            print('dx is:', dx)
            print('dd is:', dd)
            #dc = self.colors[:, i[rel]] - self.colors[:, j[rel]]                  
            print('dy_gt is:', dy_gt[rel])
            print('dx_gt is:', dx_gt[rel])
            print('dd_gt is:', dd_gt[rel])"""
        
            diff_y = (dy - dy_gt[rel]) / self.image_dimensions[3]
            diff_x = (dx - dx_gt[rel]) / self.image_dimensions[2]
            diff_d = (dd - dd_gt[rel]) / self.image_dimensions[4]
            #diff_c = (dc - dc_gt[rel])
            
            #print('diff_y is:', diff_y)
            #print('diff_x is:', diff_x)
            #print('diff_d is:', diff_d)


            dy_err = torch.mean(torch.square(weird_to_num(diff_y, replace=0.0)))
            dx_err = torch.mean(torch.square(weird_to_num(diff_x, replace=0.0)))
            dd_err = torch.mean(torch.square(weird_to_num(diff_d, replace=0.0)))
            #dc_err = torch.mean(torch.square(weird_to_num(diff_c, replace=0.0)))
            
            #print('dy_err is:', dy_err)
            #print('dx_err is:', dx_err)
            #print('dd_err is:', dd_err)
            
            sum_err += dy_err + dx_err + dd_err # + dc_err

        return sum_err

    def compute_centroids(self, batch):

        coords_y, coords_x, coords_d = self.coords       #coords_y dimensions [128, 128, 128]

        #coords_y.expand(batch.shape)
        #coords_x.expand(batch.shape)
        #coords_d.expand(batch.shape)
        coords_y = torch.unsqueeze(coords_y, 0).unsqueeze(0)   
        coords_x = torch.unsqueeze(coords_x, 0).unsqueeze(0)
        coords_d = torch.unsqueeze(coords_d, 0).unsqueeze(0)
        
        #print(coords_y.shape)
        #print(coords_x.shape)
        #print(coords_d.shape)
               
        batch.to(self.device)
		
        #print('coords_y shape:', coords_y.shape)    #coords_y, coords_z, coords_d dimensions [1, 1, 128, 128, 128]
        #print('batch shape:', batch.shape)          #batch dimensions [2, 15, 128, 128, 128]

        if self.centroid_type == 'orig':
            self.centroids_y = torch.sum(batch * coords_y.to(self.device), dim=[0,2,3,4]) / torch.sum(batch, dim=[0,2,3,4])  
            self.centroids_x = torch.sum(batch * coords_x.to(self.device), dim=[0,2,3,4]) / torch.sum(batch, dim=[0,2,3,4])  
            self.centroids_d = torch.sum(batch * coords_d.to(self.device), dim=[0,2,3,4]) / torch.sum(batch, dim=[0,2,3,4])  

        elif self.centroid_type == 'threshold':            
            batch_th = self.threshold(batch).to(self.device)
            
            self.centroids_y = torch.sum(batch_th * coords_y.to(self.device), dim=[0,2,3,4]) / torch.sum(batch_th, dim=[0,2,3,4])  
            self.centroids_x = torch.sum(batch_th * coords_x.to(self.device), dim=[0,2,3,4]) / torch.sum(batch_th, dim=[0,2,3,4]) 
            self.centroids_d = torch.sum(batch_th * coords_d.to(self.device), dim=[0,2,3,4]) / torch.sum(batch_th, dim=[0,2,3,4])  

        else:
            raise Exception("Sorry, unknown centroid calculation")

    def get_centroids(self):
        return self.centroids_y, self.centroids_x, self.centroids_d
    
    
    """def compute_colors(self, batch, data):       
                    label_id = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]) 
                    color = torch.zeros(14)
                    m = 0
                    for k in label_id:    
                        color[m] = torch.sum(batch * torch.where(self.labels==k, data, 0).to(self.device), dim=[2,3,4]) / torch.sum(batch, dim=[2,3,4])    # Batch size x N_labels   
                        m += 1
                    
                    self.colors = color"""
