import torch
import torch.nn as nn
import math

loss_l1 = nn.SmoothL1Loss(beta=1.0)
loss_bce_keepdim = nn.BCELoss(reduction='none')

def sdfLoss(pred, label):
    loss = torch.zeros_like(label, dtype = label.dtype, device = label.device)
    middle_point = label/2.0
    middle_point_abs = torch.abs(middle_point)
    shift_difference_abs = torch.abs(pred-middle_point)
    mask = shift_difference_abs > middle_point_abs
    loss[mask] = (shift_difference_abs - middle_point_abs)[mask]
    return loss


def welsch_loss(x):
    return 0.3*(1-torch.exp(-(x/0.3)**2))
    # return x**2


def shift_bce_loss(x):
    occ = torch.sigmoid(x*50.0)
    label = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)*0.5
    loss = math.log(0.5) + loss_bce_keepdim(occ,label)

    return loss


def smooth_sdfLoss(pred, label):
    loss = torch.zeros_like(label, dtype = label.dtype, device = label.device)

    middle_result = pred*label

    sign_mask = middle_result < 0 

    truncated_mask = middle_result > label**2

    loss[sign_mask] = welsch_loss(pred[sign_mask])
    loss[truncated_mask] = welsch_loss((pred-label)[truncated_mask])
    
    return loss


def numerical_ekional_(field, mlp, data, t, step, static=False):

    input_x_dx = data + torch.tensor([step,0,0], dtype=data.dtype, device=data.device)
    input_x_mdx = data + torch.tensor([-step,0,0], dtype=data.dtype, device=data.device)
    input_y_dy = data + torch.tensor([0,step,0], dtype=data.dtype, device=data.device)
    input_y_mdy = data + torch.tensor([0,-step,0], dtype=data.dtype, device=data.device)
    input_z_dz = data + torch.tensor([0,0,step], dtype=data.dtype, device=data.device)
    input_z_mdz = data + torch.tensor([0,0,-step], dtype=data.dtype, device=data.device)

    input_d = torch.cat([input_x_dx, input_y_dy, input_z_dz], dim=0)
    input_md = torch.cat([input_x_mdx, input_y_mdy, input_z_mdz], dim=0)

    input_total = torch.cat([input_d, input_md], dim=0)
    input_t = t.repeat(6,1)
    
    feature_vectors = field.get_features(input_total)

    if static:
        pred_total, _, _ = mlp(feature_vectors.float(), input_t.long())
    else:
        _, pred_total, _= mlp(feature_vectors.float(), input_t.long())

    # pred_total = mlp(feature_vectors.float())
   
    pred_total = pred_total.reshape(2,-1)

    normals = (((pred_total[0] - pred_total[1])/(2*step)).reshape(-1,data.shape[0])).T

    eikonal_loss = torch.abs(normals.norm(2,dim=-1) - 1.0)

    return eikonal_loss


def numerical_ekional(field, mlp, data, t, step, static=False):

    input_x_dx = data + torch.tensor([step,0,0], dtype=data.dtype, device=data.device)
    input_x_mdx = data + torch.tensor([-step,0,0], dtype=data.dtype, device=data.device)
    input_y_dy = data + torch.tensor([0,step,0], dtype=data.dtype, device=data.device)
    input_y_mdy = data + torch.tensor([0,-step,0], dtype=data.dtype, device=data.device)
    input_z_dz = data + torch.tensor([0,0,step], dtype=data.dtype, device=data.device)
    input_z_mdz = data + torch.tensor([0,0,-step], dtype=data.dtype, device=data.device)

    input_d = torch.cat([input_x_dx, input_y_dy, input_z_dz], dim=0)
    input_md = torch.cat([input_x_mdx, input_y_mdy, input_z_mdz], dim=0)

    input_total = torch.cat([input_d, input_md], dim=0)

    flags = torch.ones(input_total.shape[0], 1, device=input_total.device)
    valid_mask = field.get_valid_mask(input_total)
    flags[~valid_mask] = 0

    mask = (flags.reshape(6,-1).T.sum(1)) > 5.0

    valid_point = data[mask]
    t = t[mask]

    return numerical_ekional_(field, mlp, valid_point, t, step, static=False)



def numerical_ekional_3d(field, mlp, data, step):

    input_x_dx = data + torch.tensor([step,0,0], dtype=data.dtype, device=data.device)
    input_x_mdx = data + torch.tensor([-step,0,0], dtype=data.dtype, device=data.device)
    input_y_dy = data + torch.tensor([0,step,0], dtype=data.dtype, device=data.device)
    input_y_mdy = data + torch.tensor([0,-step,0], dtype=data.dtype, device=data.device)
    input_z_dz = data + torch.tensor([0,0,step], dtype=data.dtype, device=data.device)
    input_z_mdz = data + torch.tensor([0,0,-step], dtype=data.dtype, device=data.device)

    input_d = torch.cat([input_x_dx, input_y_dy, input_z_dz], dim=0)
    input_md = torch.cat([input_x_mdx, input_y_mdy, input_z_mdz], dim=0)

    input_total = torch.cat([input_d, input_md], dim=0)
    
    feature_vectors = field.get_features(input_total)

    pred_total = mlp(feature_vectors.float())
   
    pred_total = pred_total.reshape(2,-1)

    normals = (((pred_total[0] - pred_total[1])/(2*step)).reshape(-1,data.shape[0])).T

    eikonal_loss = torch.abs(normals.norm(2,dim=-1) - 1.0)

    return eikonal_loss



def double_numerical_normals(field, mlp, data, t, step):

    input_x_dx = data + torch.tensor([step,0,0], dtype=data.dtype, device=data.device)
    input_x_mdx = data + torch.tensor([-step,0,0], dtype=data.dtype, device=data.device)
    input_y_dy = data + torch.tensor([0,step,0], dtype=data.dtype, device=data.device)
    input_y_mdy = data + torch.tensor([0,-step,0], dtype=data.dtype, device=data.device)
    input_z_dz = data + torch.tensor([0,0,step], dtype=data.dtype, device=data.device)
    input_z_mdz = data + torch.tensor([0,0,-step], dtype=data.dtype, device=data.device)

    input_d = torch.cat([input_x_dx, input_y_dy, input_z_dz], dim=0)
    input_md = torch.cat([input_x_mdx, input_y_mdy, input_z_mdz], dim=0)

    input_total = torch.cat([input_d, input_md], dim=0)
    input_t = t.repeat(6,1)
    
    # feature_vectors = field.get_features(input_total.float())

    # static_pred, dynamic_pred, _ = mlp(feature_vectors.float(), input_t.long())

    feature_vectors = field.get_features(input_total)

    static_pred, dynamic_pred, _ = mlp(feature_vectors, input_t.long())
  
    difference_pred = (dynamic_pred-static_pred).reshape(2,-1)
    static_pred = static_pred.reshape(2,-1)
    dynamic_pred = dynamic_pred.reshape(2,-1)

    static_normals = (((static_pred[0] - static_pred[1])/(2*step)).reshape(-1,data.shape[0])).T
    dynamic_normals = (((dynamic_pred[0] - dynamic_pred[1])/(2*step)).reshape(-1,data.shape[0])).T
    difference_normals = (((difference_pred[0] - difference_pred[1])/(2*step)).reshape(-1,data.shape[0])).T

    return static_normals, dynamic_normals, difference_normals
