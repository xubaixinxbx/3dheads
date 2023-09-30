import torch
from torch import nn
import utils.general as utils

class VolSDFLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.eikonal_weight = kwargs['eikonal_weight']
        self.rgb_loss = utils.get_class(kwargs['rgb_loss'])(reduction='mean')
        
        self.deform_weight = kwargs.get('deform_weight', 0)
        self.deform_grad_weight = kwargs.get('deform_grad_weight', 0)
        
        self.disp_weight = kwargs.get('disp_weight', 0)
        self.disp_loss = utils.get_class(kwargs['disp_loss'])(reduction='mean')
        self.disp_grad_weight = kwargs.get('disp_grad_weight', 0)
        self.disp_grad_loss = utils.get_class(kwargs['disp_grad_loss'])(reduction='mean')
        
        self.shape_code_weight = kwargs.get('shape_code_weight',0)
        self.color_code_weight = kwargs.get('color_code_weight',0)
        self.latent_code_loss = utils.get_class(kwargs['latent_code_loss'])(reduction='mean')


    def get_rgb_loss(self,rgb_values, rgb_gt):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss
    
    def get_disp_loss(self, delta_s):
        correct_loss = self.disp_loss(delta_s, torch.zeros_like(delta_s).cuda().float())
        return correct_loss
    
    def get_disp_grad_loss(self, grad_delta_s):
        disp_grad_loss = self.disp_grad_loss(grad_delta_s, torch.zeros_like(grad_delta_s).cuda().float())
        return disp_grad_loss
    
    def get_latent_code_loss(self, code):
        latent_code_loss = self.latent_code_loss(code, torch.zeros_like(code).cuda().float())
        return latent_code_loss


    def forward(self, model_outputs, ground_truth):
        if 'rgb' in ground_truth:
            rgb_gt = ground_truth['rgb'].cuda()
            rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        else:
            rgb_loss = torch.tensor(0.0).cuda().float()

        # eik loss
        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        else:
            eikonal_loss = torch.tensor(0.0).cuda().float()
        
        # deform loss
        if 'grad_deform' in model_outputs:
            deform_grad_loss = model_outputs['grad_deform'].norm(2, dim=-1).mean()
        else:
            deform_grad_loss = torch.tensor(0.0).cuda().float()
        if 'delta_x' in model_outputs:
            deform_loss = model_outputs['delta_x'].norm(2, dim=-1).mean()
        else:
            deform_loss = torch.tensor(0.0).cuda().float()
        
        # displacement loss
        if 'delta_sdf' in model_outputs and model_outputs['delta_sdf'] is not None:
            delta_sdf = model_outputs['delta_sdf']
            delta_sdf_loss = self.get_disp_loss(delta_sdf)
        else:
            delta_sdf_loss = torch.tensor(0.0).cuda().float()
        if 'delta_sdf_grad' in model_outputs and model_outputs['delta_sdf_grad'] is not None:
            delta_sdf_grad = model_outputs['delta_sdf_grad']
            delta_sdf_grad_loss = self.get_disp_grad_loss(delta_sdf_grad)
        else:
            delta_sdf_grad_loss = torch.tensor(0.0).cuda().float()
        
        # latent code loss
        if 'code' in model_outputs:
            if 'shape_code' in model_outputs['code']:
                shape_code_loss = self.get_latent_code_loss(model_outputs['code']['shape_code'])
            else:
                shape_code_loss = torch.tensor(0.0).cuda().float()
            if 'color_code' in model_outputs['code']:
                color_code_loss = self.get_latent_code_loss(model_outputs['code']['color_code'])
            else:
                color_code_loss = torch.tensor(0.0).cuda().float()
        else:
            shape_code_loss = torch.tensor(0.0).cuda().float()
            color_code_loss = torch.tensor(0.0).cuda().float()
        
        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss + \
                self.disp_weight * delta_sdf_loss + \
                    self.disp_grad_weight * delta_sdf_grad_loss + \
                        self.shape_code_weight * shape_code_loss + \
                            self.color_code_weight * color_code_loss + \
                                self.deform_grad_weight * deform_grad_loss + \
                                    self.deform_weight * deform_loss

        output = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'disp_loss': delta_sdf_loss,
            'disp_grad_loss': delta_sdf_grad_loss,
            'shape_code_loss': shape_code_loss,
            'color_code_loss': color_code_loss,
            'deform_grad_loss': deform_grad_loss,
            'deform_loss': deform_loss
        }

        return output
