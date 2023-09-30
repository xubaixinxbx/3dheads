import torch.nn as nn
import numpy as np

from utils import rend_util
from model.embedder import *
from model.density import LaplaceDensity
from model.ray_sampler import ErrorBoundSampler

class DeformNetwork(nn.Module):
    def __init__(self, shape_code_dim, dims, d_in=3, d_out=3, \
                multires=0, weight_norm=True, deform_feature_dim=128,
                ):
        super().__init__()
        d_out = d_out + deform_feature_dim
        dims = [d_in + shape_code_dim] + dims + [d_out]
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch + shape_code_dim

        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, shape_latent_code):
        if self.embed_fn is not None:
            x = self.embed_fn(x)
        if x.shape[0] != shape_latent_code.shape[0] and shape_latent_code.shape[0] == 1:
            shape_latent_code = shape_latent_code.repeat(x.shape[0],1)
        assert shape_latent_code.shape[0] == x.shape[0], print('in deform net, shape_code.dim != x.dim')
        x =  torch.cat([x, shape_latent_code], dim=-1)
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        return x
    
    def update_embed_fn(self, alpha):
        if self.embed_fn is not None:
            self.embed_fn.update_alpha(alpha)

class DisplacementNetwork(nn.Module):
    def __init__(self, in_feature_dim, dims, d_in=3, d_out=1, multires=0, weight_norm=True,\
                skip_in=[], displace_feature_dim=128,\
                ):
        super().__init__()
        dims = [d_in + in_feature_dim] + dims + [d_out + displace_feature_dim]
        self.skip_in = skip_in
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch + in_feature_dim
        else:
            input_ch = d_in

        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                in_dim = dims[l] + input_ch
            else:
                in_dim = dims[l]
            lin = nn.Linear(in_dim, dims[l+1])
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x=None, feature=None):
        if x is not None:
            if self.embed_fn is not None:
                x = self.embed_fn(x)
            if x.shape[0] != feature.shape[0] and feature.shape[0] == 1:
                feature = feature.repeat(x.shape[0],1)
            assert feature.shape[0] == x.shape[0], print('in displacement net, shape_code.dim != x.dim')
            input_x =  torch.cat([x, feature], dim=-1)
        else:
            input_x = feature
            
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skip_in:
                input_x = torch.cat([input_x, x],-1)
            input_x = lin(input_x)
            if l < self.num_layers - 2:
                input_x = self.relu(input_x)
        return input_x
    
    def update_embed_fn(self, alpha):
        if self.embed_fn is not None:
            self.embed_fn.update_alpha(alpha)

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            temp_feature_dim,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            input_feat_dim=0,
    ):
        super().__init__()
        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in+input_feat_dim] + dims + [d_out + temp_feature_dim]
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] += input_ch - d_in
        self.num_layers = len(dims)
        self.skip_in = skip_in
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, shape_latent_code=None):
        if self.embed_fn is not None:
            input = self.embed_fn(input)
        if shape_latent_code is not None:
            if input.shape[0] != shape_latent_code.shape[0] and shape_latent_code.shape[0] == 1:
                shape_latent_code = shape_latent_code.repeat(input.shape[0],1)
            input = torch.cat([input, shape_latent_code], 1)
        
        x = input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)
        return x

    def update_embed_fn(self, alpha):
        if self.embed_fn:
            self.embed_fn.update_alpha(alpha)

class DeformTempGeoNetwork(nn.Module):
    def __init__(self,conf,**kwargs):
        super().__init__()
        self.deform_conf = conf.get_config('deform_network')
        self.implicit_conf = conf.get_config('implicit_network')
        self.sphere_scale = self.implicit_conf.get_float('sphere_scale')
        self.temp_feature_dim = kwargs['temp_feature_dim']
        shape_code_dim = kwargs['shape_code_dim']
        self.white_bkgd = kwargs['white_bkgd']
        self.deform_feature_dim = self.deform_conf.get_int('deform_feature_dim',default=0)
        self.scene_bounding_sphere = kwargs['scene_bounding_sphere']
        self.sdf_bounding_sphere = 0.0 if self.white_bkgd else self.scene_bounding_sphere
        self.plot_template = False

        self.deform_net = DeformNetwork(shape_code_dim=shape_code_dim,\
                                        **self.deform_conf)

        self.implicit_net = ImplicitNetwork(temp_feature_dim=self.temp_feature_dim,\
                                sdf_bounding_sphere=self.sdf_bounding_sphere,\
                                **self.implicit_conf)
    
    def set_plot_template(self, plot_template=False):
        self.plot_template = plot_template
    
    def update_embed_fn(self, alpha):
        self.deform_net.update_embed_fn(alpha)
        self.implicit_net.update_embed_fn(alpha)
        
    def forward(self, x, shape_code):
        delta_x_with_feature = self.deform_net(x, shape_code)
        delta_x = delta_x_with_feature[:,:3]
        deform_feature = delta_x_with_feature[:,3:]
            
        if self.plot_template:
            ref_points = x
        else:
            ref_points = x + delta_x

        sdf_with_feature = self.implicit_net(ref_points)
        template_feature = sdf_with_feature[:,1:]

        geometry_feature = torch.cat([deform_feature, template_feature], -1)
        sdf_with_feature = torch.cat([sdf_with_feature[:,:1], geometry_feature], -1) #new feature fed into render-net

        if self.plot_template:
            pass
        
        return sdf_with_feature, delta_x
    
    def gradient(self, x, shape_code):
        x.requires_grad_(True)
        y, delta_x = self.forward(x, shape_code)
        y = y[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients
    
    def get_outputs(self, x, shape_code):
        if shape_code.shape[0] == 1:
            shape_code = shape_code.repeat(x.shape[0],1)
        x.requires_grad_(True)
        output, delta_x = self.forward(x, shape_code)
        sdf = output[:,:1]

        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        feature_vectors = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        
        grad_deform = None
        if self.training:
            deform_output = torch.ones_like(delta_x, requires_grad=False, device=delta_x.device)
            grad_deform = torch.autograd.grad(delta_x, x, grad_outputs=deform_output, retain_graph=True, create_graph=True)[0]

        sdf_gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, [delta_x, grad_deform], feature_vectors, sdf_gradients
    
    def get_sdf_vals(self, x, shape_code):
        if shape_code.shape[0] == 1:
            shape_code = shape_code.repeat(x.shape[0],1)

        sdf_with_feature, delta_x = self.forward(x, shape_code)
        sdf = sdf_with_feature[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf

class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            multires=0,
            skip_in=(),
            acti_flag=True
    ):
        super().__init__()
        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]
        self.skip_in = skip_in
        self.embedview_fn = None
        self.embed_fn = None
        self.acti_flag = acti_flag

        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            if self.mode == 'idr':
                dims[0] += (input_ch -3)
            else:
                dims[0] += (input_ch - d_in)
        if self.mode == 'idr' and multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] += (input_ch -3)

        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                in_dim = dims[l] + dims[0]
            else:
                in_dim = dims[l]
            lin = nn.Linear(in_dim, dims[l + 1])
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, normals=None, view_dirs=None, feature_vectors=None):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)
        if self.embed_fn is not None:
            points = self.embed_fn(points)
        
        rendering_input = points
        if self.mode == 'idr':
            if normals is not None:
                rendering_input = torch.cat([rendering_input, normals], dim=-1)
            if view_dirs is not None:
                rendering_input = torch.cat([rendering_input, view_dirs], dim=-1)
            if feature_vectors is not None:
                rendering_input = torch.cat([rendering_input, feature_vectors], dim=-1)

        elif self.mode == 'nerf':
            if feature_vectors is not None:
                rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
            else:
                rendering_input = view_dirs
        x = rendering_input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l + 1 in self.skip_in:
                x = torch.cat([x, rendering_input], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        if self.acti_flag:
            x = self.sigmoid(x)
        return x

    def update_embed_fn(self, alpha):
        if self.embed_fn is not None:
            self.embed_fn.update_alpha(alpha)
        if self.embedview_fn is not None:
            self.embedview_fn.update_alpha(alpha)

class VolSDFNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
        self.white_bkgd = conf.get_bool('white_bkgd', default=False)
        self.bg_color = torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().cuda()
        self.num_id = conf.get_int('num_identity')
        self.dim_id_shape = conf.get_int('dim_identity_shape')
        self.dim_id_color = conf.get_int('dim_identity_color')
        self.deform_feature_dim = conf.get_int('deformTempGeo_network.deform_network.deform_feature_dim')
        self.temp_feature_dim = conf.get_int('template_feature_dim')

        self.st1_flag = conf.get_bool('st1_flag', default=False)
        self.deform_temp_geo_conf = conf.get_config('deformTempGeo_network')
        self.displacement_conf = conf.get_config('displacement_network')
        self.sdf_bounding_sphere = 0.0 if self.white_bkgd else self.scene_bounding_sphere
        self.sphere_scale = self.deform_temp_geo_conf.get_float('implicit_network.sphere_scale')

        self.plot_template = False
        self.plot_displacement = True
        self.use_tv_loss = conf.get_bool('use_tv_loss', default=True)
        
        self.deform_temp_geo_network = DeformTempGeoNetwork(shape_code_dim=self.dim_id_shape,\
                                                temp_feature_dim=self.temp_feature_dim,\
                                                white_bkgd=self.white_bkgd,\
                                                scene_bounding_sphere=self.scene_bounding_sphere,\
                                                conf=self.deform_temp_geo_conf)
        if not self.st1_flag:
            self.dis_feature_dim = self.displacement_conf.get_int('displace_feature_dim',default=0.)
            self.displacement_network = DisplacementNetwork(in_feature_dim=self.temp_feature_dim+self.deform_feature_dim,\
                                                            **self.displacement_conf)
            
            color_code_dim=self.dis_feature_dim+self.deform_feature_dim+self.temp_feature_dim+self.dim_id_color
            self.res_rendering_network = RenderingNetwork(feature_vector_size=color_code_dim,\
                                                **conf.get_config('rendering_network'))
        else:
            self.rendering_network = RenderingNetwork(feature_vector_size=self.deform_feature_dim+self.temp_feature_dim+self.dim_id_color,\
                                                  **conf.get_config('rendering_network'))

        pe_alpha = conf.get_float('pe_alpha', default=1000.0)
        self.update_embed_fn(pe_alpha)

        self.density = LaplaceDensity(**conf.get_config('density'))
        ray_sample_conf = conf.get_config('ray_sampler')
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **ray_sample_conf)
        self.id_shape_embedding = nn.Embedding(num_embeddings=self.num_id, embedding_dim=self.dim_id_shape)
        self.id_color_embedding = nn.Embedding(num_embeddings=self.num_id, embedding_dim=self.dim_id_color)
        nn.init.normal_(self.id_shape_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.id_color_embedding.weight, mean=0, std=0.01)
    
    def forward(self, input, embeddings=None):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        id = input["id"]
        if embeddings is None:
            embeddings = self.get_embedding(shape_idx=id, color_idx=id)
        shape_latent_code = embeddings['shape_code']
        color_latent_code = embeddings['color_code']

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape
        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)
        z_vals, z_samples_eik, z_samples = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self, shape_latent_code)

        N_samples = z_vals.shape[1]
        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        shape_latent_code = shape_latent_code[:,None,None,:].permute(1,2,0,3).reshape(-1, batch_size, self.dim_id_shape)
        shape_latent_code = shape_latent_code.repeat(num_pixels*N_samples, 1, 1)
        shape_latent_code_flat = shape_latent_code.permute(1,0,2).reshape(-1, self.dim_id_shape)

        sdf, grad_deform, feature_vectors, gradients = \
            self.deform_temp_geo_network.get_outputs(points_flat,shape_latent_code_flat)


        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)

        color_latent_code = color_latent_code[:,None,None,:].permute(1,2,0,3).reshape(-1, batch_size, self.dim_id_color)
        color_latent_code = color_latent_code.repeat(num_pixels*N_samples, 1, 1)
        color_latent_code_flat = color_latent_code.permute(1,0,2).reshape(-1, self.dim_id_color)
        
        if not self.st1_flag:
            if self.plot_displacement:
                if self.displacement_conf.get_int('multires') > 0: 
                    delta_sdf_with_feature = self.displacement_network(x=points_flat,feature=feature_vectors)
                else:
                    delta_sdf_with_feature = self.displacement_network(feature=feature_vectors)
                feature_vectors = torch.cat([feature_vectors, delta_sdf_with_feature[:,1:]], -1)
                delta_sdf = delta_sdf_with_feature[:, :1]
                final_sdf = sdf + delta_sdf
                final_gradients = self.get_gradient(points_flat, final_sdf)
            else:
                template_feature = feature_vectors[:,:self.temp_feature_dim]
                feature_vectors = torch.cat([feature_vectors, template_feature], -1)
                final_sdf = sdf
                final_gradients = gradients
            weights = self.volume_rendering(z_vals, final_sdf)
            
            feature_vectors_with_color = torch.cat([feature_vectors, color_latent_code_flat], -1)
            rgb_flat = self.res_rendering_network(points_flat, final_gradients, dirs_flat, feature_vectors_with_color)        
            rgb = rgb_flat.reshape(-1, N_samples, 3)
            rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        else:
            weights = self.volume_rendering(z_vals, sdf)
            feature_vectors_with_color = torch.cat([feature_vectors, color_latent_code_flat], dim=-1)
            rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors_with_color)
            rgb = rgb_flat.reshape(-1, N_samples, 3)
            rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
            final_gradients = gradients
        
        # white background assumption
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)

        output = {
            'rgb_values': rgb_values,
        }

        if self.training and not self.st1_flag and self.plot_displacement:
            if self.use_tv_loss:
                delta_sdf_grad = self.get_gradient(points_flat, delta_sdf)
                output['delta_sdf_grad'] = delta_sdf_grad
            output['delta_sdf'] = delta_sdf

        if self.training:
            output['grad_theta'] = gradients
            output['delta_x'] = grad_deform[0]
            output['grad_deform'] = grad_deform[1]
            output['code'] = embeddings

        if not self.training:
            gradients = final_gradients.detach()
            normals = gradients / gradients.norm(2, -1, keepdim=True)
            
            pts_norm = points_flat.norm(2, -1, keepdim=True)
            inside_sphere = (pts_norm < 1).float().detach() # make sure your model is inside unit sphere.
            normals = normals * inside_sphere

            normals = normals.reshape(-1, N_samples, 3)
            normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)

            output['normal_map'] = normal_map
            # output['z_samples_eik'] = z_samples

        return output
    
    def get_sdf_vals(self, x, shape_code):
        if shape_code.shape[0] == 1:
            shape_code = shape_code.repeat(x.shape[0],1)
        sdf_with_feature, _ = self.deform_temp_geo_network(x, shape_code)
        if not self.st1_flag and self.plot_displacement:
            if self.displacement_conf.get_int('multires') > 0: 
                delta_sdf_with_feature = self.displacement_network(x=x, feature=sdf_with_feature[:,1:])
            else:
                delta_sdf_with_feature = self.displacement_network(sdf_with_feature[:,1:])
            sdf = sdf_with_feature[:,:1] + delta_sdf_with_feature[:,:1]
        else:
            sdf = sdf_with_feature[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf


    def get_gradient(self, x, y):
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients
    
    def update_embed_fn(self, alpha):
        self.deform_temp_geo_network.update_embed_fn(alpha)
        if not self.st1_flag:
            self.displacement_network.update_embed_fn(alpha)
            self.res_rendering_network.update_embed_fn(alpha)
        else:
            self.rendering_network.update_embed_fn(alpha)

    
    def set_plot_template(self, plot_template=False):
        self.plot_template=plot_template
        self.deform_temp_geo_network.set_plot_template(plot_template)

    def set_plot_displacement(self, plot_displacement=True):
        self.plot_displacement = plot_displacement
        
    def get_alpha_from_sdf(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        assert torch.isnan(dists).sum() == 0, print('dists met a nan')
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        return  alpha

    def volume_rendering(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        assert torch.isnan(dists).sum() == 0, print('dists met a nan')
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        assert torch.isnan(alpha).sum() == 0, print('alpha met a nan')
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        assert torch.isnan(transmittance).sum() == 0, print('transmittance met a nan')
        weights = alpha * transmittance # probability of the ray hits something here

        return weights

    def get_embedding(self, shape_idx=[-1], color_idx=[-1]):
        shape_embedding = self.id_shape_embedding
        color_embedding = self.id_color_embedding
        all_ids = torch.arange(0, self.num_id).cuda()
        if shape_idx[0] == -1:
            shape_code = shape_embedding(all_ids)
        else:
            shape_code = shape_embedding(shape_idx)
        if color_idx[0] == -1:
            color_code = color_embedding(all_ids)
        else:
            color_code = color_embedding(color_idx)
        embeddings = {
                'shape_code' : shape_code,
                'color_code' : color_code,
            }
        return embeddings
    
    def set_one_embedding(self, code_st=None, code_ed=None, alpha=0.):
        return code_st * (1.0-alpha) + alpha * code_ed

    def init_embedding_for_fine_tune(self, num=1):
        self.fine_tune_flag = True
        self.fine_tune_shape_code = nn.Embedding(num_embeddings=num, embedding_dim=self.dim_id_shape).cuda()
        self.fine_tune_color_code = nn.Embedding(num_embeddings=num, embedding_dim=self.dim_id_color).cuda()
        nn.init.normal_(self.fine_tune_shape_code.weight, mean=0, std=0.01)
        nn.init.normal_(self.fine_tune_color_code.weight, mean=0, std=0.01)