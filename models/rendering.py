import torch
import torch.nn as nn
from models.networks import GeometryNet, AppearanceNet, SDensity


class NeuSRenderer(nn.Module):
    """
    Neural Renderer according to NeuS 
    https://arxiv.org/abs/2106.10689
    """
    def __init__(self,
                 geometry_net,
                 appearance_net,
                 s_density,
                 num_samples,
                 perturb,
                 white_bkgd):
        """
        Args:
            geometry_net (nn.Module): geoemtry network
            appearance_net (nn.Module): appearance network
            s_density (nn.Module): single parameter s_density network
            num_samples (int): number of points to sample along each ray
            perturb (bool): if True, use randomized stratified sampling
            white_bkgd (bool): if True, assume white background
        """
        super().__init__()

        self.geometry_net = geometry_net
        self.appearance_net = appearance_net
        self.s_density = s_density
        self.num_samples = num_samples
        self.perturb = perturb
        self.white_bkgd = white_bkgd

    def forward(self, rays_o, rays_d):
        """
        Given a camera ray, render its color

        Inputs:
            rays_o [batch_size, 3]: ray origins
            rays_d [batch_size, 3]: ray directions
        """
        near, far = self.near_far(rays_o, rays_d)
        t_vals, points = self.sample_points(rays_o, rays_d, near, far) # [batch_size, num_samples+1],
                                                                       # [batch_size, num_samples, 3]
        rays_d = rays_d[..., None, :].expand_as(points) # [batch_size, num_samples, 3]

        sdf, geometric_feature = self.geometry_net(points) # [batch_size, num_samples],
                                                           # [batch_size, num_samples, feature_dim]
        sdf_grad = self.geometry_net.gradient(sdf, points) # [batch_size, num_samples, 3]
        
        rgb = self.appearance_net(points, rays_d, sdf_grad, geometric_feature) # [batch_size, num_samples, 3]
        
        dists = t_vals[..., 1:] - t_vals[..., :-1] # [batch_size, num_samples]
        grad_proj = torch.einsum("ijk, ijk -> ij", sdf_grad, rays_d) # [batch_size, num_samples]

        prev_sdf = sdf - 0.5 * grad_proj * dists # [batch_size, num_samples]
        next_sdf = sdf + 0.5 * grad_proj * dists # [batch_size, num_samples]

        inv_s = self.s_density().clip(1e-7, 1e7)
        prev_cdf = torch.sigmoid(prev_sdf * inv_s) # [batch_size, num_samples]
        next_cdf = torch.sigmoid(next_sdf * inv_s) # [batch_size, num_samples]
        alpha = (prev_cdf - next_cdf) / (prev_cdf + 1e-7) # [batch_size, num_samples]
        alpha = alpha.clip(0.0, 1.0)

        transparency = torch.cat([
                                torch.ones_like(alpha[:, :1]),
                                torch.cumprod(1 - alpha[:, :-1] + 1e-7, dim=-1)
                            ], dim=-1) # [batch_size, num_samples]
        weight = alpha * transparency # [batch_size, num_samples]
        weight_sum = weight.sum(dim=-1, keepdim=True) # [batch_size, 1]

        color = torch.einsum("bnc, bn -> bc", rgb, weight) # [batch_size, 3]
        if self.white_bkgd:
            color = color + (1 - weight_sum)
        
        return {
            "color": color,
            "sdf_grad": sdf_grad,
            "weight_sum": weight_sum
        }

    def near_far(self, rays_o, rays_d):
        """
        For each ray, find the nearest point on the ray to the origin (with depth mid).
        Then define the near and far bounds as mid-1 and mid+1.
        https://github.com/Totoro97/NeuS/issues/11
        
        Inputs:
            rays_o [batch_size, 3]: ray origins
            rays_d [batch_size, 3]: ray directions
        Outputs:
            near [batch_size, 1]: near bound of the rays
            far [batch_size, 1]: far bound of the rays
        """
        mid = -torch.einsum("ij, ij -> i", rays_o, rays_d)
        near, far = mid - 1.0, mid + 1.0

        return near[..., None], far[..., None]

    def sample_points(self, rays_o, rays_d, near, far):
        """
        Sample points along the ray

        Inputs:
            rays_o [batch_size, 3]: ray origins
            rays_d [batch_size, 3]: ray directions
            near [batch_size, 1]: near bound of the rays
            far [batch_size, 1]: far bound of the rays
        Outputs:
            t_vals [batch_size, num_samples+1]: sampled t values
            points [batch_size, num_samples, 3]: coordinate of the sampled points
        """
        t_vals = torch.linspace(0., 1., self.num_samples+1,
                                device=rays_o.device) # [num_samples+1]
        t_vals = near + (far - near) * t_vals[None, ...] # [batch_size, num_samples+1]
        t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:]) # [batch_size, num_samples]
        if self.perturb:
            rand = torch.rand_like(t_mids) - 0.5
            t_mids = t_mids + rand * 2.0/self.num_samples
        
        points = rays_o[..., None, :] + rays_d[..., None, :] * t_mids[..., None] # [batch_size, num_samples, 3]

        return t_vals, points


def build_neus(conf):
    """
    Build NeuS Renderer from config
    """
    geometry_net = GeometryNet(**conf["geometry_net"])
    appearance_net = AppearanceNet(**conf["appearance_net"])
    s_density = SDensity(**conf["s_density"])

    model = NeuSRenderer(geometry_net,
                         appearance_net,
                         s_density,
                         **conf["neus"])
    return model
