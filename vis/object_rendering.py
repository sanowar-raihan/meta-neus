import math
import imageio
import torch


def create_posemat(radius, theta, phi):
    """
    3d transformations to create pose matrix from radius, theta and phi
    """
    trans_t = lambda t : torch.as_tensor([
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, t],
                            [0, 0, 0, 1],
                            ], dtype=torch.float)

    rot_theta = lambda theta : torch.as_tensor([
                                    [torch.cos(theta), 0, -torch.sin(theta), 0],
                                    [0, 1, 0, 0],
                                    [torch.sin(theta), 0, torch.cos(theta), 0],
                                    [0, 0, 0, 1],
                                    ], dtype=torch.float)

    rot_phi = lambda phi : torch.as_tensor([
                                [1, 0, 0, 0],
                                [0, torch.cos(phi), -torch.sin(phi), 0],
                                [0, torch.sin(phi), torch.cos(phi), 0],
                                [0, 0, 0, 1],
                                ], dtype=torch.float)
        
    pose = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
    pose = torch.as_tensor([[-1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]
                            ], dtype=torch.float) @ pose
    
    return pose


def get_360_poses(radius=2, phi=-math.pi/5, num_poses=120):
    """
    Create spherical camera poses for 360-degree view around the object
    """
    radius = torch.as_tensor(radius, dtype=torch.float)
    phi = torch.as_tensor(phi, dtype=torch.float)

    all_poses = []
    for theta in torch.linspace(0, 2*math.pi, num_poses+1)[:-1]:
        all_poses.append(create_posemat(radius, theta, phi))
    all_poses = torch.stack(all_poses, dim=0)
    
    return all_poses


def synthesize_views(conf, model, instrinsics, poses, device):
    """
    Render novel views of the scene from specified camera poistions
    """
    H, W, focal = instrinsics
    yy, xx = torch.meshgrid(torch.arange(0., H), 
                            torch.arange(0., W))
    direction = torch.stack([(xx - 0.5*W + 0.5)/focal,
                             -(yy - 0.5*H + 0.5)/focal,
                             -torch.ones_like(xx)], dim=-1) # [H, W, 3]
    rays_d = torch.einsum("hwc, nrc -> nhwr", 
                          direction, poses[:, :3, :3]) # [N, H, W, 3]
    rays_d = rays_d / torch.linalg.norm(rays_d, dim=-1, keepdim=True)

    rays_o = poses[:, :3, -1] # [N, 3]
    rays_o = rays_o[:, None, None, :].expand_as(rays_d) # [N, H, W, 3]

    rays_o = rays_o.reshape(-1, 3).to(device)
    rays_d = rays_d.reshape(-1, 3).to(device)
    
    synth = []
    num_rays = rays_d.shape[0]
    for i in range(0, num_rays, conf["test_batchsize"]):
        raybatch_o = rays_o[i:i+conf["test_batchsize"]]
        raybatch_d = rays_d[i:i+conf["test_batchsize"]]
        output = model(raybatch_o, raybatch_d)
        synth.append(output["color"].detach().clone())
        del output

    synth = torch.cat(synth, dim=0)
    synth = torch.clip(synth, min=0, max=1)
    synth = (255*synth).to(torch.uint8)
    synth = synth.reshape(-1, H, W, 3)

    return synth


def create_360_rendering(conf, model, device, object_dir):
    """
    Create 360 rendering of a specific object
    """
    intrinsics = [128, 128, 177.77] # [H, W, focal]
    poses_360 = get_360_poses() # [N, 4, 4]
    views = synthesize_views(conf, model, intrinsics, poses_360, device)
    
    render_path = object_dir.joinpath("rendering.mp4")
    imageio.mimwrite(render_path, views.cpu().numpy(), fps=30)
