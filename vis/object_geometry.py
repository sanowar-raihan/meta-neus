import numpy as np
import torch
import mcubes


def extract_geometry(model, device, bound_min, bound_max, resolution, threshold):
    """
    Apply marching cubes to extract a triangle mesh from the trained SDF
    """
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = torch.zeros([resolution, resolution, resolution])
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    points = torch.stack([xx, yy, zz], dim=-1)
                    points = points.to(device)
                    sdf, _ = model.geometry_net(points)
                    u[xi*N : xi*N+len(xs),
                      yi*N : yi*N+len(ys),
                      zi*N : zi*N+len(zs)] = sdf.cpu()
    
    vertices, triangles = mcubes.marching_cubes(u.numpy(), threshold)
    vertices = vertices / (resolution-1) * (bound_max - bound_min) + bound_min

    return vertices, triangles


def save_object_geometry(model, device, object_dir):
    """
    Given a trained model, save the geometry of the underlying object
    """
    bound_min = np.asarray([-1.01, -1.01, -1.01])
    bound_max = np.asarray([1.01, 1.01, 1.01])
    resolution = 128
    threshold = 0.0
    vertices, triangles = extract_geometry(model,
                                           device,
                                           bound_min,
                                           bound_max,
                                           resolution,
                                           threshold)
    
    geometry_path = object_dir.joinpath("geometry.obj")
    mcubes.export_obj(vertices, triangles, geometry_path)
