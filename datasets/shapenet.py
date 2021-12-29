from pathlib import Path
import json
import imageio
import torch
from torch.utils.data import Dataset


class ShapenetDataset(Dataset):
    """
    Returns images, masks and camera rays for a particular shapenet object
    """
    def __init__(self, all_folders, num_views):
        """
        Args:
            all_folders: each folder path contains individual object information
            num_views: number of views to return for each object
        """
        super().__init__()
        self.all_folders = all_folders
        self.num_views = num_views

    def __getitem__(self, idx):
        folderpath = self.all_folders[idx]
        meta_path = folderpath.joinpath("transforms.json")
        with open(meta_path, "r") as read_file:
            meta_data = json.load(read_file)
        
        all_views = []
        all_poses = []
        for frame_idx in range(self.num_views):
            frame = meta_data["frames"][frame_idx]
            
            view_name = frame["file_path"]
            view_path = folderpath.joinpath(view_name)
            view = imageio.imread(view_path)
            all_views.append(torch.as_tensor(view, dtype=torch.float))

            pose = frame["transform_matrix"]
            all_poses.append(torch.as_tensor(pose, dtype=torch.float))

        all_views = torch.stack(all_views, dim=0) / 255. # [N, H, W, 4]
        
        # composite the images to a white background
        images = all_views[...,:3] * all_views[...,-1:] + 1-all_views[...,-1:] # [N, H, W, 3]
        masks = (all_views[...,-1:] > 0.5).float() # [N, H, W, 1]

        all_poses = torch.stack(all_poses, dim=0) # [N, 4, 4]

        # all images of an object has the same camera intrinsics
        H, W = images[0].shape[:2]
        camera_angle_x = meta_data["camera_angle_x"]
        camera_angle_x = torch.as_tensor(camera_angle_x, dtype=torch.float)

        # focal length from camera angle equation
        # tan(angle/2) = (W/2)/focal
        focal = 0.5 * W / torch.tan(0.5 * camera_angle_x)
        
        yy, xx = torch.meshgrid(torch.arange(0., H),
                                torch.arange(0., W))
        direction = torch.stack([(xx - 0.5*W + 0.5)/focal,
                                 -(yy - 0.5*H + 0.5)/focal,
                                 -torch.ones_like(xx)], dim=-1) # [H, W, 3]
        rays_d = torch.einsum("hwc, nrc -> nhwr",
                              direction, all_poses[:, :3, :3]) # [N, H, W, 3]
        rays_d = rays_d / torch.linalg.norm(rays_d, dim=-1, keepdim=True)

        rays_o = all_poses[:, :3, -1]  # [N, 3]
        rays_o = rays_o[:, None, None, :].expand_as(rays_d)  # [N, H, W, 3]

        return images, masks, rays_o, rays_d

    def __len__(self):
        return len(self.all_folders)


def build_shapenet(image_set, dataset_root, splits_path, num_views):
    """
    Args:
        image_set: specifies whether to return "train", "val" or "test" dataset
        dataset_root: root path of the dataset
        splits_path: file path that specifies train, val and test split
        num_views: number of views to return for a single object
    """
    root_path = Path(dataset_root)
    splits_path = Path(splits_path)
    with open(splits_path, "r") as read_file:
        splits = json.load(read_file)
    
    all_folders = [root_path.joinpath(foldername) for foldername in sorted(splits[image_set])]
    dataset = ShapenetDataset(all_folders, num_views)

    return dataset
