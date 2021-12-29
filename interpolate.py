from pathlib import Path
import argparse
import json
import math
import imageio
import torch
from models.rendering import build_neus
from vis.object_rendering import create_posemat, synthesize_views


def define_camera():
    """
    Define a custom camera for view synthesis
    """
    intrinsics = [128, 128, 177.77] # [H, W, focal]

    radius = torch.as_tensor(2.0, dtype=torch.float)
    theta = torch.as_tensor(math.pi/6, dtype=torch.float)
    phi = torch.as_tensor(-math.pi/5, dtype=torch.float)
    pose = create_posemat(radius, theta, phi)
    
    return intrinsics, pose[None, :, :]


def interpolate():
    parser = argparse.ArgumentParser(description="interpolate the geometry or appearance of two objects")
    parser.add_argument("--config", type=str, required=True,
                        help="config file for the object class (cars or chairs)")
    parser.add_argument("--first-weight", type=str, required=True,
                        help="weight file of the first object")
    parser.add_argument("--second-weight", type=str, required=True,
                        help="weight file of the second object")
    parser.add_argument("--property", type=str, required=True,
                        help="property to interpolate (geometry or appearance)")
    args = parser.parse_args()

    with open(args.config) as config:
        conf = json.load(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_neus(conf)
    model.to(device)

    first_state = torch.load(args.first_weight, map_location=device)
    second_state = torch.load(args.second_weight, map_location=device)
    
    resultdir = Path(conf["resultdir"])
    resultdir.mkdir(exist_ok=True)
    
    print(f"interpolating {args.property} ...")

    video_frames = []
    intrinsics, pose = define_camera()
    for t in torch.linspace(0, 1, 60):
        state_dict = {name:
            first_state[name]*(1-t) + second_state[name]*t
            if args.property in name else first_state[name]
            for name in first_state
        }
        model.load_state_dict(state_dict)
        frame = synthesize_views(conf, model, intrinsics, pose, device)
        video_frames.append(frame)

    video_frames = torch.cat(video_frames, axis=0)
    video_path = resultdir.joinpath(f"{args.property}_interpolation.mp4")
    imageio.mimwrite(video_path, video_frames.cpu().numpy(), fps=30)

    print(f"interpolation video created at {video_path}")


if __name__ == "__main__":
    interpolate()
