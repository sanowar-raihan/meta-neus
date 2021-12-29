from pathlib import Path
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.shapenet import build_shapenet
from models.rendering import build_neus
from vis.object_rendering import create_360_rendering
from vis.object_geometry import save_object_geometry


def test_time_optimize(conf, model, optim, imgs, masks, rays_o, rays_d):
    """"
    Test-time-optimize the meta trained model on available views
    """
    imgs = imgs.reshape(-1, 3)
    masks = masks.reshape(-1, 1)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    num_rays = rays_d.shape[0]
    for step in range(conf["tto_steps"]):
        indices = torch.randint(num_rays, size=[conf["tto_batchsize"]])
        img_batch, mask_batch = imgs[indices], masks[indices]
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]

        optim.zero_grad()
        output = model(raybatch_o, raybatch_d)
        color_error = F.l1_loss(output["color"], img_batch, reduction="none") * mask_batch
        color_loss = color_error.sum() / (3 * mask_batch.sum() + 1e-7)

        grad_norm = torch.linalg.norm(output["sdf_grad"], dim=-1)
        eikonal_loss = F.mse_loss(grad_norm, torch.ones_like(grad_norm))

        mask_loss = F.binary_cross_entropy(output["weight_sum"].clip(1e-3, 1-1e-3),
                                            mask_batch)
        
        loss = color_loss + conf["igr_weight"] * eikonal_loss + conf["mask_weight"] * mask_loss
        loss.backward()
        optim.step()


def report_result(conf, model, imgs, masks, rays_o, rays_d):
    """
    Report view-synthesis result on heldout views
    """
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    synth = []
    num_rays = rays_d.shape[0]
    for i in range(0, num_rays, conf["test_batchsize"]):
        raybatch_o = rays_o[i:i+conf["test_batchsize"]]
        raybatch_d = rays_d[i:i+conf["test_batchsize"]]
        output = model(raybatch_o, raybatch_d)
        synth.append(output["color"].detach().clone())
        del output
    synth = torch.cat(synth, dim=0).reshape_as(imgs)
    
    error = F.mse_loss(imgs, synth, reduction="none") * masks
    loss = error.sum() / (3 * masks.sum() + 1e-7)
    psnr = -10*torch.log10(loss)

    return psnr


def test():
    parser = argparse.ArgumentParser(description="meta-neus for few-shot surface reconstruction")
    parser.add_argument("--config", type=str, required=True,
                        help="config file for the object class (cars or chairs)")
    parser.add_argument("--meta-weight", type=str, required=True,
                        help="path to the meta-trained weight file")
    args = parser.parse_args()

    with open(args.config) as config:
        conf = json.load(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = build_shapenet(image_set="test",
                              dataset_root=conf["dataset_root"],
                              splits_path=conf["splits_path"],
                              num_views=conf["tto_views"]+conf["test_views"])
    test_loader = DataLoader(test_set, batch_size=1, num_workers=1, shuffle=False)

    model = build_neus(conf)
    model.to(device)

    meta_state = torch.load(args.meta_weight, map_location=device)

    resultdir = Path(conf["resultdir"])
    resultdir.mkdir(exist_ok=True)

    test_psnrs = []
    for idx, (imgs, masks, rays_o, rays_d) in enumerate(test_loader, start=1):
        imgs, masks = imgs.squeeze(dim=0).to(device), masks.squeeze(dim=0).to(device)
        rays_o, rays_d = rays_o.squeeze(dim=0).to(device), rays_d.squeeze(dim=0).to(device)

        tto_imgs, test_imgs = torch.split(imgs, [conf["tto_views"], conf["test_views"]], dim=0)
        tto_masks, test_masks = torch.split(masks, [conf["tto_views"], conf["test_views"]], dim=0)
        tto_origs, test_origs = torch.split(rays_o, [conf["tto_views"], conf["test_views"]], dim=0)
        tto_dirs, test_dirs = torch.split(rays_d, [conf["tto_views"], conf["test_views"]], dim=0)

        model.load_state_dict(meta_state)
        optim = torch.optim.Adam(model.parameters(), conf["tto_lr"])

        test_time_optimize(conf, model, optim, tto_imgs, tto_masks, tto_origs, tto_dirs)
        object_psnr = report_result(conf, model, test_imgs, test_masks, test_origs, test_dirs)

        object_dir = resultdir.joinpath(f"{idx}")
        object_dir.mkdir(exist_ok=True)

        save_object_geometry(model, device, object_dir)
        create_360_rendering(conf, model, device, object_dir)
        torch.save(model.state_dict(), object_dir.joinpath("weights.pth"))

        print(f"Object id:{idx}, psnr:{object_psnr:.3f}, geometry extracted and object rendered")
        test_psnrs.append(object_psnr)
    
    test_psnr = torch.stack(test_psnrs).mean()
    print("----------------------------------")
    print(f"test dataset mean psnr: {test_psnr:.3f}")


if __name__ == "__main__":
    test()
