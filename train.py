import argparse
import json
import copy
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.shapenet import build_shapenet
from models.rendering import build_neus


def inner_loop(conf, model, optim, imgs, masks, rays_o, rays_d):
    """
    Train the inner model for a specified number of iteration
    """
    imgs = imgs.reshape(-1, 3)
    masks = masks.reshape(-1, 1)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    num_rays = rays_d.shape[0]
    for step in range(conf["inner_steps"]):
        indices = torch.randint(num_rays, size=[conf["train_batchsize"]])
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


def val_meta(conf, model, val_loader, device):
    """
    Validate the meta trained model for few-shot view synthesis
    """
    meta_trained_state = model.state_dict()
    val_model = copy.deepcopy(model)

    val_psnrs = []
    for imgs, masks, rays_o, rays_d in val_loader:
        imgs, masks = imgs.squeeze(dim=0).to(device), masks.squeeze(dim=0).to(device)
        rays_o, rays_d = rays_o.squeeze(dim=0).to(device), rays_d.squeeze(dim=0).to(device)

        tto_imgs, test_imgs = torch.split(imgs, [conf["tto_views"], conf["test_views"]], dim=0)
        tto_masks, test_masks = torch.split(masks, [conf["tto_views"], conf["test_views"]], dim=0)
        tto_origs, test_origs = torch.split(rays_o, [conf["tto_views"], conf["test_views"]], dim=0)
        tto_dirs, test_dirs = torch.split(rays_d, [conf["tto_views"], conf["test_views"]], dim=0)

        val_model.load_state_dict(meta_trained_state)
        val_optim = torch.optim.Adam(val_model.parameters(), conf["tto_lr"])

        inner_loop(conf, val_model, val_optim, tto_imgs, tto_masks, tto_origs, tto_dirs)
        
        object_psnr = report_result(conf, val_model, test_imgs, test_masks, test_origs, test_dirs)
        val_psnrs.append(object_psnr)

    val_psnr = torch.stack(val_psnrs).mean()
    return val_psnr


def main():
    parser = argparse.ArgumentParser(description="meta-neus for few-shot surface reconstruction")
    parser.add_argument("--config", type=str, required=True,
                        help="config file for the object class (cars or chairs)")
    args = parser.parse_args()

    with open(args.config) as config:
        conf = json.load(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = build_shapenet(image_set="train",
                               dataset_root=conf["dataset_root"],
                               splits_path=conf["splits_path"],
                               num_views=conf["train_views"])
    train_loader = DataLoader(train_set, batch_size=1, num_workers=1, shuffle=True)

    val_set = build_shapenet(image_set="val",
                             dataset_root=conf["dataset_root"],
                             splits_path=conf["splits_path"],
                             num_views=conf["tto_views"]+conf["test_views"]) 
    val_loader = DataLoader(val_set, batch_size=1, num_workers=1, shuffle=False)

    meta_model = build_neus(conf)
    meta_model.to(device)

    meta_optim = torch.optim.Adam(meta_model.parameters(), lr=conf["meta_lr"])

    ckpt_dir = Path(conf["ckptdir"])
    ckpt_dir.mkdir(exist_ok=True)
    
    # Train the meta_model using Reptile meta learning
    # https://arxiv.org/abs/1803.02999
    for epoch in range(conf["meta_epochs"]):
        for step, (imgs, masks, rays_o, rays_d) in enumerate(train_loader, start=1):
            imgs, masks = imgs.squeeze(dim=0).to(device), masks.squeeze(dim=0).to(device)
            rays_o, rays_d = rays_o.squeeze(dim=0).to(device), rays_d.squeeze(dim=0).to(device)

            meta_optim.zero_grad()

            inner_model = copy.deepcopy(meta_model)
            inner_optim = torch.optim.Adam(inner_model.parameters(), conf["inner_lr"])
            inner_loop(conf, inner_model, inner_optim, imgs, masks, rays_o, rays_d)
            
            with torch.no_grad():
                for meta_param, inner_param in zip(meta_model.parameters(), inner_model.parameters()):
                    meta_param.grad = meta_param - inner_param
            
            meta_optim.step()

            iteration = step + epoch * len(train_loader)
            if (iteration % conf["report_iter"] == 0) or (iteration % len(train_loader) == 0):
                val_psnr = val_meta(conf, meta_model, val_loader, device)
                print(f"iteration: {iteration}, val_psnr: {val_psnr:0.3f}")

                ckpt_path = ckpt_dir.joinpath(f"meta_iter{iteration}.pth")
                torch.save(meta_model.state_dict(), ckpt_path)


if __name__ == "__main__":
    main()
