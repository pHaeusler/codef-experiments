import cv2
import numpy as np
import json

import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam
from typing import List
from dataset import VideoDataset, make_image_grid, load_image

from models import AnnealedHash
from models import IMLP, ImplicitVideo_Hash, Deform_Hash3d_Warp

from torch import nn

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import LightningModule, Trainer

from pytorch_lightning.loggers import WandbLogger
from torchvision.transforms.functional import rgb_to_grayscale
import argparse
from enum import Enum


class WarpType(Enum):
    MLP = 0
    HASH_GRID = 1


def compute_gradient_loss(pred, gt):
    pred = pred.permute(0, 3, 1, 2)
    gt = gt.permute(0, 3, 1, 2)
    pred = rgb_to_grayscale(pred)
    gt = rgb_to_grayscale(gt)

    sobel_kernel_x = (
        torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    sobel_kernel_y = (
        torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )

    gradient_a_x = torch.nn.functional.conv2d(pred, sobel_kernel_x, padding=1)
    gradient_a_y = torch.nn.functional.conv2d(pred, sobel_kernel_y, padding=1)

    gradient_b_x = torch.nn.functional.conv2d(gt, sobel_kernel_x, padding=1)
    gradient_b_y = torch.nn.functional.conv2d(gt, sobel_kernel_y, padding=1)

    pred_grad = torch.cat([gradient_a_x, gradient_a_y], dim=1)
    gt_grad = torch.cat([gradient_b_x, gradient_b_y], dim=1)
    return nn.L1Loss()(pred_grad, gt_grad)


def pre_train_implicit_video(model, ref_image, pretrain_iters=200):
    optimizer_mapping = torch.optim.Adam(model.parameters(), lr=0.001)

    B = 1
    grid = make_image_grid(w, h).repeat(B, 1, 1)
    grid = grid.reshape(-1, 2)

    grid = grid.cuda()
    model = model.to(torch.float32).cuda()
    ref = ref_image.unsqueeze(0).to(torch.float16).cuda()

    for i in range(pretrain_iters):
        rgbs = model(grid).reshape(B, h, w, 3)
        model.zero_grad()
        loss = (rgbs - ref).square().mean() + compute_gradient_loss(ref, rgbs)
        print(f"PRETRAIN: {i} loss={loss}")
        loss.backward()
        optimizer_mapping.step()


def deform_xyt(ts_w, grid, warping_field):
    grid_col = grid.reshape(grid.shape[0], -1, 2)
    ts_w = ts_w.repeat(1, grid.shape[1] * grid.shape[2]).unsqueeze(2)
    input_xyt = torch.cat([grid_col, ts_w], dim=2).reshape(-1, 3)
    deform = warping_field(input_xyt)
    deforms = deform.reshape(*grid.shape)  # Unroll batches
    return deforms + grid, deforms


def sample_canonical(canonical_img, deformed_grids):
    canonical_img = canonical_img.unsqueeze(0).repeat(deformed_grids.shape[0], 1, 1, 1)
    can = canonical_img.permute(0, 3, 1, 2)
    results = torch.nn.functional.grid_sample(
        can,
        deformed_grids,
        mode="bilinear",
        padding_mode="border",
    )
    return results.permute(0, 2, 3, 1)


def compute_flow_loss(ts_w, grid, deforms, flow, warping_field, t_step):
    next_ts_w = ts_w + t_step
    _, next_deforms = deform_xyt(next_ts_w, grid, warping_field)
    delta_deform = next_deforms - deforms
    return torch.nn.functional.l1_loss(delta_deform, flow)


def to_cv_im(im_tensor):
    im = im_tensor.squeeze(0).cpu().numpy()
    im = np.clip(im, 0, 1)
    im = (im * 255).astype(np.uint8)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


class ImplicitVideoSystem(LightningModule):
    def __init__(
        self,
        w: int,
        h: int,
        lr: float,
        lr_milestones: List[int],
        annealed: bool,
        warping_type: WarpType = WarpType.MLP,
    ):
        super(ImplicitVideoSystem, self).__init__()
        self.h = h
        self.w = w
        self.lr = lr
        self.lr_milestones = lr_milestones
        self.annealed = annealed
        self.annealed_step = 100
        self.annealed_begin_step = 0
        self.canonical_img = None

        if self.annealed:
            self.embedding_hash = AnnealedHash(
                in_channels=2,
                annealed_step=self.annealed_step,
                annealed_begin_step=self.annealed_begin_step,
            )

        with open("hash.json") as f:
            config = json.load(f)

        self.implicit_video = ImplicitVideo_Hash(config=config)

        if warping_type == WarpType.MLP:
            self.warping_field = IMLP(
                input_dim=3,
                output_dim=2,
                use_positional=False,
                num_layers=6,
                hidden_dim=512,
            )
        else:
            self.warping_field = Deform_Hash3d_Warp(config=config)

    def configure_optimizers(self):
        parameters = list(self.warping_field.parameters()) + list(
            self.implicit_video.parameters()
        )
        self.optimizer = Adam(parameters, lr=self.lr)
        scheduler = MultiStepLR(
            self.optimizer, milestones=self.lr_milestones, gamma=0.5
        )
        lr_dict = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [self.optimizer], [lr_dict]

    def forward(self, ts_w, grid, encode_w):
        if encode_w:
            deformed_grids, deforms = deform_xyt(ts_w, grid, self.warping_field)
        else:
            deformed_grids, deforms = grid, torch.zeros_like(grid)
        if self.canonical_img is None:
            pe_deformed_grid = deformed_grids.reshape(-1, 2)
            rgbs = self.implicit_video(pe_deformed_grid)
            rgbs = rgbs.reshape(grid.shape[0], self.h, self.w, 3)
        else:
            rgbs = sample_canonical(self.canonical_img, deformed_grids)
        return rgbs, deforms

    def training_step(self, batch, batch_idx):
        rgbs = batch["rgbs"]  # [B, H, W, 3]
        ts_w = batch["ts_w"]  # [B, 1]
        grid = batch["grid"]  # [B, H, W, 2]
        flow = batch["flow"]  # [B, H, W, 2]

        total_length = len(self.trainer.train_dataloader.dataset)

        rgbs_pred, deforms = self.forward(ts_w, grid, True)

        color_loss = nn.MSELoss()(rgbs_pred, rgbs)
        grad_loss = compute_gradient_loss(rgbs_pred, rgbs)
        # flow_loss = compute_flow_loss(
        #     ts_w, grid, deforms, flow, self.warping_field, t_step=1.0 / total_length
        # )
        flow_loss = 0
        loss = color_loss + grad_loss + flow_loss

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/color_loss", color_loss, prog_bar=True)
        self.log("train/flow_loss", flow_loss, prog_bar=True)
        self.log("train/grad_loss", grad_loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ts_w = batch["ts_w"]
        grid = batch["grid"]
        idxs = batch["idxs"]

        if not os.path.exists("results"):
            os.makedirs("results")

        with torch.no_grad():
            rgbs_pred, deforms = self.forward(ts_w=ts_w, grid=grid, encode_w=True)
            rgbs_pred_c, _ = self.forward(ts_w=ts_w, grid=grid, encode_w=False)

            for i in range(len(rgbs_pred)):
                idx = idxs[i].item()

                cv2.imwrite(f"results/{idx:05d}.png", to_cv_im(rgbs_pred[i]))

                im = deforms[i].squeeze(0)
                im = im.mean(dim=-1).cpu().numpy()
                im = ((im - np.min(im)) / (np.max(im) - np.min(im))) * 255
                im = np.clip(im, 0, 255).astype(np.uint8)
                cv2.imwrite(f"results/deform_{idx:05d}.png", im)

                # cv2.imwrite(f"results/rgbs_{idx:05d}.png", to_cv_im(rgbs[i]))

            cv2.imwrite("results/canonical_0.png", to_cv_im(rgbs_pred_c[0]))

        return {}


def sample_video(
    model: ImplicitVideoSystem, w, h, canonical_image: str, length: int, batch_size: int
):
    if canonical_image:
        model.canonical_img = (
            torch.from_numpy(load_image(canonical_image, w, h)).float() / 255
        ).cuda()

    model = model.cuda()
    grid = make_image_grid(w, h).repeat(batch_size, 1, 1, 1).cuda()
    ts_w = torch.linspace(0, 1, length).unsqueeze(1)

    num_batches = length // batch_size
    idx = 1
    for i in range(num_batches):
        ts_w_b = ts_w[i * batch_size : (i + 1) * batch_size].cuda()

        with torch.no_grad():
            rgbs, _ = model.forward(ts_w=ts_w_b, grid=grid, encode_w=True)

        for i in range(len(rgbs)):
            cv2.imwrite(f"results/{idx:05d}.png", to_cv_im(rgbs[i]))
            idx += 1
            if idx > length:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--batch_size", default=10, type=int)
    train_parser.add_argument("--image_dir", required=True)
    train_parser.add_argument("--canonical", default=None)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--batch_size", default=10, type=int)
    generate_parser.add_argument("--checkpoint", required=True)
    generate_parser.add_argument("--canonical", default=None)
    generate_parser.add_argument(
        "--length", required=True, help="Number of images in sequence", type=int
    )

    args = parser.parse_args()

    model_save_path = "checkpoints"
    log_wandb = False
    lr = 0.002  # works well for can im
    lr = 0.002
    lr_milestones = [100, 150, 200]
    w = 540
    h = 540
    annealed = True

    if args.command == "generate":
        model = ImplicitVideoSystem.load_from_checkpoint(
            args.checkpoint,
            w=w,
            h=h,
            lr=lr,
            lr_milestones=lr_milestones,
            annealed=annealed,
        )
        model.eval()
        sample_video(
            model,
            w=w,
            h=h,
            canonical_image=args.canonical,
            length=args.length,
            batch_size=args.batch_size,
        )
        exit(0)

    if args.command == "train":
        dataset = VideoDataset(w=w, h=h, root_dir=args.image_dir)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        model = ImplicitVideoSystem(
            w=w, h=h, lr=lr, lr_milestones=lr_milestones, annealed=annealed
        )

        if args.canonical:
            model.canonical_img = (
                torch.from_numpy(load_image(args.canonical, w, h)).float() / 255
            ).cuda()
        else:
            # pre_train_implicit_video(model.implicit_video, ref_image=dataset[0]["rgbs"])
            pass

        model.warping_field._initialize_weights()

        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{model_save_path}",
            filename="{epoch:d}",
            mode="max",
            save_top_k=-1,
            every_n_epochs=10,
            save_last=True,
        )

        logger = WandbLogger(project="codef") if log_wandb else None

        trainer = Trainer(
            max_epochs=1000,
            precision=16,
            callbacks=[
                checkpoint_callback,
                LearningRateMonitor(logging_interval="step"),
            ],
            logger=logger,
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
        )

        trainer.fit(
            model,
            train_dataloaders=dataloader,
            val_dataloaders=dataloader,
        )
