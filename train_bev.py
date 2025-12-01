# train_geom_bev.py

import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import timm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

IMG_HEIGHT = 384
IMG_WIDTH = 768

BEV_HEIGHT = 188
BEV_WIDTH = 126

X_MIN, X_MAX = 0.0, 150.0
Y_MIN, Y_MAX = -50.0, 50.0

IGNORE_VALUE = 255

CAMERA_NAMES = [
    "/camera/inner/frontal/middle",
    "/camera/inner/frontal/far",
    "/side/left/forward",
    "/side/right/forward",
]

INTRINSICS_NAMES = [
    "/camera/inner/frontal/middle/intrinsic_params",
    "/camera/inner/frontal/far/intrinsic_params",
    "/side/left/forward/intrinsic_params",
    "/side/right/forward/intrinsic_params",
]

CAR2CAM_NAMES = [
    "/camera/inner/frontal/middle/car_to_cam",
    "/camera/inner/frontal/far/car_to_cam",
    "/side/left/forward/car_to_cam",
    "/side/right/forward/car_to_cam",
]

GT_GRID_COL = "gt_occupancy_grid"
PRED_GRID_COL = "predicted_occupancy_grid"
ROVER_COL = "rover"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--encoder_name", type=str, default="resnet18")
    parser.add_argument("--encoder_pretrained", action="store_true")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)

    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--use_gaussian_loss", action="store_true")
    parser.add_argument("--gaussian_kernel_size", type=int, default=3)
    parser.add_argument("--gaussian_sigma", type=float, default=1.5)
    parser.add_argument("--gaussian_alpha", type=float, default=0.7)

    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--val_check_interval", type=float, default=1.0 / 4.0)
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument("--thr_step", type=float, default=0.05)

    parser.add_argument("--rover_embed_dim", type=int, default=32)

    args = parser.parse_args()
    return args


class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0.0, std=0.02, p: float = 0.5):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return img
        noise = torch.randn_like(img) * self.std + self.mean
        return torch.clamp(img + noise, 0.0, 1.0)


class StaticGridDataset(Dataset):

    def __init__(
        self,
        data_dir: Path,
        mode: str = "train",
        img_height: int = IMG_HEIGHT,
        img_width: int = IMG_WIDTH,
        use_augment: bool = True,
        rover2idx: Optional[Dict[str, int]] = None,
    ):
        assert mode in ("train", "val", "test")
        self.data_dir = Path(data_dir).resolve()
        self.mode = mode
        self.img_height = img_height
        self.img_width = img_width
        self.use_augment = use_augment and (mode == "train")

        self.info = pd.read_csv(self.data_dir / "info.csv", index_col=0)
        self.sample_indices: List[int] = list(range(len(self.info)))

        if self.mode in ("train", "val"):
            self.sample_indices = self._filter_fully_ignored_samples()

        if rover2idx is not None:
            self.rover2idx = rover2idx
        else:
            all_rovers = self.info[ROVER_COL].unique().tolist()
            self.rover2idx = {r: i for i, r in enumerate(sorted(all_rovers))}
        self.num_rovers = len(self.rover2idx)

        self.base_transform = T.Compose(
            [
                T.Resize((self.img_height, self.img_width)),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        if self.use_augment:
            self.augment_transform = T.Compose(
                [
                    T.ColorJitter(
                        brightness=0.3,
                        contrast=0.3,
                        saturation=0.3,
                        hue=0.05,
                    ),
                    T.RandomGrayscale(p=0.1),
                    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
                ]
            )
            self.noise_transform = AddGaussianNoise(mean=0.0, std=0.02, p=0.5)
        else:
            self.augment_transform = None
            self.noise_transform = None

    def __len__(self) -> int:
        return len(self.sample_indices)

    def _is_fully_ignored_grid(self, grid: torch.Tensor) -> bool:
        valid = (grid != IGNORE_VALUE) & (grid >= 0)
        return valid.sum().item() == 0

    def _filter_fully_ignored_samples(self) -> List[int]:
        kept_indices: List[int] = []
        dropped = 0

        for idx in self.sample_indices:
            row = self.info.iloc[idx]
            gt_path = self._resolve(row[GT_GRID_COL])
            grid = self._load_gt_grid(gt_path)

            if self._is_fully_ignored_grid(grid):
                dropped += 1
            else:
                kept_indices.append(idx)

        print(
            f"[StaticGridDataset] mode={self.mode}: "
            f"dropped {dropped} fully-ignored samples, kept {len(kept_indices)}"
        )
        return kept_indices

    def _resolve(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        if len(p.parts) > 0 and p.parts[0] == self.data_dir.name:
            p = Path(*p.parts[1:])
        return (self.data_dir / p).resolve()

    @staticmethod
    def _load_gt_grid(path: Path) -> torch.Tensor:
        grid = np.load(path)
        grid = np.squeeze(grid)
        return torch.from_numpy(grid.astype(np.int64))

    def _load_images_intrinsics_extrinsics(
        self,
        row: pd.Series,
    ):
        image_paths = [self._resolve(row[name]) for name in CAMERA_NAMES]
        intrinsics_paths = [self._resolve(row[name]) for name in INTRINSICS_NAMES]
        car2cam_paths = [self._resolve(row[name]) for name in CAR2CAM_NAMES]

        images: List[torch.Tensor] = []
        intrinsics_scaled: List[torch.Tensor] = []
        car2cam: List[torch.Tensor] = []

        for img_path, k_path, ext_path in zip(
            image_paths,
            intrinsics_paths,
            car2cam_paths,
        ):
            img = Image.open(img_path).convert("RGB")
            w_orig, h_orig = img.size

            if self.augment_transform is not None:
                img = self.augment_transform(img)

            img_t = self.base_transform(img)
            if self.noise_transform is not None:
                img_t = self.noise_transform(img_t)
            images.append(img_t)

            K = np.load(k_path).astype(np.float32)
            sx = self.img_width / float(w_orig)
            sy = self.img_height / float(h_orig)
            K_scaled = K.copy()
            K_scaled[0, 0] *= sx
            K_scaled[0, 2] *= sx
            K_scaled[1, 1] *= sy
            K_scaled[1, 2] *= sy
            intrinsics_scaled.append(torch.from_numpy(K_scaled))

            ext = np.load(ext_path).astype(np.float32)
            car2cam.append(torch.from_numpy(ext))

        images_tensor = torch.stack(images, dim=0)
        intrinsics_tensor = torch.stack(intrinsics_scaled, 0)
        car2cam_tensor = torch.stack(car2cam, 0)

        return images_tensor, intrinsics_tensor, car2cam_tensor

    def _get_rover_idx(self, row: pd.Series) -> torch.Tensor:
        rover_name = row[ROVER_COL]
        rover_idx = self.rover2idx.get(rover_name, 0)
        return torch.tensor(rover_idx, dtype=torch.long)

    def __getitem__(self, idx: int):
        num_samples = len(self.sample_indices)

        if self.mode == "test":
            real_idx = self.sample_indices[idx]
            row = self.info.iloc[real_idx]
            images, intrinsics, car2cam = self._load_images_intrinsics_extrinsics(row)
            pred_path = str(self._resolve(row[PRED_GRID_COL]))
            rover_idx = self._get_rover_idx(row)
            return images, intrinsics, car2cam, pred_path, rover_idx

        max_attempts = min(5, num_samples)
        last_sample = (None, None, None, None, None)

        for attempt in range(max_attempts):
            real_idx = self.sample_indices[(idx + attempt) % num_samples]
            row = self.info.iloc[real_idx]

            images, intrinsics, car2cam = self._load_images_intrinsics_extrinsics(row)
            gt_path = self._resolve(row[GT_GRID_COL])
            gt_grid = self._load_gt_grid(gt_path)
            rover_idx = self._get_rover_idx(row)

            valid = (gt_grid != IGNORE_VALUE) & (gt_grid >= 0)
            n_valid = int(valid.sum().item())

            last_sample = (images, intrinsics, car2cam, gt_grid, rover_idx)

            if n_valid > 0:
                return images, intrinsics, car2cam, gt_grid, rover_idx

        images, intrinsics, car2cam, gt_grid, rover_idx = last_sample
        return images, intrinsics, car2cam, gt_grid, rover_idx


class FeatureIPMProjector(nn.Module):
    def __init__(
        self,
        bev_height: int = BEV_HEIGHT,
        bev_width: int = BEV_WIDTH,
        x_min: float = X_MIN,
        x_max: float = X_MAX,
        y_min: float = Y_MIN,
        y_max: float = Y_MAX,
        img_height: int = IMG_HEIGHT,
        img_width: int = IMG_WIDTH,
    ):
        super().__init__()
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.img_height = img_height
        self.img_width = img_width

        dx = (x_max - x_min) / bev_height
        dy = (y_max - y_min) / bev_width

        x_centers = torch.linspace(x_min + dx / 2.0, x_max - dx / 2.0, bev_height)
        y_centers = torch.linspace(y_min + dy / 2.0, y_max - dy / 2.0, bev_width)

        X, Y = torch.meshgrid(x_centers, y_centers, indexing="ij")
        self.register_buffer("X_grid", X)
        self.register_buffer("Y_grid", Y)

    @staticmethod
    def _normalize_intrinsics(K: torch.Tensor) -> torch.Tensor:
        if K.dim() != 3:
            raise ValueError(f"Intrinsics must be [B,*,*], got {K.shape}")
        b, h, w = K.shape
        if h < 3 or w < 3:
            raise ValueError(f"Intrinsics too small: {K.shape}")
        if (h, w) == (3, 3):
            return K
        return K[:, :3, :3]

    @staticmethod
    def _normalize_extrinsics(M: torch.Tensor) -> torch.Tensor:
        if M.dim() != 3:
            raise ValueError(f"Extrinsics must be [B,*,*], got {M.shape}")
        b, h, w = M.shape
        if (h, w) == (4, 4):
            return M
        if (h, w) == (3, 4):
            bottom = torch.tensor(
                [[0.0, 0.0, 0.0, 1.0]],
                device=M.device,
                dtype=M.dtype,
            )
            bottom = bottom.expand(b, -1).view(b, 1, 4)
            return torch.cat([M, bottom], dim=1)
        raise ValueError(f"Unsupported extrinsics shape: {M.shape}")

    def _build_sampling_grid(
        self,
        intrinsics: torch.Tensor,
        car_to_cam: torch.Tensor,
        feat_h: int,
        feat_w: int,
    ) -> torch.Tensor:
        device = intrinsics.device
        B = intrinsics.shape[0]

        K = self._normalize_intrinsics(intrinsics).to(device)
        M = self._normalize_extrinsics(car_to_cam).to(device)

        car_to_cam = M

        X = self.X_grid.to(device).view(1, self.bev_height, self.bev_width)
        Y = self.Y_grid.to(device).view(1, self.bev_height, self.bev_width)
        X = X.expand(B, -1, -1)
        Y = Y.expand(B, -1, -1)
        Z = torch.zeros_like(X)
        ones = torch.ones_like(X)

        P_car = torch.stack([X, Y, Z, ones], dim=-1).view(B, -1, 4)

        P_cam = torch.bmm(P_car, car_to_cam.transpose(1, 2))
        P_cam3 = P_cam[..., :3]

        Xc = P_cam3[..., 0]
        Yc = P_cam3[..., 1]
        Zc = P_cam3[..., 2]

        valid = Zc > 0.1

        P_cam3_T = P_cam3.transpose(1, 2)
        p_img = torch.bmm(K, P_cam3_T).transpose(1, 2)
        z = p_img[..., 2].clamp(min=1e-4)
        u = p_img[..., 0] / z
        v = p_img[..., 1] / z

        down_ratio_x = self.img_width / float(feat_w)
        down_ratio_y = self.img_height / float(feat_h)

        u_feat = u / down_ratio_x
        v_feat = v / down_ratio_y

        valid = (
            valid
            & (u_feat >= 0.0)
            & (u_feat <= feat_w - 1.0)
            & (v_feat >= 0.0)
            & (v_feat <= feat_h - 1.0)
        )

        u_norm = (u_feat / (feat_w - 1)) * 2.0 - 1.0
        v_norm = (v_feat / (feat_h - 1)) * 2.0 - 1.0

        u_norm = torch.where(valid, u_norm, torch.full_like(u_norm, 2.0))
        v_norm = torch.where(valid, v_norm, torch.full_like(v_norm, 2.0))

        grid = torch.stack([u_norm, v_norm], dim=-1).view(
            B, self.bev_height, self.bev_width, 2
        )
        return grid

    def forward(
        self,
        feats: torch.Tensor,
        intrinsics: torch.Tensor,
        car_to_cam: torch.Tensor,
    ) -> torch.Tensor:
        B, N_cam, C_feat, Hf, Wf = feats.shape
        device = feats.device

        bev_list = []
        for cam_idx in range(N_cam):
            feat_cam = feats[:, cam_idx]
            K = intrinsics[:, cam_idx]
            M = car_to_cam[:, cam_idx]

            grid = self._build_sampling_grid(K, M, feat_h=Hf, feat_w=Wf)

            bev_cam = F.grid_sample(
                feat_cam,
                grid.to(device),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            bev_list.append(bev_cam)

        bev_stack = torch.stack(bev_list, dim=1)
        bev_feats = bev_stack.mean(dim=1)
        return bev_feats


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BEVUNetDecoder(nn.Module):
    def __init__(
        self, in_channels: int, base_channels: int = 64, dropout_p: float = 0.1
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.enc1 = DoubleConv(in_channels, c1, dropout_p)
        self.enc2 = DoubleConv(c1, c2, dropout_p)
        self.enc3 = DoubleConv(c2, c3, dropout_p)
        self.enc4 = DoubleConv(c3, c4, dropout_p)
        self.bottom = DoubleConv(c4, c5, dropout_p)

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c4 + c4, c4, dropout_p)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c3 + c3, c3, dropout_p)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2, dropout_p)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1, dropout_p)

        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)

    @staticmethod
    def _center_crop_to(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _, _, Hx, Wx = x.shape
        _, _, Ht, Wt = target.shape
        if Hx == Ht and Wx == Wt:
            return x
        dh = Hx - Ht
        dw = Wx - Wt
        top = dh // 2
        left = dw // 2
        bottom = top + Ht
        right = left + Wt
        return x[:, :, top:bottom, left:right]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottom(self.pool(e4))

        d4 = self.up4(b)
        e4_c = self._center_crop_to(e4, d4)
        d4 = self.dec4(torch.cat([d4, e4_c], dim=1))

        d3 = self.up3(d4)
        e3_c = self._center_crop_to(e3, d3)
        d3 = self.dec3(torch.cat([d3, e3_c], dim=1))

        d2 = self.up2(d3)
        e2_c = self._center_crop_to(e2, d2)
        d2 = self.dec2(torch.cat([d2, e2_c], dim=1))

        d1 = self.up1(d2)
        e1_c = self._center_crop_to(e1, d1)
        d1 = self.dec1(torch.cat([d1, e1_c], dim=1))

        logits = self.out_conv(d1)
        logits = F.interpolate(
            logits,
            size=(BEV_HEIGHT, BEV_WIDTH),
            mode="bilinear",
            align_corners=False,
        )
        return logits


class GeomBEVStaticOccModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet18",
        encoder_pretrained: bool = True,
        num_cams: int = 4,
        num_rovers: int = 32,
        rover_embed_dim: int = 32,
    ):
        super().__init__()
        self.num_cams = num_cams
        self.num_rovers = num_rovers
        self.rover_embed_dim = rover_embed_dim

        self.encoder = timm.create_model(
            encoder_name,
            in_chans=3,
            features_only=True,
            pretrained=encoder_pretrained,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_HEIGHT, IMG_WIDTH)
            feats = self.encoder(dummy)

        if len(feats) >= 2:
            self.feat_indices = [-1, -2]
        else:
            self.feat_indices = [-1]

        self.num_scales = len(self.feat_indices)

        bev_channels_per_scale = 64
        self.bev_channels_per_scale = bev_channels_per_scale
        self.c_bev_total = self.num_scales * bev_channels_per_scale

        self.input_projs = nn.ModuleList()
        for idx in self.feat_indices:
            c_back = feats[idx].shape[1]
            self.input_projs.append(
                nn.Conv2d(c_back, bev_channels_per_scale, kernel_size=1)
            )

        self.projector = FeatureIPMProjector(
            bev_height=BEV_HEIGHT,
            bev_width=BEV_WIDTH,
            x_min=X_MIN,
            x_max=X_MAX,
            y_min=Y_MIN,
            y_max=Y_MAX,
            img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH,
        )

        self.rover_embedding = nn.Embedding(num_rovers, rover_embed_dim)
        self.rover_proj = nn.Sequential(
            nn.Linear(rover_embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
        )

        self.bev_decoder = BEVUNetDecoder(
            in_channels=self.c_bev_total + 128,
            base_channels=64,
            dropout_p=0.1,
        )

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        car_to_cam: torch.Tensor,
        rover_idx: torch.Tensor,
    ) -> torch.Tensor:
        B, N_cam, C_in, H, W = images.shape
        assert N_cam == self.num_cams, f"Expected {self.num_cams} cams, got {N_cam}"

        x = images.view(B * N_cam, C_in, H, W)
        feats_list = self.encoder(x)

        bev_feats_scales = []

        for proj, idx in zip(self.input_projs, self.feat_indices):
            feat = feats_list[idx]
            feat = proj(feat)

            _, C_feat, Hf, Wf = feat.shape
            feat = feat.view(B, N_cam, C_feat, Hf, Wf)

            bev_scale = self.projector(feat, intrinsics, car_to_cam)
            bev_feats_scales.append(bev_scale)

        bev_feats = torch.cat(bev_feats_scales, dim=1)

        rover_emb = self.rover_embedding(rover_idx)
        rover_feat = self.rover_proj(rover_emb)
        rover_spatial = rover_feat.unsqueeze(-1).unsqueeze(-1)
        rover_spatial = rover_spatial.expand(-1, -1, BEV_HEIGHT, BEV_WIDTH)

        bev_feats = torch.cat([bev_feats, rover_spatial], dim=1)

        logits = self.bev_decoder(bev_feats)
        return logits


def make_gaussian_kernel(
    kernel_size: int, sigma: float, device: torch.device
) -> torch.Tensor:
    ax = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2.0
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def gaussian_supervision_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_value: int = IGNORE_VALUE,
    kernel_size: int = 7,
    sigma: float = 2.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    device = logits.device
    B, _, H, W = logits.shape

    valid = (target != ignore_value) & (target >= 0)
    valid_f = valid.float().unsqueeze(1)

    hard = (target == 1).float().unsqueeze(1)
    hard = hard * valid_f

    kernel = make_gaussian_kernel(kernel_size, sigma, device)
    soft = F.conv2d(hard, kernel, padding=kernel_size // 2).clamp(0.0, 1.0)

    loss_hard = F.binary_cross_entropy_with_logits(
        logits,
        hard,
        reduction="none",
    )
    loss_hard = (loss_hard * valid_f).sum() / (valid_f.sum() + 1e-6)

    loss_soft = F.binary_cross_entropy_with_logits(
        logits,
        soft,
        reduction="none",
    )
    loss_soft = (loss_soft * valid_f).sum() / (valid_f.sum() + 1e-6)

    return alpha * loss_hard + (1.0 - alpha) * loss_soft


@torch.no_grad()
def compute_iou_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_value: int = IGNORE_VALUE,
    threshold: float = 0.5,
) -> float:
    with torch.no_grad():
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs >= threshold).long()

        valid = (targets != ignore_value) & (targets >= 0)
        if valid.sum() == 0:
            return 0.0

        preds = preds[valid]
        t = targets[valid].clamp(0, 1)

        ious = []
        for c in (0, 1):
            pred_c = preds == c
            target_c = t == c

            inter = (pred_c & target_c).sum().float()
            union = (pred_c | target_c).sum().float()

            if union > 0:
                iou_c = inter / (union + 1e-6)
                ious.append(iou_c)
            else:
                pass

        if not ious:
            return 0.0
        return float(torch.stack(ious).mean().item())


class StaticGridDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: Path,
        val_dir: Path,
        test_dir: Optional[Path],
        batch_size: int,
        num_workers: int,
        use_augment: bool = True,
    ):
        super().__init__()
        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.test_dir = Path(test_dir) if test_dir is not None else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_augment = use_augment
        
        self.rover2idx: Optional[Dict[str, int]] = None
        self.num_rovers: int = 0

    def _build_global_rover2idx(self) -> Dict[str, int]:
        all_rovers = set()
        
        train_info = pd.read_csv(self.train_dir / "info.csv", index_col=0)
        all_rovers.update(train_info[ROVER_COL].unique().tolist())
        
        val_info = pd.read_csv(self.val_dir / "info.csv", index_col=0)
        all_rovers.update(val_info[ROVER_COL].unique().tolist())
        
        if self.test_dir is not None:
            test_info = pd.read_csv(self.test_dir / "info.csv", index_col=0)
            all_rovers.update(test_info[ROVER_COL].unique().tolist())
        
        rover2idx = {r: i for i, r in enumerate(sorted(all_rovers))}
        print(f"[DataModule] Found {len(rover2idx)} unique rovers: {list(rover2idx.keys())}")
        return rover2idx

    def setup(self, stage: Optional[str] = None):
        if self.rover2idx is None:
            self.rover2idx = self._build_global_rover2idx()
            self.num_rovers = len(self.rover2idx)
        
        if stage in (None, "fit"):
            self.train_dataset = StaticGridDataset(
                self.train_dir,
                mode="train",
                use_augment=self.use_augment,
                rover2idx=self.rover2idx,
            )
            self.val_dataset = StaticGridDataset(
                self.val_dir,
                mode="val",
                use_augment=False,
                rover2idx=self.rover2idx,
            )
        if stage in (None, "test") and self.test_dir is not None:
            self.test_dataset = StaticGridDataset(
                self.test_dir,
                mode="test",
                use_augment=False,
                rover2idx=self.rover2idx,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class GeomBEVLightningModule(pl.LightningModule):
    def __init__(
        self,
        encoder_name: str = "resnet18",
        encoder_pretrained: bool = True,
        lr: float = 5e-5,
        weight_decay: float = 5e-4,
        thresholds: Optional[List[float]] = None,
        use_gaussian_loss: bool = True,
        gaussian_kernel_size: int = 7,
        gaussian_sigma: float = 1.5,
        gaussian_alpha: float = 0.5,
        num_rovers: int = 32,
        rover_embed_dim: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = GeomBEVStaticOccModel(
            encoder_name=encoder_name,
            encoder_pretrained=encoder_pretrained,
            num_cams=len(CAMERA_NAMES),
            num_rovers=num_rovers,
            rover_embed_dim=rover_embed_dim,
        )

        self.thresholds = thresholds or [0.4, 0.5, 0.6]
        self.use_gaussian_loss = use_gaussian_loss
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_alpha = gaussian_alpha

    def forward(self, images, intrinsics, car2cam, rover_idx):
        return self.model(images, intrinsics, car2cam, rover_idx)

    def _loss(self, logits, gt):
        if self.use_gaussian_loss:
            return gaussian_supervision_loss(
                logits,
                gt,
                ignore_value=IGNORE_VALUE,
                kernel_size=self.gaussian_kernel_size,
                sigma=self.gaussian_sigma,
                alpha=self.gaussian_alpha,
            )
        else:
            logits_flat = logits.squeeze(1)
            valid = (gt != IGNORE_VALUE) & (gt >= 0)
            t = gt.clamp(0, 1).float()
            loss = F.binary_cross_entropy_with_logits(
                logits_flat,
                t,
                reduction="none",
            )
            loss = (loss * valid.float()).sum() / (valid.float().sum() + 1e-6)
            return loss

    def training_step(self, batch, batch_idx):
        images, intrinsics, car2cam, gt, rover_idx = batch
        logits = self(images, intrinsics, car2cam, rover_idx)
        loss = self._loss(logits, gt)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            for thr in self.thresholds:
                iou = compute_iou_from_logits(
                    logits, gt, threshold=thr, ignore_value=IGNORE_VALUE
                )
                self.log(
                    f"train_iou_t{int(thr * 100)}",
                    torch.tensor(iou, device=self.device),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=(thr == 0.5),
                )
        return loss

    def validation_step(self, batch, batch_idx):
        images, intrinsics, car2cam, gt, rover_idx = batch
        logits = self(images, intrinsics, car2cam, rover_idx)
        loss = self._loss(logits, gt)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        for thr in self.thresholds:
            iou = compute_iou_from_logits(
                logits, gt, threshold=thr, ignore_value=IGNORE_VALUE
            )
            self.log(
                f"val_iou_t{int(thr * 100)}",
                torch.tensor(iou, device=self.device),
                on_step=False,
                on_epoch=True,
                prog_bar=(thr == 0.5),
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


@torch.no_grad()
def run_test_inference(
    core_model: GeomBEVStaticOccModel,
    test_dir: Path,
    device: torch.device,
    exp_dir: Path,
    threshold: float = 0.5,
    rover2idx: Optional[Dict[str, int]] = None,
):
    core_model.eval().to(device)

    test_dataset = StaticGridDataset(
        test_dir, 
        mode="test", 
        use_augment=False,
        rover2idx=rover2idx,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    bin_dir = exp_dir / "predicted_static_grids"
    prob_dir = exp_dir / "predicted_probs"
    bin_dir.mkdir(parents=True, exist_ok=True)
    prob_dir.mkdir(parents=True, exist_ok=True)

    for images, intrinsics, car2cam, pred_path, rover_idx in test_loader:
        images = images.to(device)
        intrinsics = intrinsics.to(device)
        car2cam = car2cam.to(device)
        rover_idx = rover_idx.to(device)

        logits = core_model(images, intrinsics, car2cam, rover_idx)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        pred_bin = (probs >= threshold).astype(np.uint8)[None, :, :]

        original_pred_path = Path(pred_path[0])
        fname = original_pred_path.name

        out_bin_path = bin_dir / fname
        out_prob_path = prob_dir / fname

        np.save(out_bin_path, pred_bin.astype(np.uint8))
        np.save(out_prob_path, probs.astype(np.float32))


@torch.no_grad()
def find_best_threshold(
    core_model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    thresholds: Optional[List[float]] = None,
) -> float:
    core_model.eval().to(device)

    if thresholds is None:
        thresholds = [i / 20.0 for i in range(1, 20)]

    best_thr = 0.5
    best_iou = -1.0

    for thr in thresholds:
        total_iou = 0.0
        n_samples = 0

        for batch in val_loader:
            images, intrinsics, car2cams, grid, rover_idx = batch
            images = images.to(device)
            intrinsics = intrinsics.to(device)
            car2cams = car2cams.to(device)
            grid = grid.to(device)
            rover_idx = rover_idx.to(device)

            logits = core_model(images, intrinsics, car2cams, rover_idx)
            iou = compute_iou_from_logits(logits, grid, threshold=thr)
            bs = images.shape[0]
            total_iou += iou * bs
            n_samples += bs

        mean_iou = total_iou / max(n_samples, 1)
        print(f"Threshold {thr:.2f}: val mIoU = {mean_iou:.4f}")
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_thr = thr

    print(f"[VAL] Best threshold = {best_thr:.2f} with mIoU = {best_iou:.4f}")
    return best_thr


def load_best_module_from_checkpoint(
    ckpt_path: Path,
    encoder_name: str,
    encoder_pretrained: bool,
    lr: float,
    weight_decay: float,
    thresholds: List[float],
    use_gaussian_loss: bool,
    gaussian_kernel_size: int,
    gaussian_sigma: float,
    gaussian_alpha: float,
    num_rovers: int,
    rover_embed_dim: int,
    device: torch.device,
) -> GeomBEVLightningModule:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"]

    keys_to_drop = [
        k for k in state.keys() if "projector.X_grid" in k or "projector.Y_grid" in k
    ]
    print("Dropping keys from checkpoint:", keys_to_drop)
    for k in keys_to_drop:
        state.pop(k)

    lit_module_new = GeomBEVLightningModule(
        encoder_name=encoder_name,
        encoder_pretrained=encoder_pretrained,
        lr=lr,
        weight_decay=weight_decay,
        thresholds=thresholds,
        use_gaussian_loss=use_gaussian_loss,
        gaussian_kernel_size=gaussian_kernel_size,
        gaussian_sigma=gaussian_sigma,
        gaussian_alpha=gaussian_alpha,
        num_rovers=num_rovers,
        rover_embed_dim=rover_embed_dim,
    )

    missing_unexpected = lit_module_new.load_state_dict(state, strict=False)
    print("Missing keys:", missing_unexpected.missing_keys)
    print("Unexpected keys:", missing_unexpected.unexpected_keys)

    lit_module_new.to(device)
    lit_module_new.eval()
    return lit_module_new


def main():
    args = parse_args()

    train_root = Path("aidao/autonomy_yandex_dataset_train")
    val_root = Path("aidao/autonomy_yandex_dataset_val")
    test_root = Path("aidao/autonomy_yandex_dataset_test")

    exp_root = Path(args.output_root)
    exp_dir = exp_root / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    data_module = StaticGridDataModule(
        train_dir=train_root,
        val_dir=val_root,
        test_dir=test_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augment=True,
    )
    
    data_module.setup("fit")
    num_rovers = data_module.num_rovers
    print(f"[Main] Using {num_rovers} rovers with embed_dim={args.rover_embed_dim}")

    lit_module = GeomBEVLightningModule(
        encoder_name=args.encoder_name,
        encoder_pretrained=args.encoder_pretrained,
        lr=args.lr,
        weight_decay=args.weight_decay,
        thresholds=[0.4, 0.5, 0.6],
        use_gaussian_loss=args.use_gaussian_loss,
        gaussian_kernel_size=args.gaussian_kernel_size,
        gaussian_sigma=args.gaussian_sigma,
        gaussian_alpha=args.gaussian_alpha,
        num_rovers=num_rovers,
        rover_embed_dim=args.rover_embed_dim,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=args.patience,
        min_delta=0.001,
    )

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-epoch{epoch:02d}-val_loss{val_loss:.4f}",
        dirpath=exp_dir / "checkpoints",
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=True,
        log_every_n_steps=10,
        val_check_interval=args.val_check_interval,
        callbacks=[early_stop, checkpoint],
        default_root_dir=str(exp_dir),
        accumulate_grad_batches=args.grad_accum,
    )

    trainer.fit(lit_module, datamodule=data_module)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_ckpt_path = Path(checkpoint.best_model_path)

    print(f"Best checkpoint: {best_ckpt_path}")

    best_module = load_best_module_from_checkpoint(
        ckpt_path=best_ckpt_path,
        encoder_name=args.encoder_name,
        encoder_pretrained=args.encoder_pretrained,
        lr=args.lr,
        weight_decay=args.weight_decay,
        thresholds=[0.4, 0.5, 0.6],
        use_gaussian_loss=args.use_gaussian_loss,
        gaussian_kernel_size=args.gaussian_kernel_size,
        gaussian_sigma=args.gaussian_sigma,
        gaussian_alpha=args.gaussian_alpha,
        num_rovers=num_rovers,
        rover_embed_dim=args.rover_embed_dim,
        device=device,
    )
    core_model = best_module.model

    val_loader = data_module.val_dataloader()
    thr_grid = [0.35, 0.4, 0.45, 0.5]
    best_thr = find_best_threshold(core_model, val_loader, device, thresholds=thr_grid)

    with (exp_dir / "best_threshold.txt").open("w") as f:
        f.write(f"{best_thr:.4f}\n")
    
    import json
    with (exp_dir / "rover2idx.json").open("w") as f:
        json.dump(data_module.rover2idx, f, indent=2)

    if test_root is not None:
        run_test_inference(
            core_model=core_model,
            test_dir=test_root,
            device=device,
            exp_dir=exp_dir,
            threshold=best_thr,
            rover2idx=data_module.rover2idx,
        )


if __name__ == "__main__":
    main()