import torch
from torch.utils.data import Dataset
import glob
import cv2
import numpy as np


def make_image_grid(w, h):
    grid_y, grid_x = np.linspace(-1, 1, w), np.linspace(-1, 1, h)
    y, x = np.meshgrid(grid_y, grid_x)
    grid = np.stack((y, x), axis=-1).astype(np.float32)
    return torch.from_numpy(grid)


def load_image(image_path: str, w: int, h: int):
    input_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    return cv2.resize(
        input_image,
        (w, h),
        interpolation=cv2.INTER_AREA,
    )


def compute_dense_optical_flow(images):
    flow_maps = []
    prev_frame = cv2.cvtColor(images[0], cv2.COLOR_RGB2GRAY)
    for _, image in enumerate(images[1:], start=1):
        next_frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.1, 0
        )
        # Normalize flow
        flow[..., 0] /= image.shape[0]
        flow[..., 1] /= image.shape[1]
        flow_maps.append(flow)
        prev_frame = next_frame
    flow_maps.append(flow)
    return np.array(flow_maps)


class VideoDataset(Dataset):
    def __init__(self, root_dir, w: int, h: int):
        all_images_path = sorted(glob.glob(f"{root_dir}/*"))
        self.all_images = [load_image(ip, w, h) for ip in all_images_path]
        self.flow = torch.from_numpy(compute_dense_optical_flow(self.all_images))
        self.all_images = torch.from_numpy(
            np.array([(i).astype(np.float32) / 255.0 for i in self.all_images])
        )
        self.grid = make_image_grid(w, h)
        self.ts_w = torch.linspace(0, 1, len(self.all_images)).unsqueeze(-1)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        return {
            "rgbs": self.all_images[idx],
            "grid": self.grid,
            "ts_w": self.ts_w[idx],
            "flow": self.flow[idx],
            "idxs": idx,
        }
