from __future__ import annotations

import torch
import torch.nn.functional as F


class MaskCropLPIPS:
    def __init__(self, device: torch.device | str, resize: int = 224, margin_ratio: float = 0.25) -> None:
        import lpips

        self.net = lpips.LPIPS(net="vgg").to(device).eval()
        self.resize = resize
        self.margin_ratio = margin_ratio
        for p in self.net.parameters():
            p.requires_grad_(False)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        b, _, h, w = pred.shape
        for i in range(b):
            ys, xs = torch.where(mask[i, 0] > 0.5)
            if ys.numel() == 0:
                continue
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            mh = max(y1 - y0, 1)
            mw = max(x1 - x0, 1)
            pad_y = int(mh * self.margin_ratio)
            pad_x = int(mw * self.margin_ratio)
            y0, y1 = max(0, y0 - pad_y), min(h, y1 + pad_y)
            x0, x1 = max(0, x0 - pad_x), min(w, x1 + pad_x)
            p = F.interpolate(pred[i : i + 1, :, y0:y1, x0:x1], (self.resize, self.resize), mode="bilinear", align_corners=False)
            t = F.interpolate(target[i : i + 1, :, y0:y1, x0:x1], (self.resize, self.resize), mode="bilinear", align_corners=False)
            losses.append(self.net(p * 2 - 1, t * 2 - 1).mean())
        if not losses:
            return pred.new_tensor(0.0)
        return torch.stack(losses).mean()
