import numpy as np
_HAS_TORCH=False
try:
    import torch, torch.nn as nn
    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net=nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,1,3,padding=1))
        def forward(self,x): return self.net(x)
    _dev="cuda" if torch.cuda.is_available() else "cpu"
    _model=_Net().to(_dev); _model.eval(); _HAS_TORCH=True
except Exception: _HAS_TORCH=False

def run_denoise(image, strength=0.7, iterations=1):
    strength=float(np.clip(strength,0,1)); img=image.astype(np.float32)
    for _ in range(max(1,int(iterations))):
        if _HAS_TORCH:
            import torch
            gray=img if img.ndim==2 else img.mean(2)
            t=torch.tensor(gray).unsqueeze(0).unsqueeze(0).float().to(_dev)
            with torch.no_grad(): out=_model(t)
            d=out.cpu().numpy()[0,0]
            if img.ndim==3: d=np.stack([d]*img.shape[2],2)
        else:
            import cv2
            if img.ndim==2:
                img8=(img*255).clip(0,255).astype(np.uint8)
                d=cv2.bilateralFilter(img8,9,50,50).astype(np.float32)/255
            else:
                img8=(img*255).clip(0,255).astype(np.uint8)
                d=cv2.bilateralFilter(img8,9,50,50).astype(np.float32)/255
        img=(1-strength)*img+strength*d
    return np.clip(img,0,1).astype(np.float32)
