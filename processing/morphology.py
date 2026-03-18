import cv2, numpy as np

def morphological(image, operation="erosion", kernel_size=3, iterations=1, **kwargs):
    k=np.ones((int(kernel_size),int(kernel_size)),np.uint8)
    def op(ch):
        ch8=(ch*255).clip(0,255).astype(np.uint8)
        if operation=="erosion":   r=cv2.erode(ch8,k,iterations=int(iterations))
        elif operation=="dilation":r=cv2.dilate(ch8,k,iterations=int(iterations))
        elif operation=="opening": r=cv2.morphologyEx(ch8,cv2.MORPH_OPEN,k)
        elif operation=="closing": r=cv2.morphologyEx(ch8,cv2.MORPH_CLOSE,k)
        else:                      r=ch8
        return r.astype(np.float32)/255
    if image.ndim==2: return op(image)
    return np.stack([op(image[:,:,c]) for c in range(image.shape[2])],2).astype(np.float32)
