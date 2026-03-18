import numpy as np
from skimage.feature import blob_log

def image_stats(image):
    flat=image.ravel()
    return {"mean":float(np.mean(flat)),"median":float(np.median(flat)),
            "std":float(np.std(flat)),"min":float(flat.min()),"max":float(flat.max()),
            "snr":float(np.mean(flat)/(np.std(flat)+1e-9))}

def measure_fwhm(image):
    gray=image if image.ndim==2 else image.mean(2)
    stars=blob_log(gray,max_sigma=8,threshold=0.03)
    if len(stars)==0: return 0.0
    return float(np.mean([s[2] for s in stars])*2.35)
