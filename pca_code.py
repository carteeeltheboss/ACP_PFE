

from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, color
from skimage.util import img_as_ubyte
from sklearn.decomposition import PCA

 
IMG_PATH = "original.jpg"          
OUT_DIR  = Path("images")
K_LIST   = [100, 50, 25, 10]       
OUT_DIR.mkdir(exist_ok=True)
 

 
img_raw = io.imread(IMG_PATH)
if img_raw.ndim == 2:
    img_gray = img_as_float(img_raw)
elif img_raw.shape[2] == 4:
    img_gray = color.rgb2gray(color.rgba2rgb(img_as_float(img_raw)))
elif img_raw.shape[2] == 3:
    img_gray = color.rgb2gray(img_as_float(img_raw))
else:
    raise ValueError(f"Format non géré : {img_raw.shape}")

h, w = img_gray.shape
io.imsave(OUT_DIR / "original.png", img_as_ubyte(img_gray), check_contrast=False)

 
k_max = K_LIST[0]
if k_max > min(h, w):
    k_max = min(h, w)
    K_LIST = [k for k in K_LIST if k <= k_max]

pca = PCA(n_components=k_max, svd_solver="randomized")
X_t = pca.fit_transform(img_gray)

 
def psnr(orig, rec):
    mse = np.mean((orig - rec) ** 2)
    return 20 * np.log10(1.0 / np.sqrt(mse))

sizes_kb, psnr_vals = [], []
for k in K_LIST:
    Xk  = X_t[:, :k]
    rec = np.clip((Xk @ pca.components_[:k, :]) + pca.mean_, 0, 1)
    path = OUT_DIR / f"pca_{k}.png"
    io.imsave(path, img_as_ubyte(rec), check_contrast=False)
    sizes_kb.append(os.path.getsize(path) / 1024)
    psnr_vals.append(psnr(img_gray, rec))

 
plt.figure(figsize=(6, 4))

 
line_size, = plt.plot(K_LIST, sizes_kb, "o-", label="Taille (Ko)")
plt.xlabel("Nombre de composantes principales $k$ (gauche → droite : $k$ décroît)")
plt.ylabel("Taille du fichier (Ko)")

 
ax2 = plt.gca().twinx()
line_psnr, = ax2.plot(K_LIST, psnr_vals, "r--s", label="PSNR (dB)")
ax2.set_ylabel("PSNR (dB)")

 
plt.xlim(max(K_LIST), min(K_LIST))
ax2.set_xlim(max(K_LIST), min(K_LIST))

 
plt.legend([line_size, line_psnr],
           [line_size.get_label(), line_psnr.get_label()],
           loc="upper right")

plt.title("Compression par ACP : ↓k ⇒ ↓taille, ↓PSNR")
plt.tight_layout()
plt.savefig(OUT_DIR / "pca_storage_vs_quality.png", dpi=300)
plt.close()

print("✅ Terminé – fichiers créés dans :", OUT_DIR.resolve())
