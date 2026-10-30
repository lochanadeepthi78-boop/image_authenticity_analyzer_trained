import os, sys, numpy as np
from pathlib import Path
from PIL import Image

BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(BASE))
import analyzer

def features(path):
    img = Image.open(path).convert("RGB")
    ela = analyzer.error_level_analysis(img)
    noise = analyzer.analyze_noise(img)
    freq = analyzer.analyze_frequency_spectrum(img)
    dct = analyzer.analyze_dct_artifacts(img)
    _, _, _, tex = analyzer.analyze_ai_generated(noise, freq, dct, img, False)
    return np.array([noise["noise_std"], noise["noise_mean"], noise["edge_density_pct"],
        freq["low_freq_pct"], freq["mid_freq_pct"], freq["high_freq_pct"],
        freq["spectral_flatness"], dct["grid_strength_coeff"], tex["entropy"],
        tex["edge_sharpness"], tex["texture_uniformity"], tex["smoothness"],
        ela["mean"], ela["modified_pixels_pct"]], dtype=np.float64)

X=[]; y=[]
for label, folder in [(0, DATASET/"real"), (1, DATASET/"ai_generated")]:
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            X.append(features(p)); y.append(label)
X=np.asarray(X); y=np.asarray(y, dtype=np.float64)
if len(X) < 20 or len(set(y.tolist())) < 2:
    raise SystemExit("Need images in dataset/real and dataset/ai_generated before training.")
mean=X.mean(0); std=X.std(0); std[std < 1e-8]=1.0
Z=(X-mean)/std
w=np.zeros(Z.shape[1]); b=0.0
for epoch in range(2500):
    p=1/(1+np.exp(-np.clip(Z@w+b,-30,30)))
    gw=Z.T@(p-y)/len(y)+0.01*w; gb=(p-y).mean()
    lr=0.08*(0.98**(epoch//200))
    w-=lr*gw; b-=lr*gb
p=1/(1+np.exp(-np.clip(Z@w+b,-30,30)))
acc=float(((p>=.5)==y).mean()*100)
np.savez(MODEL_DIR/"ai_classifier.npz", mean=mean, std=std, weights=w, bias=b)
print(f"Trained model saved to {MODEL_DIR/'ai_classifier.npz'}")
print(f"Training accuracy: {acc:.1f}% on {len(y)} images")
