"""
Builds a dataset:  500 real, 500 AI-generated, 500 edited images.

Usage:
    python build_dataset.py --real_pool ./real_photos --ai_pool ./ai_images --out ./dataset

Requirements:  pip install pillow numpy

Inputs:
    real_pool : folder with >= 1000 real photos (500 stay "real", 500 get edited)
    ai_pool   : folder with >= 500 AI-generated images

Output:
    dataset/real/       500 untouched real images
    dataset/ai/         500 AI-generated images
    dataset/edited/     500 edited versions of DIFFERENT real photos
    dataset/labels.csv  filename, label, edit_type
"""
import argparse, csv, io, random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

N = 500
SIZE = (512, 512)
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS)


def load(path):
    return Image.open(path).convert("RGB")


def normalize(img):
    # Same size and same PNG format for every class, so the model can't
    # "cheat" by learning file format / resolution differences.
    return ImageOps.fit(img, SIZE, Image.LANCZOS)


# ---------- edit functions ----------
def jpeg_compress(img):
    q = random.choice([10, 20, 30, 40, 50])
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB"), f"jpeg_q{q}"


def blur(img):
    r = random.uniform(1, 3)
    return img.filter(ImageFilter.GaussianBlur(r)), "gaussian_blur"


def sharpen(img):
    return img.filter(ImageFilter.UnsharpMask(radius=3, percent=200)), "sharpen"


def resize_cycle(img):
    f = random.choice([0.25, 0.4, 0.5])
    small = img.resize((int(img.width * f), int(img.height * f)), Image.BILINEAR)
    return small.resize(img.size, Image.BICUBIC), "resize_down_up"


def brightness_contrast(img):
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.5))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.6))
    return img, "brightness_contrast"


def color_filter(img):
    choice = random.choice(["grayscale", "sepia", "saturate"])
    if choice == "grayscale":
        return ImageOps.grayscale(img).convert("RGB"), "filter_grayscale"
    if choice == "sepia":
        a = np.asarray(img, dtype=np.float32)
        m = np.array([[.393, .769, .189], [.349, .686, .168], [.272, .534, .131]])
        return Image.fromarray(np.clip(a @ m.T, 0, 255).astype(np.uint8)), "filter_sepia"
    return ImageEnhance.Color(img).enhance(2.2), "filter_saturate"


def add_noise(img):
    a = np.asarray(img, dtype=np.float32)
    a += np.random.normal(0, random.uniform(8, 25), a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), "gaussian_noise"


def crop_rescale(img):
    w, h = img.size
    s = random.uniform(0.6, 0.85)
    cw, ch = int(w * s), int(h * s)
    x, y = random.randint(0, w - cw), random.randint(0, h - ch)
    return img.crop((x, y, x + cw, y + ch)).resize(img.size, Image.LANCZOS), "crop_rescale"


def splice(img, donors):
    # paste a patch from another real photo -> simple copy-paste forgery
    donor = normalize(load(random.choice(donors)))
    pw, ph = random.randint(80, 200), random.randint(80, 200)
    dx, dy = random.randint(0, donor.width - pw), random.randint(0, donor.height - ph)
    patch = donor.crop((dx, dy, dx + pw, dy + ph))
    x, y = random.randint(0, img.width - pw), random.randint(0, img.height - ph)
    out = img.copy()
    out.paste(patch, (x, y))
    return out, "splice_patch"


SINGLE_EDITS = [jpeg_compress, blur, sharpen, resize_cycle,
                brightness_contrast, color_filter, add_noise, crop_rescale]


def edit_image(img, donors):
    # 70%: one edit, 30%: two chained edits, plus occasional splice
    if random.random() < 0.15:
        return splice(img, donors)
    n = 1 if random.random() < 0.7 else 2
    names = []
    for fn in random.sample(SINGLE_EDITS, n):
        img, name = fn(img)
        names.append(name)
    return img, "+".join(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_pool", required=True)
    ap.add_argument("--ai_pool", required=True)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    real_files = list_images(args.real_pool)
    ai_files = list_images(args.ai_pool)
    assert len(real_files) >= 2 * N, f"Need >= {2*N} real images, found {len(real_files)}"
    assert len(ai_files) >= N, f"Need >= {N} AI images, found {len(ai_files)}"

    random.shuffle(real_files)
    random.shuffle(ai_files)
    real_sel = real_files[:N]
    edit_src = real_files[N:2 * N]   # different photos -> no leakage between classes
    ai_sel = ai_files[:N]

    out = Path(args.out)
    for d in ("real", "ai", "edited"):
        (out / d).mkdir(parents=True, exist_ok=True)

    rows = [("filename", "label", "edit_type")]

    for i, p in enumerate(real_sel):
        name = f"real_{i:04d}.png"
        normalize(load(p)).save(out / "real" / name)
        rows.append((f"real/{name}", "real", "none"))

    for i, p in enumerate(ai_sel):
        name = f"ai_{i:04d}.png"
        normalize(load(p)).save(out / "ai" / name)
        rows.append((f"ai/{name}", "ai_generated", "none"))

    for i, p in enumerate(edit_src):
        name = f"edited_{i:04d}.png"
        img, edit_type = edit_image(normalize(load(p)), real_files)
        img.save(out / "edited" / name)
        rows.append((f"edited/{name}", "edited", edit_type))

    with open(out / "labels.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"Done: {N} real, {N} AI, {N} edited -> {out.resolve()}")


if __name__ == "__main__":
    main()
