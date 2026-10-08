"""
Generates a synthetic chest-X-ray-like image dataset for TwinVerse (Review-2 prototype).

IMPORTANT (documented per project guidance):
No real chest X-ray dataset is bundled with this environment/network. To keep the
end-to-end pipeline (upload -> preprocess -> CNN -> prediction -> Grad-CAM) fully
functional for a live demo, this script procedurally generates grayscale images that
mimic the coarse structure of a chest X-ray (a lung-field oval with rib-like texture)
and injects a localized "opacity" patch to represent an abnormal finding.

This is a stand-in for a real dataset (e.g. NIH ChestX-ray14 / CheXpert) so the CNN
+ Grad-CAM machinery can be demonstrated. Swap DATA_DIR for a real, licensed dataset
before any clinical claim is made — see README for instructions.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

RNG = np.random.default_rng(7)
HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "xray_synth")
IMG_SIZE = 128
N_PER_CLASS = 300


def base_chest_image():
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=15)
    draw = ImageDraw.Draw(img)
    # lung field (bright oval, chest cavity)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2 + 6
    draw.ellipse([cx - 46, cy - 52, cx + 46, cy + 52], fill=90)
    # spine shadow
    draw.rectangle([cx - 6, cy - 55, cx + 6, cy + 55], fill=55)
    # rib-like texture (arcs)
    for i in range(-4, 5):
        y0 = cy + i * 12
        draw.arc([cx - 44, y0 - 10, cx + 44, y0 + 30], start=200, end=340, fill=70, width=2)
    arr = np.array(img).astype(np.float32)
    arr += RNG.normal(0, 6, arr.shape)
    return arr, (cx, cy)


def add_opacity(arr, cx, cy):
    """Add a localized bright/dense patch to simulate an abnormal finding."""
    ox = cx + int(RNG.integers(-28, 28))
    oy = cy + int(RNG.integers(-30, 30))
    radius = int(RNG.integers(8, 16))
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    mask = ((xx - ox) ** 2 + (yy - oy) ** 2) <= radius ** 2
    intensity = RNG.uniform(35, 60)
    arr = arr.copy()
    arr[mask] += intensity
    return arr, (ox, oy, radius)


def save_img(arr, path):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr, mode="L").filter(ImageFilter.GaussianBlur(0.6))
    im.save(path)


def main():
    for cls in ["normal", "abnormal"]:
        os.makedirs(os.path.join(OUT_DIR, cls), exist_ok=True)

    for i in range(N_PER_CLASS):
        arr, (cx, cy) = base_chest_image()
        save_img(arr, os.path.join(OUT_DIR, "normal", f"normal_{i:04d}.png"))

    for i in range(N_PER_CLASS):
        arr, (cx, cy) = base_chest_image()
        arr, _ = add_opacity(arr, cx, cy)
        save_img(arr, os.path.join(OUT_DIR, "abnormal", f"abnormal_{i:04d}.png"))

    print(f"Saved {N_PER_CLASS} normal + {N_PER_CLASS} abnormal synthetic images to {OUT_DIR}")


if __name__ == "__main__":
    main()
