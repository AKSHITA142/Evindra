#!/usr/bin/env python3
"""Recolor the blue-teal Evidra logos to the neon-lime brand hue.
Preserves 3D shading (value) and alpha; rotates every saturated pixel to lime.
"""
import sys
from PIL import Image

LIME_HUE = 60  # yellow-lime, closer to #76FF03 (hue ~85deg)

def colorize(src, dst, sat_boost=1.12, val_boost=1.0, hue=LIME_HUE):
    im = Image.open(src).convert("RGBA")
    r, g, b, a = im.split()
    hsv = Image.merge("RGB", (r, g, b)).convert("HSV")
    h, s, v = hsv.split()
    h = h.point(lambda _: int(hue))
    if sat_boost != 1.0:
        s = s.point(lambda x: min(255, int(x * sat_boost)))
    if val_boost != 1.0:
        v = v.point(lambda x: min(255, int(x * val_boost)))
    out = Image.merge("HSV", (h, s, v)).convert("RGB")
    r2, g2, b2 = out.split()
    Image.merge("RGBA", (r2, g2, b2, a)).save(dst)
    print("wrote", dst)

if __name__ == "__main__":
    args = sys.argv[1:]
    # args: src dst [sat] [val]
    src, dst = args[0], args[1]
    sat = float(args[2]) if len(args) > 2 else 1.12
    val = float(args[3]) if len(args) > 3 else 1.0
    colorize(src, dst, sat, val)
