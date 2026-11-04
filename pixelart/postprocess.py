"""Turn AI-generated images into real pixel art. Pure PIL, CPU only."""
import numpy as np
from PIL import Image

PICO8 = [
    (0x00, 0x00, 0x00), (0x1D, 0x2B, 0x53), (0x7E, 0x25, 0x53), (0x00, 0x87, 0x51),
    (0xAB, 0x52, 0x36), (0x5F, 0x57, 0x4F), (0xC2, 0xC3, 0xC7), (0xFF, 0xF1, 0xE8),
    (0xFF, 0x00, 0x4D), (0xFF, 0xA3, 0x00), (0xFF, 0xEC, 0x27), (0x00, 0xE4, 0x36),
    (0x29, 0xAD, 0xFF), (0x83, 0x76, 0x9C), (0xFF, 0x77, 0xA8), (0xFF, 0xCC, 0xAA),
]

_rembg_session = None


def _scaled(size, long_side):
    w, h = size
    scale = long_side / max(w, h)
    return w * scale, h * scale


def prepare_input(img, long_side=1024):
    w, h = _scaled(img.size, long_side)
    flat = Image.alpha_composite(Image.new("RGBA", img.size, "white"), img.convert("RGBA")).convert("RGB")
    return flat.resize((max(8, round(w / 8) * 8), max(8, round(h / 8) * 8)), Image.LANCZOS)


def remove_background(img):
    global _rembg_session
    import rembg  # lazy: importing this module must not download the u2net model

    if _rembg_session is None:
        _rembg_session = rembg.new_session("u2net")
    return rembg.remove(img.convert("RGB"), session=_rembg_session).convert("RGBA")


def to_grid(img, long_side):
    w, h = _scaled(img.size, long_side)
    return img.resize((max(1, round(w)), max(1, round(h))), Image.BOX)  # Pillow premultiplies RGBA


def quantize(img, colors=16, palette=None):
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A").point(lambda a: 255 if a >= 128 else 0)
    rgb = rgba.convert("RGB")
    if palette == "pico8":
        chosen = PICO8
    else:
        opaque = np.array(rgb)[np.array(alpha) > 0]  # transparent pixels must not take palette slots
        src = Image.fromarray(opaque[None]) if len(opaque) else rgb
        q = src.quantize(colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        used = sorted(i for _, i in q.getcolors())
        pal = q.getpalette()
        chosen = [tuple(pal[3 * i:3 * i + 3]) for i in used]
    q = rgb.quantize(palette=_palette_image(chosen), dither=Image.Dither.NONE)
    out = q.convert("RGB").convert("RGBA")
    out.putalpha(alpha)
    clear = Image.new("RGBA", out.size, (0, 0, 0, 0))
    return Image.composite(out, clear, alpha)


def _palette_image(colors):
    flat = [c for rgb in colors for c in rgb]
    pal = Image.new("P", (1, 1))
    pal.putpalette(flat + flat[:3] * (256 - len(colors)))  # pad with a real colour, never stray black
    return pal


def upscale_preview(img, size=512):
    k = max(1, size // max(img.size))
    return img.resize((img.width * k, img.height * k), Image.NEAREST)


def tile_preview(img):
    out = Image.new(img.mode, (img.width * 3, img.height * 3))
    for x in range(3):
        for y in range(3):
            out.paste(img, (x * img.width, y * img.height))
    return out
