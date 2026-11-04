from PIL import Image
import numpy as np
from pixelart.postprocess import (PICO8, prepare_input, to_grid, quantize,
                                  upscale_preview, tile_preview)


def opaque_colors(img):
    a = np.array(img.convert("RGBA")).reshape(-1, 4)
    return {tuple(int(c) for c in p[:3]) for p in a if p[3] == 255}


def noise(w, h, mode="RGB"):
    arr = np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr).convert(mode)


def test_to_grid_square():
    assert to_grid(noise(1024, 1024), 32).size == (32, 32)


def test_to_grid_keeps_aspect():
    assert to_grid(noise(1024, 512), 64).size == (64, 32)


def test_to_grid_extreme_aspect():
    assert to_grid(noise(2000, 50), 16).size == (16, 1)


def test_prepare_input_modes():
    for mode in ("RGBA", "L", "P"):
        out = prepare_input(noise(300, 200, mode))
        assert out.mode == "RGB" and out.size == (1024, 680)  # 682.67 rounded to a multiple of 8


def test_prepare_input_extreme_aspect():
    w, h = prepare_input(noise(2000, 50)).size
    assert w == 1024 and h >= 8 and h % 8 == 0


def test_quantize_color_limit():
    assert len(opaque_colors(quantize(noise(64, 64), 16))) <= 16


def test_quantize_few_colors():
    assert len(opaque_colors(quantize(Image.new("RGB", (8, 8), "red"), 16))) == 1


def test_quantize_alpha_binary():
    img = noise(32, 32, "RGBA")
    img.putalpha(Image.linear_gradient("L").resize((32, 32)))
    assert set(np.array(quantize(img, 8))[..., 3].ravel()) <= {0, 255}


def test_quantize_fully_transparent():
    out = quantize(Image.new("RGBA", (16, 16), (0, 0, 0, 0)), 16)
    assert out.mode == "RGBA" and np.array(out)[..., 3].max() == 0


def test_quantize_pico8():
    assert opaque_colors(quantize(noise(64, 64), palette="pico8")) <= set(PICO8)


def test_upscale_preview():
    assert upscale_preview(Image.new("RGBA", (32, 32))).size == (512, 512)


def test_tile_preview_size():
    assert tile_preview(Image.new("RGB", (32, 16))).size == (96, 48)


def test_seamless_survives_to_grid():
    base = noise(32, 32)
    periodic = tile_preview(base).resize((3072, 3072), Image.NEAREST).crop((1024, 1024, 2048, 2048))
    small = to_grid(periodic, 32)  # BOX 1024->32: each 32x32 block maps back to one source cell
    assert np.array_equal(np.array(small), np.array(base))
