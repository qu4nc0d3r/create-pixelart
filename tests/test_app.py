import gradio as gr
import pytest
import torch
from PIL import Image

import app


def FAKE(*a, **k):
    return [Image.new("RGB", (1024, 1024), "green")] * a[-2], 7  # n is the second-to-last arg


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    for name in ("txt2img", "img2img", "tile"):
        monkeypatch.setattr(app.generate, name, FAKE)
    monkeypatch.setattr(app.postprocess, "remove_background", lambda im: im.convert("RGBA"))


def sizes(paths):
    return [Image.open(p).size for p in paths]


def test_sprite():
    previews, paths, seed = app.sprite("knight", 32, "Tự động", 16, True, 3, None)
    assert sizes(paths) == [(32, 32)] * 3 and len(previews) == 3 and seed == 7


def test_sprite_empty_prompt():
    with pytest.raises(gr.Error):
        app.sprite("  ", 32, "Tự động", 16, True, 1, None)


def test_convert_strength_zero_skips_model(monkeypatch):
    monkeypatch.setattr(app.generate, "img2img", lambda *a, **k: pytest.fail("model called"))
    _, paths, _ = app.convert(Image.new("RGBA", (200, 100)), "", 0, 64, "PICO-8", 16, False, 4, None)
    assert sizes(paths) == [(64, 32)]


def test_convert_no_image():
    with pytest.raises(gr.Error):
        app.convert(None, "", 0.5, 32, "Tự động", 16, False, 1, None)


def test_make_tile():
    previews, paths, _ = app.make_tile("grass", 16, "Tự động", 16, 2, None)
    assert sizes(paths) == [(16, 16)] * 2 and previews[0].size[0] == previews[0].size[1]


def test_oom_becomes_gr_error(monkeypatch):
    def boom(*a, **k):
        raise torch.cuda.OutOfMemoryError()

    monkeypatch.setattr(app.generate, "txt2img", boom)
    with pytest.raises(gr.Error):
        app.sprite("knight", 32, "Tự động", 16, False, 4, None)


def test_auth_missing(monkeypatch):
    monkeypatch.delenv("PIXELART_USER", raising=False)
    monkeypatch.setenv("PIXELART_PASS", "x")
    with pytest.raises(SystemExit):
        app.get_auth()


def test_auth_ok(monkeypatch):
    monkeypatch.setenv("PIXELART_USER", "u")
    monkeypatch.setenv("PIXELART_PASS", "p")
    assert app.get_auth() == ("u", "p")


def test_build_ui():
    assert isinstance(app.build_ui(), gr.Blocks)


def test_all_tabs_share_one_gpu_queue():
    ids = {fn.concurrency_id for fn in app.build_ui().fns.values()}
    assert len(ids) == 1


def test_convert_strength_zero_keeps_transparency():
    img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    img.putpixel((0, 0), (255, 0, 0, 255))
    _, paths, _ = app.convert(img, "", 0, 32, "Tự động", 16, False, 1, None)
    assert Image.open(paths[0]).getpixel((5, 5))[3] == 0


def test_load_enables_vae_slicing(monkeypatch):
    import sys, types
    from unittest import mock
    pipe = mock.MagicMock()
    fake = types.SimpleNamespace(
        AutoencoderKL=mock.MagicMock(),
        StableDiffusionXLPipeline=mock.MagicMock(**{"from_pretrained.return_value.to.return_value": pipe}),
        AutoPipelineForImage2Image=mock.MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "diffusers", fake)
    monkeypatch.setattr(app.generate, "_txt2img", None)
    monkeypatch.setattr(app.generate, "_img2img", None)
    import importlib
    gen = importlib.import_module("pixelart.generate")
    gen.load()
    pipe.vae.enable_slicing.assert_called_once()
