"""Gradio UI for the internal pixel art tool. Run inside the Kaggle notebook: `python app.py`."""
import os
import sys
import tempfile

import gradio as gr
import torch

from pixelart import generate, postprocess

PALETTES = ["Tự động", "PICO-8"]


def _call(fn, *args):
    try:
        return fn(*args)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise gr.Error("Hết bộ nhớ GPU — giảm số biến thể rồi thử lại")


def _finish(images, size, palette_mode, colors, remove_bg):
    palette = "pico8" if palette_mode == "PICO-8" else None
    out_dir = tempfile.mkdtemp(prefix="pixelart-")
    sprites, paths = [], []
    for i, img in enumerate(images):
        if remove_bg:
            img = postprocess.remove_background(img)
        img = postprocess.quantize(postprocess.to_grid(img, int(size)), int(colors), palette)
        path = os.path.join(out_dir, f"pixelart_{i + 1}.png")
        img.save(path)
        sprites.append(img)
        paths.append(path)
    return sprites, paths


def _require_prompt(prompt):
    if not prompt or not prompt.strip():
        raise gr.Error("Hãy nhập mô tả")
    return prompt.strip()


def sprite(prompt, size, palette_mode, colors, remove_bg, n, seed):
    prompt = _require_prompt(prompt)
    images, seed = _call(generate.txt2img, prompt, int(n), seed)
    sprites, paths = _finish(images, size, palette_mode, colors, remove_bg)
    return [postprocess.upscale_preview(s) for s in sprites], paths, seed


def convert(image, prompt, strength, size, palette_mode, colors, remove_bg, n, seed):
    if image is None:
        raise gr.Error("Hãy tải ảnh lên")
    if strength == 0:
        images, seed = [postprocess.prepare_input(image)], 0  # pixelate only, no model
    else:
        images, seed = _call(generate.img2img, image, (prompt or "").strip(), float(strength), int(n), seed)
    sprites, paths = _finish(images, size, palette_mode, colors, remove_bg)
    return [postprocess.upscale_preview(s) for s in sprites], paths, seed


def make_tile(prompt, size, palette_mode, colors, n, seed):
    prompt = _require_prompt(prompt)
    images, seed = _call(generate.tile, prompt, int(n), seed)
    tiles, paths = _finish(images, size, palette_mode, colors, False)
    return [postprocess.upscale_preview(postprocess.tile_preview(t)) for t in tiles], paths, seed


def get_auth():
    user, password = os.environ.get("PIXELART_USER"), os.environ.get("PIXELART_PASS")
    if not user or not password:
        sys.exit("Thiếu PIXELART_USER/PIXELART_PASS trong Kaggle Secrets")
    return user, password


def _palette_controls():
    return (
        gr.Radio(PALETTES, value="Tự động", label="Bảng màu"),
        gr.Slider(4, 64, value=16, step=1, label="Số màu (khi Tự động)"),
    )


def _outputs():
    return [
        gr.Gallery(label="Xem trước", columns=4),
        gr.File(label="PNG kích thước thật", file_count="multiple"),
        gr.Number(label="Seed đã dùng", precision=0),
    ]


def _variants_and_seed():
    return (
        gr.Slider(1, 4, value=4, step=1, label="Số biến thể"),
        gr.Number(value=None, precision=0, label="Seed (để trống = ngẫu nhiên)"),
    )


def build_ui():
    with gr.Blocks(title="Pixel Art") as demo:
        with gr.Tab("Chữ → Sprite"):
            prompt = gr.Textbox(label="Mô tả", placeholder="knight with red cape")
            size = gr.Radio([16, 32, 64, 128], value=32, label="Kích thước (px)")
            palette, colors = _palette_controls()
            remove_bg = gr.Checkbox(value=True, label="Xóa nền")
            n, seed = _variants_and_seed()
            btn = gr.Button("Tạo", variant="primary")
            btn.click(sprite, [prompt, size, palette, colors, remove_bg, n, seed], _outputs())

        with gr.Tab("Ảnh → Pixel art"):
            image = gr.Image(type="pil", label="Ảnh gốc")
            prompt = gr.Textbox(label="Mô tả (tùy chọn)")
            strength = gr.Slider(0, 0.8, value=0.5, step=0.05, label="Độ biến đổi (0 = chỉ pixel hóa)")
            size = gr.Radio([16, 32, 64, 128], value=32, label="Kích thước cạnh dài (px)")
            palette, colors = _palette_controls()
            remove_bg = gr.Checkbox(value=False, label="Xóa nền")
            n, seed = _variants_and_seed()
            btn = gr.Button("Tạo", variant="primary")
            btn.click(convert, [image, prompt, strength, size, palette, colors, remove_bg, n, seed], _outputs())

        with gr.Tab("Tile liền mạch"):
            prompt = gr.Textbox(label="Mô tả", placeholder="grass with small flowers")
            size = gr.Radio([16, 32, 64], value=32, label="Kích thước (px)")
            palette, colors = _palette_controls()
            n, seed = _variants_and_seed()
            btn = gr.Button("Tạo", variant="primary")
            btn.click(make_tile, [prompt, size, palette, colors, n, seed], _outputs())
    return demo


def main():
    auth = get_auth()
    if not torch.cuda.is_available():
        sys.exit("Hãy bật GPU T4 trong Settings của notebook")
    generate.load()
    build_ui().queue(default_concurrency_limit=1).launch(share=True, auth=auth)


if __name__ == "__main__":
    main()
