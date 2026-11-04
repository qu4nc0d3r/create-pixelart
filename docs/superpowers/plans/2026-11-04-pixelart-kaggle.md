# Pixel Art Kaggle Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Một app Gradio chạy trong notebook Kaggle (GPU T4) tạo pixel art bằng SDXL + LoRA `nerijs/pixel-art-xl`, gồm 3 tab: Chữ → Sprite, Ảnh → Pixel art, Tile liền mạch.

**Architecture:** `pixelart/postprocess.py` chứa các hàm PIL thuần, chạy CPU, test được không cần GPU. `pixelart/generate.py` load pipeline SDXL một lần và lo việc sinh ảnh. `app.py` nối hai module trên thành 3 handler và giao diện Gradio có đăng nhập. Notebook Kaggle chỉ clone repo, cài thư viện và chạy `app.py`.

**Tech Stack:** Python 3.10+, Pillow, NumPy, torch (Kaggle cài sẵn), diffusers, transformers, accelerate, peft, gradio, rembg, pytest.

**Spec:** `docs/superpowers/specs/2026-11-04-pixelart-kaggle-design.md`

## Global Constraints

- Chỉ dùng model mở: `stabilityai/stable-diffusion-xl-base-1.0` (fp16), `nerijs/pixel-art-xl`, `madebyollin/sdxl-vae-fp16-fix`, rembg `u2net`. Không gọi API trả phí.
- Chỉ chạy trên GPU Kaggle. Không có nhánh code nào cho CPU khi sinh ảnh.
- Ảnh sinh ra: 1024×1024 (tab A, C); tab B có cạnh dài 1024, cạnh còn lại là bội của 8.
- Prompt luôn có tiền tố `pixel art, `. Negative mặc định: `blurry, 3d render, photo, realistic, gradient, noise`.
- Kích thước: tab A/B là 16/32/64/128, tab C là 16/32/64, mặc định 32. Số màu 4–64, mặc định 16. Số biến thể 1–4, mặc định 4. Độ biến đổi 0–0.8, mặc định 0.5.
- Bảng màu PICO-8: `000000 1D2B53 7E2553 008751 AB5236 5F574F C2C3C7 FFF1E8 FF004D FFA300 FFEC27 00E436 29ADFF 83769C FF77A8 FFCCAA`.
- Alpha ép về 0/255 với ngưỡng 128.
- Secrets: biến môi trường `PIXELART_USER`, `PIXELART_PASS`. Thiếu một trong hai thì app không khởi chạy.
- Nội dung hiển thị cho người dùng (nhãn, thông báo lỗi) viết bằng tiếng Việt, chép đúng từ spec:
  - "Hãy bật GPU T4 trong Settings của notebook"
  - "Hết bộ nhớ GPU — giảm số biến thể rồi thử lại"
  - "Hãy nhập mô tả"
  - "Hãy tải ảnh lên"
- Repo public: `https://github.com/qu4nc0d3r/create-pixelart`.

## Review Focus

1. **Ảnh tải lên ở tab B là RGBA, grayscale hoặc palette (P):** phải được chuyển sang RGB trước khi đưa vào img2img, không được crash. Test nằm trong Task 1 (`prepare_input`).
2. **Ảnh rất dẹt** (vd. 2000×50) khi thu nhỏ về cạnh dài 16: cạnh ngắn phải ≥ 1 px (`to_grid`) và ≥ 8 px (`prepare_input`). Test nằm trong Task 1.
3. **rembg xóa sạch nền, ảnh trong suốt hoàn toàn:** `quantize` không được crash và phải trả ảnh trong suốt. Test nằm trong Task 1.
4. **Ảnh có ít màu hơn N** (vd. ảnh một màu với N=16): `quantize` không crash, kết quả ≤ N màu. Test nằm trong Task 1.
5. **Ô seed của Gradio** trả `None`, số float (`42.0`) hoặc số âm: để trống hoặc âm thì lấy seed ngẫu nhiên, còn lại ép về int. Test nằm trong Task 2 (`resolve_seed`).

---

### Task 1: Hậu xử lý ảnh + khung dự án

**Files:**
- Create: `requirements.txt`, `pixelart/__init__.py` (rỗng), `pixelart/postprocess.py`, `tests/test_postprocess.py`, `.gitignore` (`__pycache__/`, `.pytest_cache/`)

**Interfaces:**
- Produces (trong `pixelart/postprocess.py`):
  - `PICO8: list[tuple[int, int, int]]` (16 màu, theo Global Constraints)
  - `prepare_input(img: Image.Image, long_side: int = 1024) -> Image.Image`: chuyển RGB, co cạnh dài về `long_side`, mỗi cạnh làm tròn về bội của 8 và tối thiểu 8
  - `remove_background(img: Image.Image) -> Image.Image`: trả RGBA
  - `to_grid(img: Image.Image, long_side: int) -> Image.Image`
  - `quantize(img: Image.Image, colors: int = 16, palette: str | None = None) -> Image.Image`: luôn trả RGBA; `palette="pico8"` thì bỏ qua `colors`
  - `upscale_preview(img: Image.Image, size: int = 512) -> Image.Image`
  - `tile_preview(img: Image.Image) -> Image.Image`

- [ ] **Step 1: Tạo `requirements.txt`**

```
diffusers>=0.30
transformers
accelerate
peft
gradio>=4.44
rembg
onnxruntime
pillow
numpy
pytest
```
Không ghi torch vì Kaggle đã cài sẵn. rembg chạy trên CPU (onnxruntime bản CPU) để tránh xung đột CUDA trên Kaggle.

Môi trường dev: `pip install -r requirements.txt && pip install torch --index-url https://download.pytorch.org/whl/cpu`

- [ ] **Step 2: Viết test fail trong `tests/test_postprocess.py`**

```python
from PIL import Image
import numpy as np
from pixelart.postprocess import (PICO8, prepare_input, to_grid, quantize,
                                  upscale_preview, tile_preview)

def opaque_colors(img):
    a = np.array(img.convert("RGBA")).reshape(-1, 4)
    return {tuple(p[:3]) for p in a if p[3] == 255}

def noise(w, h, mode="RGB"):
    return Image.fromarray(np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)).convert(mode)

def test_to_grid_square():            assert to_grid(noise(1024, 1024), 32).size == (32, 32)
def test_to_grid_keeps_aspect():      assert to_grid(noise(1024, 512), 64).size == (64, 32)
def test_to_grid_extreme_aspect():    assert to_grid(noise(2000, 50), 16).size == (16, 1)

def test_prepare_input_modes():
    for mode in ("RGBA", "L", "P"):
        out = prepare_input(noise(300, 200, mode))
        assert out.mode == "RGB" and out.size == (1024, 680)   # 682.67 làm tròn về bội 8

def test_prepare_input_extreme_aspect():
    w, h = prepare_input(noise(2000, 50)).size
    assert w == 1024 and h >= 8 and h % 8 == 0

def test_quantize_color_limit():      assert len(opaque_colors(quantize(noise(64, 64), 16))) <= 16
def test_quantize_few_colors():       assert len(opaque_colors(quantize(Image.new("RGB", (8, 8), "red"), 16))) == 1

def test_quantize_alpha_binary():
    img = noise(32, 32, "RGBA"); img.putalpha(Image.linear_gradient("L").resize((32, 32)))
    assert set(np.array(quantize(img, 8))[..., 3].ravel()) <= {0, 255}

def test_quantize_fully_transparent():
    out = quantize(Image.new("RGBA", (16, 16), (0, 0, 0, 0)), 16)
    assert out.mode == "RGBA" and np.array(out)[..., 3].max() == 0

def test_quantize_pico8():            assert opaque_colors(quantize(noise(64, 64), palette="pico8")) <= set(PICO8)

def test_upscale_preview():           assert upscale_preview(Image.new("RGBA", (32, 32))).size == (512, 512)

def test_tile_preview_size():         assert tile_preview(Image.new("RGB", (32, 16))).size == (96, 48)

def test_seamless_survives_to_grid():
    base = noise(32, 32)
    periodic = tile_preview(base).resize((3072, 3072), Image.NEAREST).crop((1024, 1024, 2048, 2048))
    small = to_grid(periodic, 32)                     # BOX 1024→32: mỗi khối 32×32 trùng một ô gốc
    assert np.array_equal(np.array(small), np.array(base))
```

- [ ] **Step 3: Chạy test để thấy fail**

Run: `pytest tests/test_postprocess.py -v`
Expected: FAIL với lỗi `ModuleNotFoundError` / `ImportError` cho `pixelart.postprocess`.

- [ ] **Step 4: Viết `pixelart/postprocess.py`**

- `to_grid`: giữ tỉ lệ, mỗi cạnh `max(1, round(...))`, dùng `Image.BOX`. Pillow tự premultiply alpha khi resize RGBA nên viền không bị lem màu.
- `prepare_input`: `convert("RGB")`, cạnh dài `long_side`, mỗi cạnh `max(8, round(x / 8) * 8)`, dùng `Image.LANCZOS`.
- `quantize`:
  - tách alpha (ảnh không có alpha thì coi alpha = 255), ép alpha về 0/255 với ngưỡng 128
  - phần RGB: tự động thì `rgb.quantize(colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)`; `pico8` thì `rgb.quantize(palette=<ảnh P có PICO8, phần còn lại của 256 màu lặp lại màu đầu>, dither=Image.Dither.NONE)`
  - đổi lại sang RGB, gắn alpha vào, pixel có alpha 0 thì đặt `(0, 0, 0, 0)`
- `remove_background`: `import rembg` bên trong hàm và giữ session `rembg.new_session("u2net")` trong biến module, chỉ tạo lần đầu. Mục đích: chỉ cần import module là không phải tải model. Gọi `rembg.remove(img.convert("RGB"), session=...)`. Hàm này không có unit test vì cần tải model.
- `upscale_preview`: hệ số `max(1, size // max(w, h))`, dùng `Image.NEAREST`.
- `tile_preview`: dán ảnh 3×3 vào ảnh mới cùng mode.

- [ ] **Step 5: Chạy test để thấy pass**

Run: `pytest tests/test_postprocess.py -v`
Expected: tất cả PASS.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore pixelart tests/test_postprocess.py
git commit -m "feat: add pixel art post-processing"
```

---

### Task 2: Sinh ảnh SDXL (`pixelart/generate.py`)

**Files:**
- Create: `pixelart/generate.py`, `tests/test_generate.py`

**Interfaces:**
- Consumes: `prepare_input` từ Task 1
- Produces (trong `pixelart/generate.py`):
  - Hằng số `BASE`, `LORA`, `VAE`, `PREFIX = "pixel art, "`, `NEGATIVE = "blurry, 3d render, photo, realistic, gradient, noise"`, `STEPS = 30`, `GUIDANCE = 7.0`
  - `resolve_seed(seed: float | int | None) -> int`
  - `seamless(*modules: torch.nn.Module)`: context manager
  - `load() -> None`
  - `txt2img(prompt: str, n: int, seed: float | int | None) -> tuple[list[Image.Image], int]`
  - `img2img(image: Image.Image, prompt: str, strength: float, n: int, seed: float | int | None) -> tuple[list[Image.Image], int]`
  - `tile(prompt: str, n: int, seed: float | int | None) -> tuple[list[Image.Image], int]`

- [ ] **Step 1: Viết test fail trong `tests/test_generate.py`**

```python
import pytest, torch
from pixelart.generate import resolve_seed, seamless

def test_resolve_seed_int():       assert resolve_seed(42.0) == 42 and isinstance(resolve_seed(42.0), int)
def test_resolve_seed_random():
    for s in (None, -1):
        assert 0 <= resolve_seed(s) < 2**31

def test_seamless_sets_and_restores():
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, padding=1), torch.nn.Linear(2, 2))
    with seamless(m):
        assert m[0].padding_mode == "circular"
    assert m[0].padding_mode == "zeros"

def test_seamless_restores_on_error():
    m = torch.nn.Conv2d(3, 3, 3, padding=1)
    with pytest.raises(RuntimeError):
        with seamless(m):
            raise RuntimeError
    assert m.padding_mode == "zeros"
```

- [ ] **Step 2: Chạy test để thấy fail**

Run: `pytest tests/test_generate.py -v`
Expected: FAIL với lỗi `ModuleNotFoundError: pixelart.generate`.

- [ ] **Step 3: Viết `pixelart/generate.py`**

- Chỉ import `torch` ở đầu file. `diffusers` import bên trong `load()`, để test và `app.py` import module được mà không phải tải model.
- `resolve_seed`: `None` hoặc số âm thì trả `random.randint(0, 2**31 - 1)`, còn lại trả `int(seed)`.
- `seamless`: duyệt `m.modules()` của mọi module truyền vào, lưu `padding_mode` cũ của từng `nn.Conv2d`, đặt `"circular"`, rồi khôi phục trong `finally`.
- `load()`:
  - `vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=torch.float16)`
  - `StableDiffusionXLPipeline.from_pretrained(BASE, vae=vae, torch_dtype=torch.float16, variant="fp16").to("cuda")`
  - `pipe.load_lora_weights(LORA)`
  - `AutoPipelineForImage2Image.from_pipe(pipe)`
  - lưu cả hai pipeline vào biến module
- Hàm sinh ảnh dùng chung:
  - `seed = resolve_seed(seed)`
  - `generator=[torch.Generator("cuda").manual_seed(seed + i) for i in range(n)]`
  - truyền thêm `num_images_per_prompt=n`, `prompt=PREFIX + prompt`, `negative_prompt=NEGATIVE`, `num_inference_steps=STEPS`, `guidance_scale=GUIDANCE`
  - txt2img và tile dùng `width=height=1024`
  - trả `(result.images, seed)`
- `img2img`: đưa `image=prepare_input(image)` và `strength=strength` vào pipeline img2img.
- `tile`: gọi txt2img trong `with seamless(pipe.unet, pipe.vae):`.
- Phần GPU không có unit test, sẽ kiểm chứng thủ công ở Task 4.

- [ ] **Step 4: Chạy test để thấy pass**

Run: `pytest tests/test_generate.py -v`
Expected: tất cả PASS.

- [ ] **Step 5: Commit**

```bash
git add pixelart/generate.py tests/test_generate.py
git commit -m "feat: add SDXL pixel art generation"
```

---

### Task 3: App Gradio (`app.py`)

**Files:**
- Create: `app.py`, `tests/test_app.py`

**Interfaces:**
- Consumes: mọi hàm của Task 1. `generate.load/txt2img/img2img/tile`, được gọi qua thuộc tính module (`generate.txt2img(...)`) để test monkeypatch được.
- Produces (trong `app.py`):
  - `sprite(prompt, size, palette_mode, colors, remove_bg, n, seed) -> tuple[list[Image.Image], list[str], int]`
  - `convert(image, prompt, strength, size, palette_mode, colors, remove_bg, n, seed) -> tuple[list[Image.Image], list[str], int]`
  - `make_tile(prompt, size, palette_mode, colors, n, seed) -> tuple[list[Image.Image], list[str], int]`
  - `get_auth() -> tuple[str, str]`
  - `build_ui() -> gr.Blocks`
  - `main() -> None`
- `palette_mode` có hai giá trị `"Tự động"` hoặc `"PICO-8"`. `"PICO-8"` ánh xạ sang `palette="pico8"`.
- Mỗi handler trả `(previews, paths, seed)`:
  - `paths`: file PNG kích thước thật, lưu trong `tempfile.mkdtemp()`
  - `previews`: ảnh qua `upscale_preview`; riêng `make_tile` là `upscale_preview(tile_preview(t))`

- [ ] **Step 1: Viết test fail trong `tests/test_app.py`**

```python
import gradio as gr, pytest, torch
from PIL import Image
import app

FAKE = lambda *a, **k: ([Image.new("RGB", (1024, 1024), "green")] * a[-2], 7)   # đối số áp chót là n

@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    for name in ("txt2img", "img2img", "tile"):
        monkeypatch.setattr(app.generate, name, FAKE)
    monkeypatch.setattr(app.postprocess, "remove_background", lambda im: im.convert("RGBA"))

def sizes(paths): return [Image.open(p).size for p in paths]

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
    def boom(*a, **k): raise torch.cuda.OutOfMemoryError()
    monkeypatch.setattr(app.generate, "txt2img", boom)
    with pytest.raises(gr.Error):
        app.sprite("knight", 32, "Tự động", 16, False, 4, None)

def test_auth_missing(monkeypatch):
    monkeypatch.delenv("PIXELART_USER", raising=False); monkeypatch.setenv("PIXELART_PASS", "x")
    with pytest.raises(SystemExit):
        app.get_auth()

def test_auth_ok(monkeypatch):
    monkeypatch.setenv("PIXELART_USER", "u"); monkeypatch.setenv("PIXELART_PASS", "p")
    assert app.get_auth() == ("u", "p")

def test_build_ui():                assert isinstance(app.build_ui(), gr.Blocks)
```

- [ ] **Step 2: Chạy test để thấy fail**

Run: `pytest tests/test_app.py -v`
Expected: FAIL với lỗi `ModuleNotFoundError: app`.

- [ ] **Step 3: Viết `app.py`**

- Import: `from pixelart import generate, postprocess`. `app.py` phải gọi `postprocess.remove_background` qua module để monkeypatch được.
- Gọi `generate.*` bằng **đối số theo vị trí**, đúng thứ tự trong chữ ký của Task 2, vì model giả trong test đọc `n` ở vị trí áp chót.
- Kiểm tra đầu vào:
  - mô tả trống (sau `strip()`) ở tab A/C thì `raise gr.Error("Hãy nhập mô tả")`
  - tab B `image is None` thì `raise gr.Error("Hãy tải ảnh lên")`
- Bọc mọi lời gọi `generate.*` bằng `try/except torch.cuda.OutOfMemoryError`: gọi `torch.cuda.empty_cache()`, rồi `raise gr.Error("Hết bộ nhớ GPU — giảm số biến thể rồi thử lại")`.
- Tab B, `strength == 0`: `images = [postprocess.prepare_input(image)]`, seed trả `0`, không gọi model.
- Helper `_finish(images, size, palette_mode, colors, remove_bg)`: với mỗi ảnh, `remove_background` (nếu bật) → `to_grid(img, int(size))` → `quantize(img, int(colors), palette)` → lưu PNG. Ép `size`, `colors`, `n` về `int` vì Gradio có thể trả float.
- `get_auth`: thiếu biến nào thì `sys.exit("Thiếu PIXELART_USER/PIXELART_PASS trong Kaggle Secrets")`.
- `build_ui`: dùng `gr.Blocks(title="Pixel Art")`, mỗi handler một `gr.Tab`. Control theo bảng 4.3 trong spec:
  - `gr.Radio` cho kích thước và bảng màu
  - `gr.Slider` cho số màu, số biến thể, độ biến đổi (`step=0.05`)
  - `gr.Checkbox` cho xóa nền (tab A mặc định bật, tab B mặc định tắt)
  - `gr.Number(value=None, precision=0)` cho seed
  - `gr.Image(type="pil")` cho ảnh tải lên
  - đầu ra: `gr.Gallery`, `gr.File(file_count="multiple")`, `gr.Number` cho seed đã dùng
- `main`:
  1. `auth = get_auth()`
  2. nếu `not torch.cuda.is_available()` thì `sys.exit("Hãy bật GPU T4 trong Settings của notebook")`
  3. `generate.load()`
  4. `build_ui().queue(default_concurrency_limit=1).launch(share=True, auth=auth)`
- Cuối file: `if __name__ == "__main__": main()`.

- [ ] **Step 4: Chạy toàn bộ test để thấy pass**

Run: `pytest -v`
Expected: tất cả test của Task 1–3 PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: add Gradio app with sprite, convert and tile tabs"
```

---

### Task 4: Notebook Kaggle + README

**Files:**
- Create: `kaggle_notebook.ipynb`, `README.md`

**Interfaces:**
- Consumes: `app.py` (`python app.py`), `requirements.txt`

- [ ] **Step 1: Viết `kaggle_notebook.ipynb`** (nbformat 4, 2 code cell, kèm một markdown cell hướng dẫn ở đầu)

Cell 1:
```python
BRANCH = "main"
!git clone --depth 1 -b {BRANCH} https://github.com/qu4nc0d3r/create-pixelart
!pip install -q -r create-pixelart/requirements.txt
```
Cell 2:
```python
import os
from kaggle_secrets import UserSecretsClient
s = UserSecretsClient()
os.environ["PIXELART_USER"] = s.get_secret("PIXELART_USER")
os.environ["PIXELART_PASS"] = s.get_secret("PIXELART_PASS")
%cd create-pixelart
!python app.py
```

- [ ] **Step 2: Kiểm tra notebook là JSON hợp lệ**

Run: `python -c "import json; nb=json.load(open('kaggle_notebook.ipynb')); print(nb['nbformat'], len(nb['cells']))"`
Expected: `4 3`

- [ ] **Step 3: Viết `README.md` bằng tiếng Việt**

Nội dung gồm:
- Mục 3 của spec: các bước chạy trên Kaggle, cách tạo Secrets, giới hạn của Kaggle
- Mô tả 3 tab
- Cách chạy test ở môi trường dev (lệnh trong Task 1, Step 1, rồi `pytest`)
- Ghi chú: trước khi merge vào `main`, đổi `BRANCH` trong cell 1 sang tên branch đang phát triển
- Cảnh báo cần kiểm tra license của `nerijs/pixel-art-xl` trước khi dùng thương mại

- [ ] **Step 4: Commit và push**

```bash
git add kaggle_notebook.ipynb README.md
git commit -m "docs: add Kaggle notebook and README"
git push -u origin claude/tender-ramanujan-juhvsg
```

- [ ] **Step 5: Kiểm chứng thủ công trên Kaggle** (người dùng làm, vì môi trường dev không có GPU)
1. Import notebook, bật GPU T4 + Internet, tạo 2 Secrets, đặt `BRANCH = "claude/tender-ramanujan-juhvsg"`, Run All.
2. Mở link `gradio.live` và đăng nhập.
3. Tab A: "knight with red cape", 32 px → tải PNG về, kiểm tra 32×32 và nền trong suốt.
4. Tab B: tải một ảnh lên, độ biến đổi 0.5, sau đó 0 → có kết quả.
5. Tab C: "grass with small flowers" → bản ghép 3×3 không thấy đường nối.
6. Ghi lại thời gian khởi động và thời gian sinh mỗi ảnh. Nếu lỗi thì gửi log.
