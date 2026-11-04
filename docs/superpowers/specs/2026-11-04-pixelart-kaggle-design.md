# Thiết kế: Công cụ tạo pixel art nội bộ chạy trên Kaggle

Ngày: 2026-11-04 · Trạng thái: chờ duyệt

## 1. Mục tiêu

Công cụ nội bộ cho **team game** tạo nhanh asset pixel art (nhân vật, vật phẩm, icon, quái, tile) để làm prototype và asset game, lấy cảm hứng từ PixelLab.

- **Miễn phí hoàn toàn:** chỉ dùng model mã nguồn mở, không gọi API trả phí.
- **Chỉ chạy trên Kaggle** (GPU T4 miễn phí). Không chạy CPU, không cài gì lên máy người dùng.
- **Thành công khi:** một người bật notebook, chia link, cả team đăng nhập và tải được PNG pixel art đúng kích thước, đúng bảng màu, dùng ngay được trong game.

## 2. Phạm vi

**v1 có:**
- A. Chữ → Sprite đơn lẻ
- B. Ảnh → Pixel art
- C. Tile lặp liền mạch

**Không có trong v1:** nhân vật 4/8 hướng, animation, lịch sử/lưu trữ ảnh, phân quyền nhiều người dùng, chọn nhiều LoRA, cache model bằng Kaggle Dataset, chế độ CPU.

## 3. Cách sử dụng

1. Mở `kaggle_notebook.ipynb` trên Kaggle. Trong Settings bật **Accelerator: GPU T4** và **Internet: On** (cần xác minh số điện thoại).
2. Trong Add-ons → Secrets, tạo `PIXELART_USER` và `PIXELART_PASS`.
3. Bấm Run All. Khoảng 3–5 phút sau, notebook in link `https://xxxx.gradio.live`.
4. Gửi link cho team. Mỗi người đăng nhập bằng user/mật khẩu ở bước 2.

Giới hạn của Kaggle: khoảng 30 giờ GPU/tuần/tài khoản, phiên tối đa 12 giờ, link đổi mỗi phiên.

## 4. Kiến trúc

```
app.py                    # Gradio UI: 3 tab, đăng nhập, hàng chờ
pixelart/__init__.py
pixelart/generate.py      # Load SDXL + LoRA; txt2img, img2img, tile
pixelart/postprocess.py   # Xóa nền, thu nhỏ về lưới, giảm màu, ghép 3×3 (chỉ CPU)
kaggle_notebook.ipynb     # Cell 1: clone + pip install; Cell 2: chạy app
requirements.txt
tests/test_postprocess.py
tests/test_app.py         # Nối dây UI với model giả
README.md                 # Hướng dẫn tiếng Việt
```

Repo là **public**, nên notebook clone thẳng `https://github.com/qu4nc0d3r/create-pixelart`.

### 4.1 Model (`pixelart/generate.py`)

| Thành phần | Nguồn |
|---|---|
| Model nền | `stabilityai/stable-diffusion-xl-base-1.0`, fp16 |
| LoRA phong cách | `nerijs/pixel-art-xl` |
| VAE | `madebyollin/sdxl-vae-fp16-fix` (tránh ảnh đen khi chạy fp16) |
| Xóa nền | `rembg` (model u2net) |

- Load một lần khi app khởi động, lên `cuda`.
- Pipeline img2img tạo từ pipeline txt2img bằng `AutoPipelineForImage2Image.from_pipe`, dùng chung trọng số nên không tốn thêm VRAM.
- Mọi ảnh sinh ra có kích thước 1024 (cạnh dài với tab B, vuông 1024×1024 với tab A và C).
- Prompt luôn được thêm tiền tố `pixel art, `. Negative prompt mặc định gồm `blurry, 3d render, photo, realistic, gradient, noise`.
- Hằng số mặc định (số bước, guidance) để ở đầu file, chỉnh được.
- **Tile liền mạch:** một context manager đặt `padding_mode="circular"` cho mọi `nn.Conv2d` trong UNet và VAE khi tạo tile, rồi khôi phục giá trị cũ khi xong (kể cả khi lỗi).
- Seed: để trống thì ngẫu nhiên. Biến thể thứ i dùng `seed + i`. Seed thực tế được hiển thị ra cho người dùng.

Giao diện module:
```python
txt2img(prompt: str, n: int, seed: int | None) -> tuple[list[Image], int]
img2img(image: Image, prompt: str, strength: float, n: int, seed: int | None) -> tuple[list[Image], int]
tile(prompt: str, n: int, seed: int | None) -> tuple[list[Image], int]
```

### 4.2 Hậu xử lý (`pixelart/postprocess.py`)

Các hàm thuần trên ảnh PIL, không phụ thuộc torch:

```python
remove_background(img) -> Image                 # RGBA, qua rembg
to_grid(img, long_side: int) -> Image           # thu nhỏ bằng Image.BOX, giữ tỉ lệ
quantize(img, colors: int | None, palette: str | None) -> Image
                                                 # alpha ép về 0/255 (ngưỡng 128);
                                                 # màu: tự động N màu hoặc PICO-8 (16 màu cố định)
upscale_preview(img, size: int = 512) -> Image  # phóng to bằng nearest-neighbor để xem
tile_preview(img) -> Image                      # ghép 3×3
```

Thứ tự trong tab A: `remove_background` (nếu bật) → `to_grid` → `quantize`. Xóa nền làm trên ảnh lớn để viền cắt chuẩn hơn.

### 4.3 Giao diện (`app.py`)

| Tab | Đầu vào | Đầu ra |
|---|---|---|
| **Chữ → Sprite** | mô tả; kích thước 16/32/64/128 (mặc định 32); bảng màu: tự động N (4–64, mặc định 16) hoặc PICO-8; xóa nền (mặc định bật); số biến thể 1–4 (mặc định 4); seed | gallery ảnh phóng to; file PNG kích thước thật; seed đã dùng |
| **Ảnh → Pixel art** | ảnh tải lên; mô tả (tùy chọn); độ biến đổi 0–0.8 (mặc định 0.5); kích thước, màu, xóa nền như tab A (xóa nền mặc định tắt); số biến thể; seed | như tab A |
| **Tile liền mạch** | mô tả; kích thước 16/32/64 (mặc định 32); bảng màu như tab A; số biến thể; seed | gallery bản ghép 3×3; file PNG tile kích thước thật; seed |

- Tab B: ảnh vào được co về cạnh dài 1024, cạnh còn lại làm tròn về bội của 8. Nếu **độ biến đổi = 0**, app bỏ qua model, chỉ chạy hậu xử lý trên ảnh gốc (lúc đó số biến thể = 1).
- File PNG kích thước thật được lưu vào thư mục tạm của phiên và trả về qua `gr.File` để tải.
- Hàng chờ: `demo.queue(default_concurrency_limit=1)`, các yêu cầu chạy lần lượt trên 1 GPU.
- Khởi chạy: `demo.launch(share=True, auth=(user, pass))`.

## 5. Xử lý lỗi

| Tình huống | Hành vi |
|---|---|
| Không có GPU (`torch.cuda.is_available()` là False) | Thoát ngay với thông báo "Hãy bật GPU T4 trong Settings của notebook". |
| Thiếu `PIXELART_USER`/`PIXELART_PASS` (đọc từ biến môi trường, notebook nạp từ Kaggle Secrets) | **Không khởi chạy**. Không bao giờ mở link public khi chưa có mật khẩu. |
| Hết VRAM (`torch.cuda.OutOfMemoryError`) | Giải phóng bộ nhớ (`torch.cuda.empty_cache()`), báo `gr.Error("Hết bộ nhớ GPU — giảm số biến thể rồi thử lại")`. App vẫn chạy tiếp. |
| Mô tả trống ở tab A/C | `gr.Error("Hãy nhập mô tả")`, không gọi model. |
| Không tải ảnh ở tab B | `gr.Error("Hãy tải ảnh lên")`. |

## 6. Kiểm thử

**Tự động (pytest, không cần GPU, không cần mạng):**
- `to_grid`: ảnh ra đúng kích thước, cả ảnh vuông lẫn không vuông.
- `quantize`:
  - số màu duy nhất không vượt quá N
  - kênh alpha chỉ có giá trị 0 và 255
  - với PICO-8, mọi màu đều nằm trong bảng PICO-8
- Liền mạch: ảnh tuần hoàn 1024×1024 sau `to_grid(…, 32)` vẫn tuần hoàn, kiểm tra cạnh trái nối cạnh phải và cạnh trên nối cạnh dưới qua `tile_preview`.
- `tile_preview`: kích thước gấp 3 lần mỗi chiều.
- `tests/test_app.py`: thay `pixelart.generate` bằng hàm trả ảnh giả, gọi các handler của 3 tab, kiểm tra kết quả đúng kích thước, đúng số file. Kiểm tra thêm: mô tả trống thì báo lỗi; thiếu secrets thì app không khởi chạy. Test không gọi `remove_background` thật vì rembg cần tải model.

**Thủ công trên Kaggle (lần đầu):** chạy notebook, tạo một sprite, một ảnh từ ảnh tải lên, một tile, rồi tải PNG về kiểm tra. Môi trường phát triển không có GPU nên phần `generate.py` chỉ được kiểm chứng ở bước này.

## 7. Rủi ro

- **License:** SDXL base dùng CreativeML Open RAIL++-M, cho phép dùng thương mại kèm điều khoản sử dụng. Cần kiểm tra license của `nerijs/pixel-art-xl` trước khi dùng asset trong sản phẩm thương mại.
- **Chính sách Kaggle** có thể thay đổi (giới hạn giờ GPU, quy định chạy UI qua link share).
- **Tốc độ** (khoảng 10–20 giây/ảnh trên T4) và thời gian khởi động (3–5 phút, tải khoảng 7 GB) mới là ước lượng, cần đo thực tế.
- **Link Gradio share** hết hạn sau khoảng 72 giờ hoặc khi phiên kết thúc. Với phiên tối đa 12 giờ của Kaggle thì không phải vấn đề.
