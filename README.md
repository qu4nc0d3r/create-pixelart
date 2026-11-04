# create-pixelart

Công cụ nội bộ cho team game tạo pixel art bằng AI. Chạy miễn phí trên GPU của Kaggle với model mở **SDXL** + LoRA [`nerijs/pixel-art-xl`](https://huggingface.co/nerijs/pixel-art-xl).

## 3 tính năng

| Tab | Làm gì |
|---|---|
| **Chữ → Sprite** | Nhập mô tả, nhận sprite 16/32/64/128 px, nền trong suốt, giảm màu (tự động hoặc PICO-8) |
| **Ảnh → Pixel art** | Tải ảnh lên. Độ biến đổi > 0 thì AI vẽ lại theo phong cách pixel; bằng 0 thì chỉ pixel hóa ảnh gốc, rất nhanh |
| **Tile liền mạch** | Nhập mô tả, nhận tile lặp không lộ đường nối, kèm bản ghép 3×3 để kiểm tra |

Mỗi lần tạo, bạn tải về **file PNG đúng kích thước thật** (vd. 32×32) để dùng ngay trong game. App cũng hiển thị seed đã dùng: nhập lại seed đó để tạo ra kết quả giống hệt.

## Chạy trên Kaggle

1. Tạo notebook mới trên [kaggle.com](https://www.kaggle.com), chọn **File → Import Notebook** rồi chọn `kaggle_notebook.ipynb` trong repo này.
2. Trong **Settings**, bật **Accelerator: GPU T4** và **Internet: On**. Kaggle yêu cầu xác minh số điện thoại trước khi cho bật hai mục này.
3. Trong **Add-ons → Secrets**, tạo hai secret `PIXELART_USER` và `PIXELART_PASS`, rồi bật cả hai cho notebook. Đây là tài khoản đăng nhập của cả team. Nếu thiếu secret, app sẽ không chạy.
4. Bấm **Run All**. Khoảng 3–5 phút sau, notebook in ra link `https://xxxx.gradio.live`. Gửi link cùng user/mật khẩu cho team.

**Giới hạn của Kaggle:**
- Khoảng 30 giờ GPU mỗi tuần cho mỗi tài khoản.
- Mỗi phiên chạy tối đa 12 giờ.
- Mỗi lần chạy lại thì link sẽ đổi.

Nhiều người tạo ảnh cùng lúc thì yêu cầu sẽ xếp hàng lần lượt, vì chỉ có 1 GPU.

> Notebook mặc định clone branch `main`. Nếu muốn chạy thử một branch chưa merge, sửa `BRANCH` ở cell đầu tiên.

## Phát triển

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pytest
```

Test không cần GPU và không cần tải model. Phần sinh ảnh (`pixelart/generate.py`) chỉ kiểm chứng được khi chạy trên Kaggle.

| File | Vai trò |
|---|---|
| `app.py` | Giao diện Gradio, đăng nhập, xử lý lỗi |
| `pixelart/generate.py` | Load SDXL + LoRA, tạo ảnh từ chữ, từ ảnh, tile liền mạch |
| `pixelart/postprocess.py` | Xóa nền, thu nhỏ về lưới pixel, giảm màu (chỉ dùng CPU) |
| `kaggle_notebook.ipynb` | Notebook để chạy trên Kaggle |

## Lưu ý license

SDXL base dùng license CreativeML Open RAIL++-M, cho phép dùng thương mại nhưng phải tuân theo các điều khoản sử dụng. **Cần kiểm tra license của LoRA `nerijs/pixel-art-xl`** trước khi đưa asset vào sản phẩm thương mại.
