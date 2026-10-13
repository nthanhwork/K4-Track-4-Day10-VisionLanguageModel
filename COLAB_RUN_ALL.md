# Một lần Run all trên Colab

1. Upload `lab_day10.ipynb` trong thư mục hiện tại vào Colab, dùng phiên mới với T4 GPU.
2. Trước khi chạy, cung cấp **một trong hai** cách xác thực Kaggle:
   - Secret `KAGGLE_API_TOKEN` mới.
   - Hai Secrets legacy `KAGGLE_USERNAME` và `KAGGLE_KEY`.
   Bật **Notebook access**. `HF_TOKEN` là tùy chọn.
3. Nếu muốn điền trực tiếp, điền các trường rỗng trong `CREDENTIALS` ở cell đầu.
   Biến môi trường có sẵn cũng được đọc. Không in hoặc commit credential.
4. Chọn **Runtime → Run all**. Xác nhận kết nối Drive khi được yêu cầu lần đầu.

Mặc định chạy đủ E0, E1, E2, E3, E4. Không cần đổi cờ giữa các phần.

```python
RUN_MODE = 'full'
RUNS = ('E0', 'E1', 'E2', 'E3', 'E4')
```

Notebook clone công cụ của đề, cài đúng phiên bản thư viện, gỡ torchao tùy chọn,
tải dataset với credential đã cung cấp, chia split, kiểm tra box/mask loss,
train và chấm điểm. E1/E2/E3 bắt đầu độc lập từ model gốc, 300 bước optimizer/setup.
Batch 1, tích lũy 8 microbatch; inference batch 4, có thể đổi thành 1 nếu thiếu VRAM.
E4 chỉ đổi data_size: 10% so với 100% của setup được chọn dựa trên val preview.

## Kết quả và chạy lại

Mặc định `RUN_LABEL='run_all_v1'`, lưu ở:

```text
/content/lab_day10_outputs/run_all_v1/
/content/drive/MyDrive/lab_day10_colab_runs/run_all_v1/
```

Trong đó có `work/full/e1/final`, `work/full/e2/final`, `work/full/e3/final`,
predictions/manifest, `report/evidence/*.json`, bảng/ảnh, mẫu `REPORT.md` và `report_bundle.zip`.
LoRA và projector E2/E3 được lưu riêng, rồi nạp lại khi suy luận.
Checkpoint, predictions và evidence được sao lưu lên Drive ngay sau mỗi phần.

Nếu Run all lần nữa với cùng cấu hình, notebook dùng lại các run hoàn tất và predictions
đúng checkpoint. Nếu session ngắt, mở phiên mới và Run all với cùng RUN_LABEL;
file đã lưu trên Drive sẽ được khôi phục. Run dở dang chưa lưu optimizer/RNG nên
train lại từ 0 khi `RESTART_INCOMPLETE=True`; không giả vờ resume từ bước đã ghi.
Nếu thay đổi cấu hình/model/dataset, dùng RUN_LABEL mới để giữ kết quả riêng.
Thời lượng phiên GPU do Colab quyết định, nên không bảo đảm phiên miễn phí đủ để chạy mọi phần.

Chọn `RUN_MODE='smoke'` để thử 2 bước và 3 ảnh val/setup; smoke nằm riêng và không tạo evidence test.
Phần bắt buộc E0–E4 có mã đầy đủ; E5 hiện có reward/test CPU, chưa chạy GRPO.

## Nộp bài

Lấy `report_bundle.zip` trên Drive, giải nén vào repo, tự điền `REPORT.md` bằng số liệu thật.
Download notebook đã chạy từ Colab và đưa vào repo sau khi xóa credential trực tiếp, nếu có.
Không commit data, khóa hoặc trọng số. Không sửa scripts/tests của đề.
Kiểm tra cuối notebook có thể FAIL vì báo cáo còn mẫu hoặc thiếu commit; hoàn thiện các mục đó
rồi chạy `python scripts/check_submission.py` trước khi nộp.

Tham khảo: [Colab FAQ](https://research.google.com/colaboratory/faq.html),
[Kaggle authentication](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication),
[PEFT torchao check](https://github.com/huggingface/peft/blob/v0.21.2/src/peft/import_utils.py),
[Transformers module matching](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/quantizers/quantizers_utils.py).
