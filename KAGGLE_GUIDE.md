# Chạy Lab Day 10 từ đầu trên Kaggle

1. Tạo Kaggle Notebook và import `lab_day10.ipynb` trong repo này.
2. Trong Settings, bật Internet và chọn GPU T4 ×2 nếu có. Notebook chỉ dùng GPU 0.
3. Add Input dataset `lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes`.
4. Chỉnh **cell cấu hình đầu tiên** rồi Run All. Không cần thêm cell thủ công cho E2/E3.

## Chọn cách chạy

```python
RUN_MODE = 'full'
RUNS = ('E0', 'E1', 'E2', 'E3', 'E4')
```

Đây là mặc định: E0 zero-shot, E1–E3 train 300 bước, E4 so sánh 10% và 100% dữ liệu.
Mỗi setup có cell thực thi riêng và bắt đầu từ model gốc. Batch mặc định là 1,
tích lũy 8 microbatch để dùng effective batch 8 trên một GPU.

Để kiểm tra môi trường trước khi chạy dài:

```python
RUN_MODE = 'smoke'
RUNS = ('E0', 'E1', 'E2', 'E3')
```

Smoke train 2 bước/setup và suy luận 3 ảnh val, lưu trong thư mục smoke riêng,
không tạo evidence test. Sau khi smoke chạy được, chuyển lại full và Run All.

Để chỉ chạy E3:

```python
RUN_MODE = 'full'
RUNS = ('E3',)
```

Model tải lại từ cache trong cùng phiên. Run full đã hoàn tất sẽ dùng lại checkpoint;
predictions chỉ được dùng lại nếu khớp run, split, danh sách ảnh và max_new_tokens.
Run dở dang sẽ train lại từ 0 khi `RESTART_INCOMPLETE=True`; chưa hỗ trợ resume
optimizer/scaler/RNG từ checkpoint giữa chừng.

## Dữ liệu và Secrets

Mặc định notebook tìm thư mục có cả `images/` và `labels-YOLO/` bên dưới `/kaggle/input`.
Nếu có nhiều dataset phù hợp, đặt `DATASET_SOURCE` trong cell cấu hình.
Cũng có thể đặt `DATASET_ZIP` để dùng file ZIP.

Add Input không cần Kaggle API key. Nếu chọn `USE_KAGGLE_API=True`, thêm Secrets
`KAGGLE_USERNAME`, `KAGGLE_KEY` và cấp quyền truy cập cho notebook.
`USE_HF_TOKEN=True` dùng Secret `HF_TOKEN`; model công khai không bắt buộc token này.
Notebook đọc Secrets bằng `UserSecretsClient` và không in giá trị khóa.

## Lưu và dùng lại Output

Kết quả được ghi trực tiếp vào:

```text
/kaggle/working/lab_day10_outputs/
  work/full/e1/final/         # adapter, projector nếu có, processor, run.json
  work/full/e2/final/
  work/full/e3/final/
  work/smoke/                # smoke nằm riêng
  predictions/full/         # raw output + manifest
  report/evidence/          # JSON do scripts/score.py tạo
  report/figures/
  report/results_table.md
  REPORT.md                 # mẫu cần tự điền
  dataset_meta.json
  report_bundle.zip         # report/evidence, ảnh, mẫu báo cáo; không chứa trọng số
```

Dataset và HF cache nằm trong `/tmp/lab_day10`, ngoài thư mục Output để giảm dung lượng.

Sau khi chạy, **Save Version với Output được lưu** hoặc tải Output về trước khi kết thúc phiên.
Không coi dữ liệu working chưa lưu là backup bền vững.
Nếu dùng Save & Run All, notebook sẽ thực thi lại từ đầu trong phiên mới theo cấu hình đang chọn.

Ở phiên sau, gắn Output cũ bằng Add Input và đặt:

```python
PREVIOUS_OUTPUT_SOURCE = '/kaggle/input/<nguon-output-cu>/lab_day10_outputs'
```

Chọn đường dẫn thực tế từ panel Input. Notebook khôi phục những file chưa có,
kiểm tra config/model revision/dataset_version trước khi dùng lại.
Nếu muốn chạy một cấu hình khác, dùng `OUTPUT_ROOT` mới để giữ kết quả cũ tách biệt.
Output Colab cũ không có manifest của bản Kaggle này; cần chạy lại để tạo kết quả tương thích.

## E4 và bài nộp

`E4_BASE_SETUP='auto'` chọn F1 cao nhất trên cùng val preview, mặc định 40 ảnh.
Đây là lựa chọn phục vụ thí nghiệm, không chứng minh setup đó tốt nhất trên toàn bộ phân bố.
Có thể chọn cụ thể `'E1'`, `'E2'` hoặc `'E3'` sau khi đọc val và tài nguyên.
Run 100% dùng lại predictions của baseline; run 10% chỉ đổi train_fraction.

Tải `report_bundle.zip`, giải nén vào repo cá nhân, tự điền `REPORT.md` và đưa notebook
đã export từ Kaggle vào repo. Không commit dữ liệu, khóa hay trọng số.
Kết quả kiểm tra cuối notebook có thể FAIL vì REPORT.md còn mẫu hoặc chưa đủ commit;
hoàn thiện các mục đó và chạy lại `python scripts/check_submission.py` trước khi nộp.
E5 hiện có reward/test CPU; chưa triển khai GRPO.

## Hai lỗi đã xử lý

- Skip quantization dùng `model.visual` để giữ ViT/projector ngoài 4-bit.
- Cell setup gỡ torchao trước khi import PEFT, tránh backend cũ gây ImportError ở E3.

Kiểm tra CPU trong repo: `python3 notebook_checks.py` và `python3 -m pytest -q`.
Các kiểm tra này chưa thay thế smoke test trên GPU Kaggle.

Tham khảo: [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks),
[Kaggle CLI và Output](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md),
[PEFT kiểm tra torchao](https://github.com/huggingface/peft/blob/v0.21.2/src/peft/import_utils.py),
[Transformers module matching](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/quantizers/quantizers_utils.py).
