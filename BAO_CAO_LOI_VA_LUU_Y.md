# Báo cáo lỗi và lưu ý kỹ thuật — Lab Day 10

Họ tên: **Nguyễn Thái Anh**  
Dataset: **`d10-21e4070301`**

Các lỗi dưới đây được ghi nhận từ traceback đã gặp, notebook Colab có output,
checkpoint/predictions tải về và evidence trong repo. Lỗi quan sát được được phân
biệt với nguy cơ mà code chủ động kiểm tra. Không sửa công cụ chấm/split của đề.

## 1. AssertionError vì visual bị lượng tử hóa ngoài ý muốn

**Hiện tượng:** E0 tải xong model rồi dừng ở assertion phát hiện `Linear4bit` trong
`model.model.visual`. Cấu hình ban đầu dùng
`llm_int8_skip_modules=['visual','lm_head']`.

**Nguyên nhân:** tên module thực bắt đầu bằng `model.visual`; skip ngắn không khớp
như dự kiến ở phiên Transformers đang dùng. Projector thuộc nhánh visual nên cũng
có nguy cơ bị chuyển thành Linear4bit, làm sai thiết kế full-projector training.

**Xử lý:** dùng `['model.visual','lm_head']`, sau load kiểm tra visual và lm_head
không có Linear4bit, trong khi LLM vẫn có Linear4bit. Đây là sửa cấu hình notebook,
không sửa `GUIDE.md` hay mã của đề.

**Bằng chứng sau xử lý:** E0/E1/E2/E3 chạy ra evidence; E2 ghi 142.135.296 tham số
trainable, trong đó 109.105.152 là projector full trainable. Xem cơ chế lọc module
ở [mã Transformers 5.19](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/quantizers/quantizers_utils.py).

## 2. ImportError torchao khi thêm LoRA ViT ở E3

**Hiện tượng:** `train_experiment(e3_config)` dừng trong PEFT import_utils với thông
báo torchao 0.10.0 không tương thích, yêu cầu phiên bản tối thiểu 0.16.0.

**Nguyên nhân:** E3 thêm adapter vào Linear thường của ViT; PEFT kiểm tra backend
torchao tùy chọn đã có trong môi trường. Backend này không cần cho bài dùng
bitsandbytes NF4, nhưng bản cài sẵn quá cũ vẫn gây ImportError khi được kiểm tra.

**Xử lý:** gỡ torchao trước khi import PEFT:

```python
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=True)
```

Nếu torchao đã được import, dùng phiên/kernel mới; gỡ package trên đĩa không xóa
module đã nằm trong bộ nhớ. Giữ torch/torchvision của Colab và các dependency
pinned của notebook; không nâng toàn bộ stack chỉ để dùng backend không cần thiết.

**Bằng chứng sau xử lý:** E3 hoàn tất, F1=0,453039, parse_valid_rate=1.0. Tham khảo
[PEFT 0.21.2 import_utils](https://github.com/huggingface/peft/blob/v0.21.2/src/peft/import_utils.py).

## 3. AssertionError do nhãn crack có chiều cao bằng 0

**Hiện tượng:** cell kiểm tra toàn bộ đáp án train dừng với nhãn:

```text
('crack', [0.639062, 0.786111, 0.003125, 0.0])
```

Chiều cao 0 dẫn đến `y2 == y1`; assertion box có diện tích dương không thể qua.
Parser nhãn của đề kiểm tra giá trị thuộc [0,1], nên giá trị 0 vẫn được chấp nhận.

**Xử lý:** bỏ box có w/h=0 khỏi JSON target, giữ các box hợp lệ và giữ ảnh. Với box
có diện tích dương nhưng quá nhỏ bị round thành một đường, chỉ làm tròn ra ngoài
ở trục bị collapse. Vẫn assert lớp đúng và số hữu hạn/trong miền; không tắt toàn
bộ validation hay tự tạo chiều cao cho nhãn vốn bằng 0.

**Lưu ý về chấm điểm:** không thay `objects` gốc, session split hoặc GT của scorer.
Nhãn GT diện tích 0 không đạt IoU≥0,5 và có thể trở thành FN không ghép được.
Vì vậy không được âm thầm lọc GT test để cải thiện F1. Báo cáo không ghi số nhãn
lỗi toàn dataset vì chưa có phép đếm đầy đủ trên nhãn gốc tại repo.

## 4. Kaggle không tìm thấy dataset khi chưa gắn Input

**Hiện tượng:** lần thử Kaggle báo tìm thấy 0 dataset hợp lệ; ảnh giao diện cho thấy
`No input attached`. Hàm discovery cần cặp thư mục `images/` và `labels-YOLO/`.

**Nguyên nhân:** GPU/runtime sẵn sàng không có nghĩa dataset đã được attach. Đây
là lỗi nguồn dữ liệu, không phải lỗi mô hình.

**Xử lý:** cung cấp nguồn dữ liệu rõ ràng, dùng Add Input phù hợp hoặc tải dataset
bằng API/ZIP. Bản Colab hiện dùng `prepare_data.py --kaggle` với credential được
cung cấp trước khi chạy. Không trộn `/kaggle/input` với đường dẫn `/content` của
Colab. Split cuối vẫn do công cụ của đề tạo.

## 5. Quota Colab ngắt E4 và giới hạn khôi phục checkpoint

**Hiện tượng:** sau E0–E3, Colab cảnh báo sắp hết quota trong khi E4 đang train.
Ảnh chụp lúc đầu cho thấy bước 50/300; bản notebook tải về sau đó có checkpoint
E4 cũ bước 100. Run dở dang đó không phải run tạo evidence E4 cuối.

**Sao lưu thực tế:** adapter/projector, processor, loss và run.json được lưu sau
bước 100/200 và final 300; predictions/manifest sau khi xong một split; evidence
sau khi chấm. File trong `/content` là tạm thời; file đã chép thành công lên Drive
được giữ độc lập với runtime. Autosave notebook không tự lưu trọng số trong GPU.

**Xử lý:** giữ checkpoint, predictions, manifest, val metrics và evidence E0–E3
trên Drive. Cell khôi phục E4 độc lập dựng lại môi trường, kiểm tra artifact trước
khi train; E0–E3 không train/infer lại. E4 được khởi tạo lại từ model gốc và train
300 bước trên subset 142 ảnh, rồi tạo evidence cuối với
`run_id=143a866fd33744c796a80fe80273414a`.

**Giới hạn:** checkpoint hiện không lưu optimizer, scheduler, GradScaler, RNG và
vị trí iterator. Không thể tiếp tục chính xác từ bước 100 chỉ bằng adapter. Run
hoàn tất E4 có thời gian 58,38 phút; không cộng thời gian cũ bị ngắt vào metadata
run mới. E4 100% dùng lại E3, không train thêm một replicate.

**Giữ log:** trong notebook cũ, chèn cell khôi phục sau E4 cũ và trước bảng kết quả;
chỉ chạy cell mới trở xuống trên một phiên GPU sử dụng được. Không chạy lại các
cell phía trên qua giao diện để tránh thay output/log cũ. Các file mã khôi phục:
`colab_resume_e4_cell.py`, `resume_e4_colab.ipynb`.

## 6. Warning processor_kwargs xuất hiện nhiều lần

**Hiện tượng:** Transformers in cảnh báo rằng kwargs truyền cho processor cần nằm
trong dictionary `processor_kwargs` thay vì `**kwargs`.

**Đánh giá:** đây là warning API, không phải traceback dừng train. Các run E1–E3
hoàn tất 300 bước, log có 0 bước gradient overflow bị bỏ qua; evidence vẫn đầy đủ.
Không thể dùng warning này để giải thích mọi sai số grounding.

**Lưu ý:** nếu cập nhật cách gọi processor, đọc API của đúng phiên bản pinned rồi
kiểm tra lại prefix prompt, 220 token ảnh, mask labels và collate trước khi train.
Không chỉ tắt mọi warning để che lỗi dữ liệu/shape; cũng không đổi dependency
giữa chừng rồi tái sử dụng artifact mà bỏ kiểm tra fingerprint.

## 7. E0 lặp JSON và grounding kém

**Quan sát:** 162/311 output hợp lệ, 149 output không parse được; trong output hợp
lệ có 59 chuỗi `[]`. E0 ghi TP=32, FP=244, FN=551 và crack F1=0. Output thật và số
đếm ở [e0_diagnostics.json](report/e0_diagnostics.json).

**Lưu ý:** cần kiểm tra output thô, parse validity và TP/FP/FN riêng. JSON đúng
cấu trúc không chứng minh box đúng; chuỗi `[]` chỉ là lỗi bỏ sót khi ảnh có GT.
Output lặp/cắt ở giới hạn 256 token có thể mất dấu đóng JSON. Trong run này chưa
có ablation tăng max_new_tokens, nên không quy mọi output lỗi về token budget.

SFT E1 nâng tỷ lệ hợp lệ lên 99,36% và tăng F1 thêm 0,207760 so với E0. Tuy vậy
SFT không loại bỏ mọi lỗi định vị: E3 vẫn có 257 FP và 337 FN.

## 8. E4 training loss thấp nhưng chất lượng tổng quát hóa giảm

**Quan sát:** E4 10% có loss bước 300 khoảng 0,0072 nhưng test F1=0,288991; E3
100% có loss cuối 0,401345 và test F1=0,453039. E4 10% có val F1=0,310502 so với
baseline E3 0,521277 trên cùng 40 ảnh preview.

**Diễn giải:** lặp lại tập 142 ảnh với cùng 300 bước tạo nguy cơ học thuộc; kết quả
phù hợp với overfitting. Training loss thấp không thay thế validation/test.
FP/ảnh tăng từ 0,826367 lên 1,723473 khi dùng 10% dữ liệu. Đây là nhận định theo
một subset/seed, không phải khẳng định nhân quả cho mọi subset 10%.

## 9. Các bẫy được code chủ động kiểm tra

| Bẫy | Kiểm tra/cách xử lý | Trạng thái bằng chứng |
|---|---|---|
| Chấm nhầm norm1000/pixel | Round trip + bộ chấm trên 9 box tham chiếu | Đã chạy; F1 đúng thang 1, sai thang 0, không phải full-test ablation |
| Mask lệch tiền tố | Assert prompt là tiền tố full sequence; decode labels | Log có 220 token ảnh và JSON + im_end |
| Quên lưu projector | Lưu extra_trainable.pt; kiểm tra key/shape khi nạp | Checkpoint E2/E3 có file riêng |
| Predictions thiếu/trùng ảnh | Kiểm tra đủ IDs, manifest, run_id và cấu hình inference | Evidence cuối có missing_predictions=0 |
| Dùng artifact khác dataset/config | Fingerprint và metadata kiểm tra trước E4 | Cell khôi phục xác minh schema legacy, dùng lại E1–E3 |
| Nhầm E4 100% là run mới | Đối chiếu run_id/metrics với E3 | Cùng run_id, metrics giống; baseline được tái sử dụng |
| CI quá lạc quan | Nêu chỉ 3 session test, bootstrap theo ảnh | Chưa chạy cluster/paired bootstrap hay nhiều seed |
| Model upstream đổi | Khóa immutable revision ở lần chạy mới | Run hiện ghi main; không sửa evidence đã chạy |

Không ghi các bẫy phòng ngừa thành lỗi đã thực sự xảy ra nếu không có traceback
hoặc log chứng minh.

## 10. Các việc còn lại khi đóng gói

`REPORT.md`, checklist và các file chẩn đoán đã được điền bằng số liệu có thật.
Không tạo output notebook hoặc metric giả để đáp ứng checklist.

- Bản `lab_day10.ipynb` trong repo hiện chưa có output; cần dùng bản Colab đã chạy
  có logs trước khi hoàn tất hồ sơ.
- `.env.example` hiện có giá trị dạng token Kaggle ở trường KAGGLE_KEY. Cần xóa
  giá trị trước khi chia sẻ; mẫu env phải để trống/placeholder. Token mới dùng
  Secrets `KAGGLE_API_TOKEN`, hoặc cặp legacy username/key đúng loại. Không đưa
  giá trị token vào báo cáo. Không có giá trị token này trong ba commit env mẫu
  đã được kiểm tra.
- Metadata thật được lưu ở `report/dataset_meta.json`; nếu thiếu file cục bộ
  `data/splits/meta.json`, công cụ vẫn WARN đối chiếu. Không commit thư mục data
  chỉ để làm mất warning.
- Phần Git chưa được xử lý theo yêu cầu; không sửa `check_submission.py` để bỏ
  điều kiện commit. Các công cụ của đề vẫn nguyên bản.

## 11. Test mẫu báo cáo mâu thuẫn với báo cáo đã điền

Sau khi điền đủ báo cáo, chạy toàn bộ pytest cho kết quả **67 passed, 1 skipped,
1 failed**. Test thất bại là
`tests/test_configs_and_docs.py::test_report_template_has_student_markers_and_ten_questions`:
test yêu cầu REPORT.md còn chuỗi marker của mẫu chưa điền. Trong khi đó
`check_submission.py` yêu cầu báo cáo không còn marker đó và hiện đã PASS phần
REPORT.md.

Giữ báo cáo đã điền, không chèn lại marker để làm test xanh, không sửa test/công
cụ của đề. Đây là xung đột giữa kiểm tra template khởi đầu và báo cáo cuối bài,
không phải lỗi scorer hay dữ liệu evidence. Các test scorer, chuẩn bị dữ liệu,
chia split và kiểm tra submission được chạy riêng: **60 passed, 1 skipped**.

## Lệnh kiểm tra lại

```bash
python report/coordinate_sanity.py
python scripts/check_submission.py
python -m pytest -q
```

Git ngoài phạm vi lần hoàn thiện này; kết quả check_submission có thể vẫn FAIL ở
mục commit dù nội dung báo cáo/evidence đã đạt. Điểm số luôn đọc từ JSON gốc,
không chỉnh JSON để phù hợp một kết luận mong muốn.
