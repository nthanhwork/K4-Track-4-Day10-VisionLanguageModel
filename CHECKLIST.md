# Checklist Lab Day 10

Họ tên: **Nguyễn Thái Anh**  
Dataset: **`d10-21e4070301`**

Checklist này phản ánh file và evidence hiện có trong repo. Không đánh dấu hoàn tất
các bước chưa có bằng chứng. Phần Git không được xử lý theo yêu cầu của người làm bài.

## Dữ liệu và quy ước

- [x] Metadata split đã lưu ở `report/dataset_meta.json`, khớp dataset_version của cả sáu evidence.
- [x] Chia theo session: train 13, val 2, test 3; không chia ngẫu nhiên frame.
- [x] Đã ghi số ảnh và box từng lớp: train 1.422/3.515, val 276/639, test 311/583.
- [x] Dùng JSON list, ba lớp `pothole`, `crack`, `manhole`, tọa độ norm1000.
- [x] Đã xử lý nhãn chiều rộng/cao 0 trong target; giữ nguyên GT và bộ chấm của đề.
- [x] Kiểm chứng hình học đổi thang: `python report/coordinate_sanity.py`; đúng thang F1=1, sai thang F1=0 trên 9 box tham chiếu.
- [ ] Chưa chấm lại output model với sai thang trên toàn bộ 311 ảnh test; báo cáo nêu đúng giới hạn này.

## Model, loss và checkpoint

- [x] E0 không train; E1 LoRA LLM; E2 thêm projector; E3 thêm LoRA ViT.
- [x] Visual/lm_head được loại khỏi NF4 bằng đúng tên `model.visual`, `lm_head` và có assertion kiểm tra.
- [x] Mask prompt, token ảnh và padding bằng labels=-100; log xác nhận 220 token ảnh và đáp án kết thúc bằng `<|im_end|>`.
- [x] E1/E2/E3 hoàn tất 300 bước, batch 1 × accumulation 8; log ghi 0 bước overflow bị bỏ qua.
- [x] Các tham số trainable được giữ fp32; forward dùng autocast fp16, GradScaler, clipping 1.0.
- [x] Adapter LoRA và projector được lưu riêng; run độc lập xác minh artifact trước khi dùng lại.
- [x] Đã ghi rõ run dở dang train lại từ đầu; không gọi đây là resume optimizer.
- [ ] Revision vẫn được ghi `main`, chưa có immutable commit hash; không sửa metadata của run cũ để che giới hạn tái lập.

## Evidence và tài nguyên

- [x] Có `report/evidence/e0.json`, `e1.json`, `e2.json`, `e3.json`.
- [x] Có `e4_data_size_0.1.json` và `e4_data_size_1.0.json`.
- [x] Cả sáu evidence dùng test, 311 ảnh, không thiếu predictions, cùng dataset_version và score_version=1.
- [x] E1–E3 có trainable_params, peak_vram_gb, train_minutes là số dương.
- [x] E1 có parse_valid_rate=0,993569, đạt yêu cầu ≥0,95.
- [x] F1 E1−E0=0,207760, đạt yêu cầu ≥0,10.
- [x] E4 chỉ đổi data_size 10%/100%, giữ 300 bước; baseline chọn từ val preview trước E4.
- [x] E4 100% dùng lại E3, cùng run_id; không coi đây là replicate mới.
- [x] Bảng tổng hợp có params, VRAM, thời gian, F1, CI95, FP/ảnh và F1 từng lớp; đối chiếu với JSON nguồn.
- [x] Nêu rõ thời gian E0 là inference, các hàng train là thời gian run thành công; không cộng lại run dở dang hay toàn bộ thời gian phiên.
- [x] Nêu giới hạn của CI bootstrap theo ảnh, ba session test và một seed train.

## Báo cáo và hồ sơ

- [x] `REPORT.md` có họ tên Nguyễn Thái Anh, dataset_version và đủ 10 câu trả lời.
- [x] Không còn placeholder trong `REPORT.md`.
- [x] Có ví dụ output E0 thật và chẩn đoán 162 output hợp lệ/149 output không hợp lệ.
- [x] Có ít nhất 6 ảnh minh họa trong báo cáo: 3 đúng, 3 có lỗi; chú thích TP/FP/FN và JSON validity.
- [x] Repo có 30 ảnh dự đoán và 4 biểu đồ loss; các PNG đọc được.
- [x] Có `CHECKLIST.md` và `BAO_CAO_LOI_VA_LUU_Y.md`.
- [x] Giữ nguyên `scripts/`, `tests/`, `configs/` của đề.
- [x] Không có dữ liệu/thư mục work/trọng số bị Git theo dõi theo kiểm tra của script.
- [x] `check_submission.py` đã PASS REPORT.md và tất cả mục evidence/meta; đã đối chiếu các liên kết, số liệu và sáu ảnh trong báo cáo.
- [ ] Một test gốc kiểm tra REPORT.md còn marker của mẫu chưa điền nên FAIL sau khi hoàn thiện báo cáo. Giữ nguyên test; xem mục 11 của báo cáo lỗi.
- [ ] Thay `lab_day10.ipynb` trong repo bằng bản Colab đã chạy có output; bản notebook hiện tại trong repo chưa có logs. Không tự ghi lại execution_count hoặc tạo output giả.
- [ ] Xóa giá trị token khỏi `.env.example` trước khi chia sẻ repo; dùng Colab Secrets/biến môi trường. Không ghi token vào bất kỳ notebook hoặc báo cáo nào.
- [ ] `python scripts/check_submission.py` chưa đạt hoàn toàn: phần Git vẫn ngoài phạm vi xử lý; khi không có `data/splits/meta.json` cục bộ, script còn WARN đối chiếu split.

## Các file dùng để đối chiếu

- [Báo cáo](REPORT.md), [báo cáo lỗi](BAO_CAO_LOI_VA_LUU_Y.md).
- [Bảng tổng hợp](report/results_table.md), [evidence](report/evidence/).
- [Metadata split](report/dataset_meta.json), [lựa chọn E4](report/e4_selection_full.json).
- [Chẩn đoán E0](report/e0_diagnostics.json), [log/quan sát run](report/run_observations.json).
- [Kiểm tra tọa độ](report/coordinate_sanity.py), [kết quả kiểm tra](report/coordinate_sanity.json).

Khi lấy artifact từ Drive, `report_bundle.zip` đủ cho evidence, ảnh, bảng và metadata.
Notebook có logs tải riêng từ Colab. `REPORT.md` trong ZIP xuất trước khi hoàn thiện
vẫn là mẫu; dùng bản đã điền trong repo này.
