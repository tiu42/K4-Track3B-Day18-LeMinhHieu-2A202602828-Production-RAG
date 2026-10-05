# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Lê Minh Hiếu
**Khóa:** K4 - Track 3B

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8238 | 0.8750 | +0.0512 |
| Answer Relevancy | 0.7171 | 0.8698 | +0.1527 |
| Context Precision | 0.9250 | 0.9000 | -0.0250 |
| Context Recall | 0.9250 | 0.9333 | +0.0083 |

## Bottom-5 Failures

### #1
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày; phí 2%/tháng trên 15.000.000 VNĐ là 300.000 VNĐ/tháng, tính pro-rata khoảng 50.000 VNĐ.
- **Got:** Không có answer runtime trong report; faithfulness = 0.0, avg = 0.6262, chẩn đoán là `LLM hallucinating`, nên câu trả lời được xem là không đủ căn cứ/không đáng tin.
- **Worst metric:** Faithfulness (0.0)
- **Câu trả lời có đúng không?** Chưa thể xác nhận từng chữ, nhưng điểm faithfulness 0 cho thấy câu trả lời không bám được vào bằng chứng; cần coi là sai hoặc ít nhất không chấp nhận được.
- **Các đoạn trích dẫn có chứa đáp án không?** Tài liệu [tam_ung.md](../data/tam_ung.md) có đủ thời hạn 15 ngày, phí 2%/tháng và số tiền 15 triệu để suy ra đáp án. Tuy nhiên report không lưu các context thực tế nên không biết chúng có được truyền vào LLM hay không.
- **Câu hỏi có cần viết lại không?** Không. Câu hỏi đã nêu rõ số tiền và số ngày; chỉ nên chuẩn hóa “phí quá hạn” và yêu cầu tính pro-rata nếu muốn giảm cách hiểu khác nhau.
- **Cần sửa module nào?** `pipeline.py` — prompt sinh đáp án và ràng buộc phải trích dẫn/calculate từ context; đặt temperature 0 (đã có) và không cho suy đoán. Có thể bổ sung kiểm thử tính toán ở M4.
- **Error Tree:** Output sai → Context nguồn đúng, context runtime chưa xác minh → Query rõ → lỗi chính ở generation/grounding.
- **Root cause:** LLM không thực hiện phép tính hoặc sinh thêm thông tin ngoài quy định dù nguồn có đáp án.
- **Suggested fix:** Tighten prompt, buộc nêu công thức và citation; lưu answer/context theo từng câu để debug lần sau.

### #2
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Hoàn trả 100% chi phí, tức 25.000.000 VNĐ, vì cam kết tối thiểu 1 năm.
- **Got:** Faithfulness = 0.5, avg = 0.8185; report chẩn đoán `LLM hallucinating`, nhưng không lưu câu trả lời runtime.
- **Worst metric:** Faithfulness (0.5)
- **Câu trả lời có đúng không?** Có khả năng chỉ đúng một phần hoặc diễn đạt thêm chi tiết không có trong nguồn; chưa đủ bằng chứng để kết luận exact answer.
- **Các đoạn trích dẫn có chứa đáp án không?** Có: [hoan_chi_dao_tao.md](../data/hoan_chi_dao_tao.md) ghi cam kết 1 năm và hoàn trả 100% khi nghỉ trước hạn. Runtime context cụ thể không được lưu.
- **Câu hỏi có cần viết lại không?** Không; dữ kiện “25 triệu” và “8 tháng” đủ để áp dụng quy tắc.
- **Cần sửa module nào?** `pipeline.py` (generation prompt/grounded answer), sau đó `m4_eval.py` nên lưu per-question answer và contexts.
- **Error Tree:** Output chưa grounded → Context nguồn đúng → Query rõ → generation.
- **Root cause:** Mô hình không nối điều kiện thời hạn với phép tính 100% × 25 triệu một cách ổn định.
- **Suggested fix:** Prompt yêu cầu trích đúng điều khoản, nêu phép tính và trả lời “không tìm thấy” nếu thiếu điều khoản.

### #3
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** 15 ngày có lương theo v2024; v2023 là 12 ngày nhưng đã bị thay thế.
- **Got:** Context precision = 0.5, avg = 0.8259; chẩn đoán `Too many irrelevant chunks`.
- **Worst metric:** Context precision (0.5)
- **Câu trả lời có đúng không?** Không có answer runtime để xác minh; đáp án đúng phải ưu tiên v2024 và phân biệt v2023 đã hết hiệu lực.
- **Các đoạn trích dẫn có chứa đáp án không?** Corpus có đáp án trong [nghi_phep_nam_v2024.md](../data/nghi_phep_nam_v2024.md), nhưng điểm precision cho thấy context có thể lẫn đoạn không liên quan hoặc phiên bản cũ; report không cho biết chính xác chunk nào được trích.
- **Câu hỏi có cần viết lại không?** Nên viết rõ “chính sách hiện hành” để tránh đồng thời lấy 12 và 15 ngày từ hai phiên bản.
- **Cần sửa module nào?** `m2_search.py`/`m3_rerank.py`: lọc metadata phiên bản hiện hành và rerank ưu tiên tài liệu v2024; giữ `pipeline.py` để hiển thị citation.
- **Error Tree:** Output chưa xác minh → Context có đáp án nhưng precision thấp → Query hơi mơ hồ về phiên bản → retrieval/reranking.
- **Root cause:** Hai tài liệu v2023/v2024 cùng chứa cụm “phép năm”, khiến candidate set có nhiễu.
- **Suggested fix:** Metadata filter `effective/current`, reranking theo version và contextual header.

### #4
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** Faithfulness = 0.5, avg = 0.8266; chẩn đoán `LLM hallucinating`.
- **Worst metric:** Faithfulness (0.5)
- **Câu trả lời có đúng không?** Chưa thể xác minh answer runtime; đáp án chuẩn là CEO, không phải Director.
- **Các đoạn trích dẫn có chứa đáp án không?** [mua_sam.md](../data/mua_sam.md) có đúng ngưỡng “trên 50 triệu → CEO”. Vì report không lưu context, chưa thể khẳng định citation đã chứa bảng này.
- **Câu hỏi có cần viết lại không?** Không bắt buộc; có thể thêm “theo quy trình mua sắm hiện hành” để giới hạn tài liệu.
- **Cần sửa module nào?** `pipeline.py`: buộc mô hình đối chiếu ngưỡng số tiền với bảng và trích dẫn dòng tương ứng; `m3_rerank.py` nếu bảng bị xếp sau đoạn quy trình.
- **Error Tree:** Output không grounded → Context nguồn đúng, runtime chưa biết → Query rõ → generation/reranking.
- **Root cause:** Mô hình dễ nhầm ngưỡng 5–50 triệu (Director) với trên 50 triệu (CEO).
- **Suggested fix:** Prompt yêu cầu nêu khoảng ngưỡng trước khi chọn người phê duyệt và thêm test boundary 50/50.000.001 triệu.

### #5
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Mỗi 120 ngày theo v2.0; quy định cũ 90 ngày đã bị thay thế.
- **Got:** Context precision = 0.5, avg = 0.8307; chẩn đoán `Too many irrelevant chunks`.
- **Worst metric:** Context precision (0.5)
- **Câu trả lời có đúng không?** Không có answer runtime; câu trả lời chỉ đúng nếu chọn 120 ngày và loại bỏ 90 ngày cũ.
- **Các đoạn trích dẫn có chứa đáp án không?** [mat_khau_v2.md](../data/mat_khau_v2.md) ghi 120 ngày và [mat_khau_v1.md](../data/mat_khau_v1.md) ghi 90 ngày nhưng đã bị thay thế. Precision thấp cho thấy citation có thể thừa/nhầm phiên bản; runtime context không được lưu.
- **Câu hỏi có cần viết lại không?** Nên thêm “theo chính sách hiện hành” để loại ambiguity giữa v1.0 và v2.0.
- **Cần sửa module nào?** `m2_search.py`/`m3_rerank.py`: lọc và ưu tiên version 2.0; `pipeline.py`: yêu cầu nêu rõ văn bản v2.0 thay thế v1.0.
- **Error Tree:** Output chưa xác minh → Context corpus đúng nhưng precision thấp → Query thiếu “hiện hành” → retrieval/reranking.
- **Root cause:** Hai phiên bản có cùng chủ đề và cùng dạng câu trả lời số ngày.
- **Suggested fix:** Dùng metadata effective date/version trong filter, rerank theo trạng thái hiện hành, và citation bắt buộc.

## Case Study (cho presentation)

**Question chọn phân tích:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?

**Error Tree walkthrough:**
1. Output đúng? → Faithfulness 0.0 nên không chấp nhận output nếu không có công thức và citation.
2. Context đúng? → Tài liệu nguồn đúng; context runtime không được lưu nên cần bổ sung tracing.
3. Query rewrite OK? → Có; câu hỏi đã đủ rõ.
4. Fix ở bước: generation grounding trong `pipeline.py`, sau đó logging/evaluation trong `m4_eval.py`.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm schema lưu `answer`, `contexts`, `ground_truth` và 4 metric cho từng câu.
- Thử prompt có cấu trúc: điều khoản → phép tính → kết luận → citation.
- Thêm test boundary và version filter cho các chính sách có tài liệu cũ/mới.
