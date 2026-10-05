# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Lê Minh Hiếu  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 05/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm/lớp cụ thể | Observation & Phân tích |
|----------------|--------|----------------|--------------------------|
| Document loading và PDF extraction | M1 | `load_documents()` và `_extract_pdf_text()` | Nạp Markdown/PDF cùng metadata nguồn trước khi chia chunk; metadata giúp truy vết citation và phiên bản tài liệu. |
| Semantic chunking | M1 | `chunk_semantic()` | Gom các câu có độ tương đồng vượt `SEMANTIC_THRESHOLD`; cách này giữ chủ đề tốt hơn cắt theo số ký tự cố định. |
| Hierarchical / parent-child chunking | M1 | `chunk_hierarchical()` và `_split_paragraph()` | Child chunk dùng để tìm kiếm với độ chính xác cao, sau đó đổi sang parent để LLM nhận đủ bảng hoặc điều khoản không bị cắt giữa chừng. |
| Vietnamese tokenization | M2 | `segment_vietnamese()` và `_tokenize()` | `underthesea` tách từ tiếng Việt; thay `_` bằng khoảng trắng để truy vấn “nghỉ phép” khớp với tài liệu. |
| BM25 lexical search | M2 | `BM25Search.index()` và `BM25Search.search()` | BM25 bắt các từ khóa chính xác như “120 ngày”, “CEO”, “15 triệu”, hữu ích khi câu hỏi có số và tên phiên bản. |
| Dense retrieval | M2 | `DenseSearch.index()` và `DenseSearch.search()` | `BAAI/bge-m3` chuyển chunk/query thành vector và Qdrant trả về các đoạn tương đồng ngữ nghĩa. |
| Hybrid search + Reciprocal Rank Fusion | M2 | `HybridSearch.search()` và `reciprocal_rank_fusion()` | RRF hợp nhất thứ hạng BM25 và dense mà không cần chuẩn hóa hai loại score, giúp giảm lỗi khi lexical hoặc semantic search đơn lẻ bỏ sót. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Chấm lại các cặp query-document bằng `BAAI/bge-reranker-v2-m3`; pipeline lấy các parent khác nhau ở top-K để giảm context trùng lặp. |
| Contextual enrichment / HyDE-like HyQA | M5 | `enrich_chunks()`, `_enrich_single_call()`, `generate_hypothesis_questions()` | Bổ sung header tài liệu, câu hỏi giả thuyết và metadata trước khi index để thu hẹp khoảng cách từ vựng giữa câu hỏi và chunk. |
| Contextual prepend | M5 / pipeline | `_document_header()` và `contextual_prepend()` | Header chứa tên và phiên bản tài liệu; điều này đặc biệt quan trọng với cặp v1/v2 như chính sách mật khẩu và phép năm. |
| RAG generation grounding | Pipeline | `run_query()` | LLM chỉ được yêu cầu trả lời dựa trên context, trả lời “Không tìm thấy” nếu context không đề cập; đây là điểm cần tăng cường sau các failure faithfulness. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` và `failure_analysis()` | Đánh giá faithfulness, answer relevancy, context precision, context recall; `failure_analysis()` tìm metric thấp nhất và gán diagnosis cho từng câu. |
| Failure reporting | M4 | `save_report()` | Lưu aggregate và Bottom-N failures vào `reports/ragas_report.json`; lần sau nên lưu thêm answer/context để điều tra nguyên nhân đến tận runtime. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải:**
  - Báo cáo RAGAS chỉ có aggregate score và danh sách diagnosis, không có answer/context cho từng câu. Vì vậy không thể kiểm chứng trực tiếp citation runtime của Bottom-5.
  - Hai bộ tài liệu có phiên bản cũ và hiện hành cùng chứa từ khóa giống nhau, ví dụ mật khẩu 90/120 ngày và phép năm 12/15 ngày. Điều này làm context precision giảm.
  - Vietnamese word segmentation nối từ ghép bằng dấu `_`, trong khi BM25 query thường tách bằng khoảng trắng, gây nguy cơ không match “nghỉ_phép” với “nghỉ phép”.
  - Cross-encoder có thể gặp lỗi tương thích nếu dùng nhầm `FlagEmbedding`; code đã chọn `sentence_transformers.CrossEncoder` để tương thích với transformers hiện tại.

- **Nguyên nhân gốc rễ & Cách debug:**
  - Đọc failure report, đối chiếu từng ground truth trong `test_set.json` với tài liệu nguồn trong `data/`, sau đó kiểm tra đường đi `M1 → M2 → M3 → M5 → pipeline → M4`.
  - Xác định hai nhóm lỗi: faithfulness thấp là vấn đề grounding/generation; context precision thấp là vấn đề retrieval, reranking hoặc thiếu filter phiên bản.
  - Kiểm tra các hàm `_tokenize()`, `reciprocal_rank_fusion()`, `CrossEncoderReranker.rerank()` và prompt trong `run_query()` thay vì chỉ nhìn aggregate score.

- **Kiến thức còn thiếu & Cách khắc phục:**
  - Cần hiểu sâu hơn cách RAGAS parse câu trả lời và citation để phân biệt “answer sai” với “context đúng nhưng LLM diễn đạt sai”.
  - Cần bổ sung tracing theo từng câu: lưu question, answer, contexts, ground truth, scores và version metadata; sau đó tạo test boundary cho ngưỡng tiền, ngày và phiên bản.
  - Cần thử metadata filter/reranking có ưu tiên tài liệu `current` hoặc ngày hiệu lực mới nhất trước khi tăng số lượng chunk đưa vào LLM.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Trợ lý hỏi đáp tài liệu nội bộ cho nhóm dự án

#### 1. Hiện trạng
- **Pipeline hiện tại:** Tài liệu được chunk, embed và tìm bằng dense retrieval; một số câu hỏi còn phụ thuộc vào từ khóa nên có thể bỏ sót các điều khoản chứa số, tên riêng hoặc phiên bản.
- **Vấn đề / Bottlenecks đang gặp:** Context bị nhiễu giữa tài liệu cũ/mới, câu trả lời có nguy cơ hallucination khi phải tính toán, và chưa có đủ log answer/context để debug một failure cụ thể.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Dùng structure-aware kết hợp hierarchical parent-child; giữ nguyên tiêu đề, bảng và phạm vi điều khoản để không tách mất ngữ cảnh.
2. **Search retrieval:** Dùng hybrid BM25 + dense với RRF; BM25 xử lý số và từ khóa chính xác, dense xử lý cách diễn đạt khác nhau.
3. **Reranking:** Dùng cross-encoder cho top-20 candidate, lấy top-3 parent khác nhau; thêm metadata filter theo `document_type`, `version` và `effective_date`.
4. **Evaluation:** Dùng RAGAS 4 metrics cùng một bộ benchmark có ground truth; theo dõi thêm accuracy cho câu trả lời số và tỷ lệ citation chứa đáp án.
5. **Enrichment:** Dùng contextual prepend và HyQA; sinh metadata chủ đề, entity, phiên bản nhưng không đưa nội dung do LLM tự bịa vào context trả lời.
6. **Grounding:** Yêu cầu output theo cấu trúc “kết luận → điều khoản → phép tính → nguồn”; nếu không có bằng chứng thì trả lời không tìm thấy thay vì suy đoán.

#### 3. Timeline triển khai
- **Tuần 1:** Chuẩn hóa tài liệu, thêm metadata phiên bản/ngày hiệu lực, triển khai hierarchical chunking và hybrid retrieval.
- **Tuần 2:** Thêm cross-encoder reranking, contextual/HyQA enrichment và filter tài liệu hiện hành.
- **Tuần 3:** Xây benchmark 30–50 câu, lưu trace per-question, chạy RAGAS và phân loại Bottom-5 theo error tree.
- **Tuần 4:** Tối ưu prompt grounding, thêm test regression cho số tiền/ngày/phiên bản, đo latency và triển khai thử cho nhóm nhỏ.

