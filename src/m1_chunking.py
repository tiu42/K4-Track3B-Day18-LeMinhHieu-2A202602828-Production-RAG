from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    # TODO: Implement semantic chunking
    # 1. from sentence_transformers import SentenceTransformer
    #    from numpy import dot
    #    from numpy.linalg import norm
    # 2. metadata = metadata or {}
    # 3. Split text thành sentences: re.split(r'(?<=[.!?])\s+|\n\n', text)
    # 4. model = SentenceTransformer("all-MiniLM-L6-v2")
    #    embeddings = model.encode(sentences)
    # 5. cosine_sim(a, b) = dot(a, b) / (norm(a) * norm(b) + 1e-9)
    # 6. Duyệt từ sentence[1]:
    #      - sim(embedding[i-1], embedding[i]) < threshold → tách chunk mới
    #      - else: gộp vào chunk hiện tại
    # 7. Return [Chunk(text=joined_group, metadata={..., "strategy": "semantic"})]

    from sentence_transformers import SentenceTransformer
    from numpy import dot
    from numpy.linalg import norm

    metadata = metadata or {}

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)

    def cosine_sim(a, b):
        return float(dot(a, b) / (norm(a) * norm(b) + 1e-9))

    groups = [[sentences[0]]]
    for i in range(1, len(sentences)):
        if cosine_sim(embeddings[i - 1], embeddings[i]) < threshold:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    return [
        Chunk(text=" ".join(group), metadata={**metadata, "chunk_index": i, "strategy": "semantic"})
        for i, group in enumerate(groups)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def _split_paragraph(para: str, child_size: int) -> list[str]:
    """Chia 1 paragraph thành các mảnh ≤ child_size mà không phá cấu trúc markdown.

    - Paragraph ngắn, hoặc chứa bảng / code block → giữ nguyên (chấp nhận vượt child_size).
    - Paragraph dài → chia theo dòng (dòng quá dài mới chia theo câu, câu quá dài mới cắt cứng).
      Mỗi mảnh mang theo header của section để không mất ngữ cảnh.
    """
    if len(para) <= child_size or re.search(r'^\s*(\||```|~~~)', para, flags=re.MULTILINE):
        return [para]

    lines = para.splitlines()
    header = lines[0].strip() if lines[0].lstrip().startswith("#") else ""
    body_size = max(child_size - len(header) - 1, child_size // 2)

    units = []
    for line in (lines[1:] if header else lines):
        line = line.strip()
        if len(line) <= body_size:
            if line:
                units.append(line)
            continue
        for sent in re.split(r'(?<=[.!?])\s+', line):
            while len(sent) > body_size:
                units.append(sent[:body_size])
                sent = sent[body_size:].strip()
            if sent:
                units.append(sent)

    groups, current = [], ""
    for unit in units:
        if current and len(current) + len(unit) + 1 > body_size:
            groups.append(current)
            current = ""
        current = f"{current}\n{unit}" if current else unit
    if current:
        groups.append(current)
    return [f"{header}\n{g}" if header else g for g in groups]


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    # TODO: Implement hierarchical chunking
    # 1. metadata = metadata or {}
    # 2. Split text bằng "\n\n" → paragraphs
    # 3. Gộp paragraphs thành parent chunks (mỗi parent ≤ parent_size chars):
    #      pid = f"parent_{len(parents)}"
    #      parents.append(Chunk(text=..., metadata={..., "chunk_type": "parent", "parent_id": pid}))
    # 4. Mỗi parent → split thành children (mỗi child ≤ child_size chars):
    #      children.append(Chunk(text=..., metadata={..., "chunk_type": "child"}, parent_id=pid))
    # 5. return (parents, children)
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Gộp paragraphs thành parent texts (mỗi parent ≤ parent_size, trừ khi 1 paragraph đã dài hơn)
    parent_texts = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > parent_size:
            parent_texts.append(current)
            current = ""
        current = f"{current}\n\n{para}" if current else para
    if current:
        parent_texts.append(current)

    parents, children = [], []
    for parent_text in parent_texts:
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(text=parent_text,
                             metadata={**metadata, "chunk_type": "parent", "parent_id": pid}))

        # Split parent theo paragraph (giữ nguyên markdown), gộp thành children ≤ child_size
        pieces = []
        for para in parent_text.split("\n\n"):
            if para.strip():
                pieces.extend(_split_paragraph(para.strip(), child_size))

        current = ""
        for piece in pieces:
            if current and len(current) + len(piece) + 2 > child_size:
                children.append(Chunk(text=current,
                                      metadata={**metadata, "chunk_type": "child",
                                                "child_index": len(children)},
                                      parent_id=pid))
                current = ""
            current = f"{current}\n\n{piece}" if current else piece
        if current:
            children.append(Chunk(text=current,
                                  metadata={**metadata, "chunk_type": "child",
                                            "child_index": len(children)},
                                  parent_id=pid))

    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    # TODO: Implement structure-aware chunking
    # 1. metadata = metadata or {}
    # 2. sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    # 3. Duyệt sections:
    #      - Nếu match header (^#{1,3}\s+): lưu header hiện tại, tạo chunk cho content trước đó
    #      - Else: gộp vào content hiện tại
    # 4. Return [Chunk(text=header+content, metadata={..., "section": header, "strategy": "structure"})]
    metadata = metadata or {}
    header_re = re.compile(r'^(#{1,3})\s+(.+?)\s*#*\s*$')

    chunks = []
    header_line, section = "", ""
    body: list[str] = []
    in_code = False

    def flush():
        content = "\n".join(body).strip()
        if not content:  # header không có nội dung (vd. "# Nghỉ phép" ngay trước "## ...") → bỏ qua
            return
        chunk_text = f"{header_line}\n\n{content}" if header_line else content
        chunks.append(Chunk(text=chunk_text,
                            metadata={**metadata, "section": section,
                                      "chunk_index": len(chunks), "strategy": "structure"}))

    # Duyệt theo dòng thay vì re.split để không nhầm "# comment" trong code block là header
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
        m = None if in_code else header_re.match(line)
        if m:
            flush()
            header_line, section = line.strip(), m.group(2)
            body = []
        else:
            body.append(line)
    flush()

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
