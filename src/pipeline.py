from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K

_llm_client = None


def _document_header(text: str, source: str) -> str:
    """Tiêu đề H1 + dòng phiên bản ("> Phiên bản: ... | Trạng thái: ...") của tài liệu.

    Dùng làm contextual prepend thay cho câu context do LLM sinh: không tốn API và
    phân biệt được các phiên bản (vd. mat_khau_v1 "ĐÃ THAY THẾ" vs v2 "hiện hành").
    """
    lines = [l.strip() for l in text.splitlines()[:5]]
    title = next((l.lstrip("# ") for l in lines if l.startswith("# ")), source)
    version = next((l.lstrip("> ") for l in lines if l.startswith(">")), "")
    return f"{title} | {version}" if version else title


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    parent_texts = {}  # parent_key → parent text (để swap child → parent lúc query)
    doc_headers = []   # header tài liệu tương ứng từng child
    for doc in docs:
        source = doc["metadata"].get("source", "")
        header = _document_header(doc["text"], source)
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        # parent_id ("parent_0", ...) chỉ unique trong 1 tài liệu → ghép thêm source
        for p in parents:
            parent_texts[f"{source}::{p.metadata['parent_id']}"] = p.text
        for child in children:
            all_chunks.append({"text": child.text,
                               "metadata": {**child.metadata, "parent_id": child.parent_id,
                                            "parent_key": f"{source}::{child.parent_id}"}})
            doc_headers.append(header)
    print(f"  ✓ {len(all_chunks)} child chunks, {len(parent_texts)} parents "
          f"from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    # Text để index = header tài liệu + child + HyQA questions. Không dùng context line do LLM
    # sinh (quá chung chung, không phân biệt phiên bản). LLM chỉ nhìn thấy parent text nên
    # header / questions ở đây chỉ phục vụ retrieval.
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = []
        for e, header in zip(enriched, doc_headers):
            text = f"{header}\n\n{e.original_text}"
            if e.hypothesis_questions:
                text += "\n\n" + "\n".join(e.hypothesis_questions)
            all_chunks.append({"text": text, "metadata": e.auto_metadata})
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)
        all_chunks = [{"text": f"{h}\n\n{c['text']}", "metadata": c["metadata"]}
                      for c, h in zip(all_chunks, doc_headers)]

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    search.parent_texts = parent_texts
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    # Rerank toàn bộ candidates (child) rồi lấy top-K parent khác nhau:
    # retrieve bằng child (precision) → đưa parent cho LLM (đủ context, không bị cắt giữa bảng/section)
    reranked = reranker.rerank(query, docs, top_k=len(docs)) or results
    parent_texts = getattr(search, "parent_texts", {})
    contexts = []
    for hit in reranked:
        context = parent_texts.get(hit.metadata.get("parent_key"), hit.text)
        if context not in contexts:
            contexts.append(context)
        if len(contexts) == RERANK_TOP_K:
            break

    from config import OPENAI_API_KEY
    if OPENAI_API_KEY and contexts:
        try:
            global _llm_client
            if _llm_client is None:
                from openai import OpenAI
                _llm_client = OpenAI()
            context_str = "\n\n---\n\n".join(contexts)
            resp = _llm_client.chat.completions.create(model="gpt-4o-mini", temperature=0, messages=[
                {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu context nói rõ là KHÔNG được / KHÔNG cần "
                                              "thì trả lời 'Không' kèm trích dẫn quy định. Chỉ nói 'Không tìm thấy.' "
                                              "khi context hoàn toàn không đề cập đến vấn đề được hỏi."},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
