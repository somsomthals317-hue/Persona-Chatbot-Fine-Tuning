# Chroma VectorDB + BGE-M3 + Hybrid RAG verification

This ZIP contains BOTH Base and Fine-Tuned deployments. The only inference engine
for generation and the `RAG/CHAT` routing step is **vLLM**, using the model ID
specified in each `app.py`. BGE-M3 is an independent CPU-only embedding model.

## What changed
- Chroma `PersistentClient` replaces the in-memory dense-vector array.
- Chroma HNSW cosine similarity uses BGE-M3 embeddings and caches the story
  corpus in the existing Modal Volume; subsequent restarts reuse the index.
- Character n-gram TF-IDF searches the same story text to improve recall of
  paraphrased Korean questions; reciprocal-rank fusion combines ranks.
- Both original wording and the router's search query are retrieved.
- Relevant passages are included in the latest user message only if dense
  similarity OR lexical relevance meets the thresholds (0.38 or 0.12).
- On cold start, the `/health` endpoint becomes ready only after Chroma indexes
  all documents and a penalty-box retrieval specifically finds the passage
  mentioning Bree, Grace and apologizing, and vLLM generates warm-up tokens.
- Temperature 0.8; same top_p=0.9, repetition_penalty=1.0, max_tokens=384,
  seed=42; 3 Few-shot demos; recent 8 completed dialogue pairs.

## Windows CMD deployment (Fine-Tuned)
```
cd Fine-Tuned
py -m modal deploy app.py
py app.py
```
For Base comparison, deploy Base separately and ensure the intended Modal API
URL and token point to that variant. Both variants currently share the same
Modal app name; deploying either can replace the other's endpoint. Run one at a time.

## Check `/health` (authenticated? GET /health is public)
`build` must be `qwen-v14-chroma-rag-natural-voice-t08`,
`rag_vector_store` must be `chroma_persistent`, `rag_ready` true,
`rag_document_count` > 0, and `engine` must be `vllm`.

## Check actual retrieval with this test in a NEW chat
`라일리가 페널티 박스에서 나온 후 누구에게 사과했어?`
Expected **document evidence**: episode2 passage mentions `브리와 그레이스`.
Expected answer: Bree and Grace, without repeatedly asking who they are.

Use Modal logs:
```
py -m modal app logs anxiety-chatbot
```
Look for `[rag-router] route=llm_rag retrieve=True`,
`[rag-retrieval] backend=chroma outcome=selected` and `episode2:86`.
`[rag-index] state=built` happens at first deployment, `state=cached`
should happen on later deployments with the same story corpus and model.
If `route=llm_chat`, the VectorDB was NOT consulted on that turn — a router
mistake, not a database mistake. If retrieval selects unrelated documents,
inspect the source IDs and the dense / lexical scores before tuning thresholds.

The included Python code has been statically checked, but live Modal GPU
model quality and actual Chroma dependency resolution must be verified after
re-deployment. It is not guaranteed that every natural-language question is
correctly routed or answered.
