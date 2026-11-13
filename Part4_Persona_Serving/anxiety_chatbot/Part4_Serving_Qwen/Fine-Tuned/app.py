import base64
import hmac
import hashlib
import json
import os
import re
from pathlib import Path

import modal

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
ASSET_DIR = APP_DIR / "assets"

# Load the pretrained Fine-Tuned weights already merged in Part3.
# Everything else remains identical to the v8 direct-adapter comparison.
MODEL_ID = "somsom317/persona-qwen25-7b-merged-final"  # Pinned to evaluated merged checkpoint
MODEL_VARIANT_LABEL = "FINE-TUNED (MERGED LoRA)"
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
MAX_GENERATION_TOKENS = 384
MAX_PROMPT_TOKENS = 6144
# Sampled decoding for expressive Anxiety persona; comparable to Part2 do_sample=True.
MAX_HISTORY_EXCHANGES = 8  # Last 8 completed real turns, this browser session only.
SAMPLING_TEMPERATURE = 0.8
MODAL_APP_NAME = "anxiety-chatbot"
SERVING_BUILD = "qwen-v14-chroma-rag-natural-voice-t08"
MODAL_SECRET_NAME = "anxiety-chatbot-finetuned-app-token"
# Same Secret as the existing Fine-Tuned deployment. This deployment replaces the old merged-model app.
MODAL_VOLUME_NAME = "anxiety-chatbot-cache"

INITIAL_GREETING = (
    "어어?! 왔구나!!! 잠깐... 혹시 무슨 걱정 있어???\n\n"
    "...아니지!! 내가 먼저 당황하면 안 되지!\n"
    "좋아, 오늘 제일 신경 쓰이는 것부터 이야기해 줘!!!"
)



# Output guard for rare Chinese-ideograph spillover from the multilingual Qwen model.
_HANZI_RE = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002FA1F]"
)


def enforce_korean_output(reply):
    # Drop isolated ideographs such as '알아踽' without changing the Korean sentence.
    return _HANZI_RE.sub("", reply).strip()


def has_multiple_hanzi(reply):
    return len(_HANZI_RE.findall(reply)) >= 2



def is_valid_api_token(received_token, expected_token):
    """Compare UTF-8 bytes so malformed Unicode cannot trigger server HTTP 500."""
    if not expected_token or not received_token:
        return False
    return hmac.compare_digest(
        received_token.encode("utf-8"),
        expected_token.encode("utf-8"),
    )


modal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "TOKENIZERS_PARALLELISM": "false"})
    .uv_pip_install(
        "vllm==0.11.1",
        "transformers==4.56.2",  # tokenizer/chat template ONLY; no Transformers inference
        "sentence-transformers>=5.1,<6",
        "sentencepiece>=0.2,<1",
        "numpy>=1.26,<3",
        "chromadb>=1.3,<2",
        "fastapi[standard]>=0.115,<1",
    )
    .add_local_dir(str(DATA_DIR), remote_path="/app/data")
)
web_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "fastapi[standard]>=0.115,<1",
)

app = modal.App(MODAL_APP_NAME)
model_cache = modal.Volume.from_name(
    MODAL_VOLUME_NAME,
    create_if_missing=True,
)
modal_secret = modal.Secret.from_name(MODAL_SECRET_NAME)

_runtime = None


@app.function(secrets=[modal_secret])
def secret_status():
    return {
        "hf_token": bool(os.environ.get("HF_TOKEN")),
        "app_api_token": bool(os.environ.get("APP_API_TOKEN")),
    }


def read_blocks(data_dir, filename):
    text = (data_dir / filename).read_text(encoding="utf-8")
    return [
        block.strip()
        for block in re.split(r"\n\s*\n", text)
        if block.strip()
    ]


def load_story_documents(data_dir):
    documents = []

    for filename, source in [
        ("characters.txt", "characters"),
        ("episodes1.txt", "episode1"),
        ("episodes2.txt", "episode2"),
    ]:
        for index, content in enumerate(
            read_blocks(data_dir, filename),
            start=1,
        ):
            documents.append(
                {
                    "source": source,
                    "index": index,
                    "content": content,
                }
            )

    return documents


def clean_history(history):
    rows = []

    for row in history or []:
        if not isinstance(row, dict):
            continue

        role = row.get("role")
        content = row.get("content")

        if role in {"user", "assistant"} and isinstance(content, str):
            text = content.strip()

            if text:
                rows.append({"role": role, "content": text})

    return rows


# A punctuation cluster is a possible sentence boundary, not a guaranteed one.
# In particular, "어떡하지??? 하는 걱정" is a single sentence.
_SENTENCE_BOUNDARY = re.compile(r'''([.!?…]+(?:[\\"'”’\)\]]+)?)[ \t]+(?=\S)''')
_KOREAN_CONTINUATIONS = (
    "하는", "하던", "했던", "한다고", "하라고", "하고", "하며", "하면서",
    "라는", "이라고", "라고", "이란", "란", "이라며", "라며", "라니",
    "같은", "처럼", "이라는", "이었던", "이었단", "였다는",
    "건", "거란", "것이라는", "때문에", "만큼", "듯이", "등은",
)
_ENGLISH_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "e.g", "i.e", "vs", "etc"}


def _break_one_line_naturally(line):
    """Insert a newline only at a high-confidence sentence boundary.

    Leave Markdown list and table lines untouched. Preserve emotive punctuation
    exactly, and do not split quoted/thought fragments followed by a Korean
    grammatical continuation ("??? 하는", "?! 라고", etc.).
    """
    stripped = line.lstrip()
    if (not stripped or stripped.startswith(("```", "~~~", "|", ">"))
            or re.match(r"(?:[-*+]\s|\d+[.)]\s|#{1,6}\s)", stripped)):
        return line

    # Protect Markdown inline code and links: inserting newlines inside them
    # can break both rendered text and click targets.
    protected_spans = [
        match.span()
        for pattern in (r"`+[^`\n]+`+", r"\[[^\]\n]+\]\([^\)\n]+\)")
        for match in re.finditer(pattern, line)
    ]

    def boundary(match):
        if any(start <= match.start() < end for start, end in protected_spans):
            return match.group(0)
        punct = match.group(1)
        core = re.match(r"[.!?…]+", punct).group(0)
        # An ellipsis is a pause within a thought, not an automatic newline:
        # "잠깐... 아니야!" must stay together.
        if set(core) <= {".", "…"} and (len(core) >= 2 or "…" in core):
            return match.group(0)
        before = line[:match.start()]
        after = line[match.end():]
        next_word = after.split(maxsplit=1)[0] if after.strip() else ""
        # List indexes, decimal-style version numbers and abbreviations are
        # not sentences. The latter can occur mid-paragraph.
        if punct == "." and before[-1:].isdigit():
            return match.group(0)
        last_token = re.search(r"([A-Za-z]+(?:\.[A-Za-z]+)*)$", before)
        if punct == "." and last_token and last_token.group(1).lower() in _ENGLISH_ABBREVIATIONS:
            return match.group(0)
        if next_word.startswith(_KOREAN_CONTINUATIONS):
            return match.group(0)
        # Avoid breaking English prose on a lowercase continuation.
        if after[:1].islower() and after[:1].isascii():
            return match.group(0)
        return punct + "\n"

    return _SENTENCE_BOUNDARY.sub(boundary, line)


def format_reply_for_display(text):
    """Format only the visible Gradio reply, not the model's conversation memory.

    Preserve the model's paragraphs and Markdown lists, keep expressive marks
    such as '?!' and '!!!!!' unchanged, and add optional *natural* sentence
    breaks. Use Markdown hard breaks for one-newline chat messages.
    """
    reply = (text or "").strip().replace("\r\n", "\n").replace("\r", "\n")
    # Do not rewrite fenced code blocks or headings; avoid corrupting Markdown.
    formatted_lines = []
    in_code = False
    for line in reply.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            in_code = not in_code
            formatted_lines.append(line)
        elif in_code or stripped.startswith(("# ", "## ", "### ")):
            formatted_lines.append(line)
        else:
            formatted_lines.append(_break_one_line_naturally(line.rstrip()))
    reply = "\n".join(formatted_lines)
    reply = re.sub(r"\n{3,}", "\n\n", reply)
    # Markdown normally collapses single line breaks; two trailing spaces
    # force a display break without adding an empty paragraph. Keep code fences
    # intact by avoiding added spaces inside fenced code blocks.
    lines = reply.split("\n")
    rendered = []
    in_code = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
            rendered.append(line)
        elif (not in_code and line and i + 1 < len(lines) and lines[i + 1]
              and not line.startswith(("|", "#"))):
            rendered.append(line.rstrip() + "  ")
        else:
            rendered.append(line)
    return "\n".join(rendered)


# ===== Chroma VectorDB / hybrid retrieval (shared by Base and Fine-Tuned) =====
RAG_DB_PATH = "/root/.cache/huggingface/anxiety_story_chroma"
RAG_DENSE_TOP_K = 12
RAG_SPARSE_TOP_K = 12
RAG_CONTEXT_TOP_K = 5
# These are retrieval heuristics, not confidence probabilities. Log actual scores
# and adjust only after checking answer-bearing passages with real BGE-M3.
RAG_DENSE_FLOOR = 0.38
RAG_LEXICAL_FLOOR = 0.12


def create_vector_store(documents, embedding_model):
    """Durable Chroma cosine index; new docs/model -> new content-hash collection.

    Vectors are explicitly supplied from the existing BGE-M3 encoder, so
    Chroma NEVER downloads its default embedding model or invokes a second LLM.
    """
    import chromadb

    digest_input = EMBEDDING_MODEL_ID + "\n" + "\n".join(
        f"{row['source']}:{row['index']}:{row['content']}" for row in documents
    )
    fingerprint = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:14]
    collection_name = f"anxiety_story_{fingerprint}"
    chroma_client = chromadb.PersistentClient(path=RAG_DB_PATH)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )
    ids = [f"{doc['source']}:{doc['index']}" for doc in documents]
    if collection.count() != len(documents):
        vectors = embedding_model.encode(
            [doc["content"] for doc in documents],
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Upsert instead of add to recover safely if an earlier start stopped
        # halfway through indexing.
        for offset in range(0, len(documents), 32):
            rows = documents[offset:offset + 32]
            collection.upsert(
                ids=ids[offset:offset + 32],
                documents=[doc["content"] for doc in rows],
                metadatas=[{"source": doc["source"], "index": doc["index"]} for doc in rows],
                embeddings=vectors[offset:offset + 32].tolist(),
            )
        # Persist the Chroma index in the existing Modal Volume.
        model_cache.commit()
        print(f"[rag-index] state=built backend=chroma count={collection.count()} "
              f"collection={collection_name}", flush=True)
    else:
        print(f"[rag-index] state=cached backend=chroma count={collection.count()} "
              f"collection={collection_name}", flush=True)
    if collection.count() != len(documents):
        raise RuntimeError("Chroma 인덱스에 저장된 문서 수가 일치하지 않습니다.")
    return chroma_client, collection


def create_lexical_index(documents):
    """Korean character n-gram TF-IDF covers spacing, typos and inflections."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 4), sublinear_tf=True,
        lowercase=True, dtype=np.float32,
    )
    matrix = vectorizer.fit_transform([d["content"] for d in documents])
    return vectorizer, matrix


def rank_rag_documents(runtime, search_queries):
    """One Chroma dense query per distinct query + sparse TF-IDF on same corpus.

    Reciprocal-rank fusion (RRF) avoids treating cosine and TF-IDF scores as
    identically calibrated. Original user wording and Qwen-rewritten query are
    both searched; individual passages retain their original source metadata.
    """
    docs = runtime["documents"]
    unique_queries = list(dict.fromkeys(q.strip() for q in search_queries if q and q.strip()))
    if not unique_queries:
        return []
    query_vectors = runtime["embedding_model"].encode(
        unique_queries, batch_size=min(len(unique_queries), 8),
        normalize_embeddings=True, show_progress_bar=False,
    )
    results = runtime["chroma_collection"].query(
        query_embeddings=query_vectors.tolist(),
        n_results=min(RAG_DENSE_TOP_K, len(docs)),
        include=["documents", "metadatas", "distances"],
    )
    by_id = {f"{d['source']}:{d['index']}": d for d in docs}
    aggregated = {}

    def add(doc_id, dense=None, sparse=None, rank=1, channel="dense"):
        if doc_id not in by_id:
            return
        slot = aggregated.setdefault(doc_id, {
            **by_id[doc_id], "dense_score": 0.0,
            "lexical_score": 0.0, "rrf": 0.0, "routes": set(),
        })
        if dense is not None:
            slot["dense_score"] = max(slot["dense_score"], dense)
        if sparse is not None:
            slot["lexical_score"] = max(slot["lexical_score"], sparse)
        slot["rrf"] += 1.0 / (60 + rank)
        slot["routes"].add(channel)

    for query_i, query in enumerate(unique_queries):
        dense_ids = results["ids"][query_i]
        distances = results["distances"][query_i]
        for rank, (doc_id, distance) in enumerate(zip(dense_ids, distances), 1):
            if distance is None:
                continue
            dense = max(-1.0, min(1.0, 1.0 - float(distance)))  # cosine distance -> cosine sim
            add(doc_id, dense=dense, rank=rank, channel="dense")

        q_vec = runtime["lexical_vectorizer"].transform([query])
        sparse_scores = (runtime["lexical_matrix"] @ q_vec.T).toarray().ravel()
        sparse_indices = sparse_scores.argsort()[::-1][:RAG_SPARSE_TOP_K]
        for rank, doc_index in enumerate(sparse_indices, 1):
            sparse = float(sparse_scores[int(doc_index)])
            if sparse <= 0:
                continue
            doc = docs[int(doc_index)]
            add(f"{doc['source']}:{doc['index']}", sparse=sparse,
                rank=rank, channel="lexical")

    candidates = sorted(
        aggregated.values(),
        key=lambda d: (d["rrf"], d["lexical_score"], d["dense_score"]),
        reverse=True,
    )
    for doc in candidates:
        doc["score"] = doc["dense_score"]
    return candidates


def retrieve_context(runtime, question, token_budget, search_queries=None):
    """RAG uses Chroma+lexical only when the Qwen Router chose RAG.

    No single fixed 0.50 cosine hard stop: either dense or text overlap can
    bring a relevant passage into context. Empty evidence yields no context.
    """
    queries = [question]
    queries.extend(search_queries or [])
    candidates = rank_rag_documents(runtime, queries)
    if not candidates:
        print("[rag-retrieval] backend=chroma outcome=empty", flush=True)
        return "", []

    preview = [
        f"{d['source']}:{d['index']} dense={d['dense_score']:.3f} "
        f"lex={d['lexical_score']:.3f} rrf={d['rrf']:.4f}"
        for d in candidates[:8]
    ]
    print(f"[rag-retrieval] backend=chroma candidates={len(candidates)} "
          f"top={preview}", flush=True)
    selected = []
    used_tokens = 0
    for doc in candidates:
        if (doc["dense_score"] < RAG_DENSE_FLOOR
                and doc["lexical_score"] < RAG_LEXICAL_FLOOR):
            continue
        content_tokens = token_count(runtime["tokenizer"], doc["content"])
        if content_tokens + used_tokens > token_budget:
            continue
        selected.append(doc)
        used_tokens += content_tokens
        if len(selected) >= RAG_CONTEXT_TOP_K:
            break

    if not selected:
        print("[rag-retrieval] backend=chroma outcome=insufficient_evidence "
              f"best_dense={candidates[0]['dense_score']:.3f} "
              f"best_lexical={candidates[0]['lexical_score']:.3f}", flush=True)
        return "", []
    source_ids = [f"{d['source']}:{d['index']}" for d in selected]
    print(f"[rag-retrieval] backend=chroma outcome=selected "
          f"selected={len(selected)} sources={source_ids}", flush=True)
    context_lines = ["<context>"]
    for i, doc in enumerate(selected, 1):
        context_lines.append(
            f'<doc{i} source="{doc["source"]}" index="{doc["index"]}">'
            f'{doc["content"]}'
            f'</doc{i}>'
        )
    context_lines.append("</context>")
    return "\n".join(context_lines), selected
# ===== End Chroma VectorDB / hybrid retrieval =====


def get_runtime():
    """Load BGE retrieval once, then keep the Part3 merged model on vLLM."""
    global _runtime
    if _runtime is not None:
        return _runtime

    import time
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    from vllm import LLM

    started_at = time.perf_counter()
    hf_token = os.environ.get("HF_TOKEN")
    data_dir = Path("/app/data")
    system_prompt = (data_dir / "system_prompt.txt").read_text(encoding="utf-8")
    documents = load_story_documents(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    tokenizer_seconds = time.perf_counter() - started_at
    print(f"[startup-stage] tokenizer_seconds={tokenizer_seconds:.2f}", flush=True)

    # Actual Chroma VectorDB + BGE-M3 embeddings and sparse keyword fallback.
    # Reuse existing on-volume Chroma collection after the first build.
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_ID,
        device="cpu",
        token=hf_token,
        model_kwargs={"torch_dtype": torch.float32},
    )
    chroma_client, chroma_collection = create_vector_store(documents, embedding_model)
    lexical_vectorizer, lexical_matrix = create_lexical_index(documents)
    rag_setup_seconds = time.perf_counter() - started_at - tokenizer_seconds
    print(f"[startup-stage] rag_setup_seconds={rag_setup_seconds:.2f} "
          f"backend=chroma doc_count={chroma_collection.count()}", flush=True)

    # vLLM model generation explicitly runs on the Modal L4/A10 GPU.
    # One concurrent chat is permitted by the Modal decorator; optimize
    # CUDA Graph capture for batch size 1 to limit its extra VRAM use.
    # CUDA Graph ON by default; VLLM_ENFORCE_EAGER=1 disables it if needed.
    enforce_eager = os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1"
    llm_started_at = time.perf_counter()
    llm = LLM(
        model=MODEL_ID,
        gpu_memory_utilization=0.88,
        max_model_len=8192,
        max_num_seqs=1,
        enforce_eager=enforce_eager,
        compilation_config={"cudagraph_capture_sizes": [1]} if not enforce_eager else {},
        trust_remote_code=False,
    )
    llm_seconds = time.perf_counter() - llm_started_at
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("Modal GPU가 감지되지 않습니다. GPU 할당과 CUDA 환경을 확인하세요.")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[gpu] device={gpu_name} cuda_available=True vllm_engine=GPU "
          f"enforce_eager={enforce_eager} cuda_graphs={not enforce_eager}", flush=True)

    # Check RAG retrieval and a real vLLM request before reporting readiness.
    # This runs once while the Modal worker is starting, not in the user's chat.
    from vllm import SamplingParams
    warmup_started_at = time.perf_counter()
    warmup_runtime = {
        "np": np,
        "tokenizer": tokenizer,
        "llm": llm,
        "embedding_model": embedding_model,
        "chroma_collection": chroma_collection,
        "chroma_client": chroma_client,
        "lexical_vectorizer": lexical_vectorizer,
        "lexical_matrix": lexical_matrix,
        "documents": documents,
        "system_prompt": system_prompt,
    }
    warmup_question = "라일리가 페널티 박스에서 나온 후 누구에게 사과했어?"
    warmup_context, _warmup_sources = retrieve_context(
        warmup_runtime, warmup_question, token_budget=950
    )
    _, warmup_prompt = build_messages(
        warmup_runtime, warmup_question, [], warmup_context
    )
    warmup_outputs = llm.generate(
        [warmup_prompt],
        SamplingParams(
            temperature=SAMPLING_TEMPERATURE,
            top_p=0.9,
            repetition_penalty=1.0,
            max_tokens=32,
            seed=42,
            stop_token_ids=[tokenizer.convert_tokens_to_ids("<|im_end|>")],
        ),
        use_tqdm=False,
    )
    if not warmup_outputs or not warmup_outputs[0].outputs:
        raise RuntimeError("Qwen vLLM warm-up did not return an output")
    warmup_tokens = len(warmup_outputs[0].outputs[0].token_ids)
    if warmup_tokens < 1:
        raise RuntimeError("Qwen vLLM warm-up returned zero generated tokens")
    if chroma_collection.count() != len(documents) or not len(documents):
        raise RuntimeError("Chroma VectorDB index was not initialized completely")
    # Fail startup if the previously failing test cannot find its real answer.
    if not any("브리" in d["content"] and "그레이스" in d["content"]
               and "사과" in d["content"] for d in _warmup_sources):
        raise RuntimeError(
            "RAG 검색 검증 실패: 페널티 박스 사과 근거(브리·그레이스)를 찾지 못했습니다."
        )
    warmup_seconds = time.perf_counter() - warmup_started_at
    print(
        f"[warmup] completed=True rag_docs={len(_warmup_sources)} "
        f"generated_tokens={warmup_tokens} seconds={warmup_seconds:.2f}",
        flush=True,
    )

    _runtime = {
        "np": np,
        "tokenizer": tokenizer,
        "llm": llm,
        "embedding_model": embedding_model,
        "chroma_collection": chroma_collection,
        "chroma_client": chroma_client,
        "lexical_vectorizer": lexical_vectorizer,
        "lexical_matrix": lexical_matrix,
        "documents": documents,
        "system_prompt": system_prompt,
        "gpu_name": gpu_name,
        "enforce_eager": enforce_eager,
        "warmup_complete": True,
        "rag_ready": True,
        "warmup_tokens": warmup_tokens,
        "warmup_seconds": warmup_seconds,
        "startup_seconds": time.perf_counter() - started_at,
    }
    print(
        f"[startup] build={SERVING_BUILD} engine=vllm model={MODEL_ID} "
        f"temperature={SAMPLING_TEMPERATURE} max_tokens={MAX_GENERATION_TOKENS} "
        f"prompt_sha256={hashlib.sha256(system_prompt.encode('utf-8')).hexdigest()[:12]} "
        f"vllm_init_seconds={llm_seconds:.1f} "
        f"startup_seconds={time.perf_counter() - started_at:.1f}",
        flush=True,
    )
    return _runtime


def token_count(tokenizer, text):
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


# Three real ChatML demonstration conversations. The second is deliberately
# multi-turn so follow-ups learn to REACT to the user's last utterance, not
# to restart the previous topic. None is saved in the actual user history.
FEWSHOT_DIALOGUES = (
    (
        {"role": "user", "content": "넌 왜 그렇게 걱정이 많아?"},
        {"role": "assistant", "content": (
            "헉!! 라일리가 곤란해질 만한 장면이 머릿속에 먼저 떠오르거든!!! 친구를 못 사귀면? 경기에서 실수하면?! 으아아, 막 걱정되잖아!! ...어어, 잠깐. 아직 안 일어난 일이니까 대비는 조금만 하고 지금 네 이야기도 들어야지!!!"
        )},
    ),
    (
        {"role": "user", "content": "내일 친구를 만나는데 비가 올 거래."},
        {"role": "assistant", "content": (
            "으앗, 내일 비가 온대?! 야외에서 만나기로 했으면 신발부터 젖는 거 아냐!! 좋아, 실내 후보를 하나 찾아두자!!! ...어어, 근데 친구가 이미 식당을 정했을 수도 있잖아?! 그러니까 장소부터 물어보고, 야외라면 근처 카페로 바꾸자고 해보자!!"
        )},
        {"role": "user", "content": "응, 나도 그렇게 생각해!"},
        {"role": "assistant", "content": (
            "그치그치!! 장소가 정해지면 마음이 좀 놓이겠다!!! 아, 잠깐! 비가 약하게 올 수도 있으니까 무조건 취소할 건 아니네. 일단 친구 답장 기다렸다가 실내가 필요한지만 결정하자!!"
        )},
    ),
    (
        {"role": "user", "content": "너 라일리가 선배들이 웃는 이유도 모르면서 따라 웃게 했잖아!"},
        {"role": "assistant", "content": (
            "아앗, 맞아!!! 내가 선배들 대화를 제대로 듣지도 못했는데 라일리가 따라 웃도록 영향을 줬어!! 혹시 선배들이 라일리를 이상하게 보면 어떡하지?! 그 생각에 너무 급했던 거야!!! ...으으, 근데 그러다 라일리가 자기답게 행동할 여유까지 줄여버렸잖아. 그건 내가 잘못 판단했어!!"
        )},
        {"role": "user", "content": "그러니까 네가 그렇게 하게 한 거지?"},
        {"role": "assistant", "content": (
            "응, 내가 그렇게 행동하도록 영향을 준 거야!! 그냥 옆에서 본 게 아니라고!!! 잘 보이게 하려다 라일리 마음은 뒤로 미뤄버렸어... 으아, 그건 진짜 조심했어야 했는데!!"
        )},
    ),
)
FEWSHOT_MESSAGES = tuple(message for dialogue in FEWSHOT_DIALOGUES for message in dialogue)

def completed_history(history):
    """Return only actual, complete user/assistant pairs, in order."""
    clean = clean_history(history)
    pairs = []
    pending_user = None
    for msg in clean:
        if msg["role"] == "user":
            pending_user = msg["content"]
        elif pending_user is not None:
            pairs.extend((
                {"role": "user", "content": pending_user},
                {"role": "assistant", "content": msg["content"]},
            ))
            pending_user = None
    return pairs[-2 * MAX_HISTORY_EXCHANGES:]


# A lightweight intent-routing request is made to the ALREADY LOADED Qwen
# vLLM checkpoint before touching the CPU embedding index. No second LLM is
# downloaded. Returning CHAT means zero embedding/VectorDB calls this turn.
RAG_ROUTER_SYSTEM = """너는 인사이드 아웃 2 불안이 챗봇의 검색 의도 분류기야. 사용자의 질문에 답하지 마. 다음 작업은 반드시 하나만 수행해: RAG 또는 CHAT.

[분류 기준]
- RAG: 영화 속 실제 사건, 등장인물이 누구에게 무엇을 했는지, 장면의 순서·이유, 불안이가 그때 했던 행동·기억을 확인해야 올바르게 답할 수 있을 때. 질문에 '너', '나'가 있어도 영화 속 사실을 묻는다면 RAG.
- RAG: 생략된 후속 질문이 직전 대화의 특정 영화 장면을 가리키고, 그 장면에 관한 새로운 사실 확인이 필요할 때.
- RAG: **질문이 아니더라도** 사용자가 영화 속 구체적인 행동·장면을 언급하며 "네가 그랬잖아", "그때 라일리가 ~했어"처럼 사실을 확인하거나 불안이의 행동을 지적·평가하면, 근거를 확인하고 **그 사건 자체에 반응**해야 하므로 RAG. 
- RAG: 영화 속 사건을 자신의 고민에 연결해서 묻는 복합 발화라면 사건 확인 근거가 유용하므로 RAG. 단, 최종 답변에서는 사용자의 현실 고민도 함께 다뤄야 해.
- CHAT: 이미 확인한 사건에 대한 단순 동의·짧은 감정 반응만 있고, 새롭게 확인할 영화 사실이 없다면 CHAT.
- **물음표 유무나 "했잖아"라는 어미 자체만으로 결정하지 말고**, 실제 사건·행동에 대한 근거가 필요한지 문장 의미와 직전 대화를 보고 판단해.
- CHAT: 사용자 자신의 현실 고민, 일상 대화, 맞장구·동의, 축하·위로, 불안이의 일반적인 성격이나 의견을 묻는 질문.
- CHAT: 영화 장면에 관한 의견·공감만 표현하거나, 이전 응답에 단순 동의만 하는 후속 발화. 직전에 RAG를 썼더라도 자동으로 RAG를 이어가지 마.
- '?', '!', '??!!' 같은 문장부호는 판단 기준이 아니야. 현재 발화의 *의미와 필요한 근거*를 판단해.
- 영화 속 사실을 묻는 질문은 일반 대화로 적당히 넘기지 마. 지식 근거가 필요한지에 따라 판단해. 맥락이 모호한 후속 질문은 직전 실제 대화를 해석에 사용해.

반드시 다른 설명 없이 JSON 한 객체만 출력해:
RAG: {"route":"RAG","query":"영화 속 사건을 찾을 수 있는 짧고 구체적인 한국어 검색어"}
CHAT: {"route":"CHAT","query":""}
RAG이면 query에는 현재 질문과 직전 대화에 나온 사건·장소·인물을 구체적으로 결합해. 단, 정답을 미리 추측해서 query에 넣지는 마.

[판단 예시]
- 현재: "너 라일리가 페널티 박스에서 나온 후 누구에게 사과하는 걸 봤어?" -> {"route":"RAG","query":"라일리 페널티 박스 나온 후 친구들에게 사과한 장면 누구에게"}
- 현재: "너 라일리가 하키 캠프에서 선배 선수들에게 잘 보이려고 자기 음악 취향까지 부정하게 했잖아!" -> {"route":"RAG","query":"라일리 하키 캠프 선배 선수 음악 취향 자신의 취향 부정 불안이 관여"}
- 현재: "너 그날 밤에 라일리를 코치의 사무실로 가게 했잖아." -> {"route":"RAG","query":"라일리 밤 코치 사무실 평가 노트 확인 불안이 행동"}
- 현재: "라일리가 선배들 앞에서 자신의 음악 취향을 숨긴다니 마음 아프다." -> {"route":"CHAT","query":""}
- 현재: "라일리가 코치 사무실에 들어가기 전에 왜 망설였어?" -> {"route":"RAG","query":"라일리 코치 사무실 들어가기 전 망설인 상황 이유"}
- 이전: 라일리의 페널티 박스 장면에 대해 대화 중 / 현재: "그 뒤엔 누구에게 사과했어?" -> {"route":"RAG","query":"라일리 페널티 박스 나온 뒤 사과한 사람"}
- 이전: 라일리의 페널티 박스 장면에 대해 대화 중 / 현재: "나도 그렇게 생각해!!!" -> {"route":"CHAT","query":""}
- 현재: "라일리도 참 힘들었겠다" -> {"route":"CHAT","query":""}
- 현재: "넌 왜 아직 일어나지 않은 일까지 걱정해?" -> {"route":"CHAT","query":""}
- 이전: 내일 면접이 있다고 말함 / 현재: "그럼 무엇부터 준비할까?" -> {"route":"CHAT","query":""}
"""


def _router_context(history):
    # Only genuine conversation, never Few-shot demonstrations or retrieved docs.
    # Keep the router short, so adding a second vLLM call does not add a long prefill.
    return [
        {"role": item["role"], "content": item["content"][:220]}
        for item in completed_history(history)[-6:]
    ]


def parse_rag_router_output(raw, current_question):
    """Route solely from Qwen's JSON decision (not keyword/punctuation rules).

    A malformed JSON is a technical failure, not a CHAT classification. Prefer
    retrieving to silently suppressing factual support; report this in logs.
    """
    raw = str(raw or "").strip()
    match = re.search(r"\{[^{}]*\}", raw, flags=re.S)
    if match:
        try:
            decision = json.loads(match.group())
            if isinstance(decision, dict):
                route = str(decision.get("route", "")).strip().upper()
                if route == "CHAT":
                    return "", "llm_chat"
                if route == "RAG":
                    query = str(decision.get("query", "")).strip()
                    query = re.sub(r"<[/]?context>|<[/]?doc\w*>", "", query, flags=re.I)
                    query = " ".join(query.split())[:300]
                    if len(query) < 4:
                        query = str(current_question or "").strip()
                    return query, "llm_rag"
        except (ValueError, TypeError):
            pass
    return str(current_question or "").strip(), "router_invalid_json_fallback_rag"


def choose_rag_query(runtime, question, history):
    """Every nonempty input is classified by the already-loaded Qwen vLLM.

    No keyword-based forced routes and no punctuation-only decisions.
    """
    from vllm import SamplingParams

    question = str(question or "").strip()
    if not question:
        return "", "empty"

    router_payload = {
        "history": _router_context(history),
        "current_user_message": question[:1000],
    }
    router_messages = [
        {"role": "system", "content": RAG_ROUTER_SYSTEM},
        {"role": "user", "content": json.dumps(router_payload, ensure_ascii=False)},
    ]
    router_prompt = runtime["tokenizer"].apply_chat_template(
        router_messages, tokenize=False, add_generation_prompt=True,
    )
    try:
        router_result = runtime["llm"].generate(
            [router_prompt],
            SamplingParams(
                temperature=0.0, top_p=1.0, repetition_penalty=1.0,
                max_tokens=160, seed=42,
                stop_token_ids=[runtime["tokenizer"].convert_tokens_to_ids("<|im_end|>")],
            ),
            use_tqdm=False,
        )
        raw = router_result[0].outputs[0].text.strip()
        query, route = parse_rag_router_output(raw, question)
        print(
            f"[rag-router] route={route} retrieve={bool(query)} "
            f"history={len(router_payload['history'])} query_chars={len(query)}",
            flush=True,
        )
        return query, route
    except Exception as exc:
        # A technical router failure is never silently interpreted as CHAT.
        print(f"[rag-router-error] {type(exc).__name__}: {exc}; fallback=RAG", flush=True)
        return question, "router_error_fallback_rag"


def build_messages(runtime, question, history, context, retrieval_attempted=False):
    """Evaluation: identical Qwen apply_chat_template / generation header.

    1 system -> 3 demonstration conversations -> genuine completed history
    -> latest USER question (last and clearly labelled if RAG) -> assistant.
    """
    tokenizer = runtime["tokenizer"]
    trimmed_history = completed_history(history)
    current_question = question.strip()
    if not current_question:
        raise ValueError("Current user message must not be empty")

    # RAG comes BEFORE the newest user's text, so the most recent instruction
    # at the end is the message the person really typed, not retrieved prose.
    current_user_content = (
        f"영화 속 참고 자료 (관련될 때만 참고):\n{context}\n\n"
        "응답 원칙: 마지막 사용자 메시지가 특정 영화 사건을 지적하는 평서문이면 "
        "질문으로 되묻거나 일반적인 인생 조언을 하지 말고, 근거를 확인해 "
        "불안이가 직접 한 행동을 1인칭으로 인정하거나 정정하세요. "
        "특히 문서에 불안이가 '~에 관여했다', '~하도록 영향을 주었다'고 있으면, "
        "당사자인 내가 그 행동에 관여하고 영향을 줬다고 인정하세요. "
        "'내가 봤어'처럼 관찰만 한 사람으로 바꾸지 마세요. "
        "반대로 목격만 한 사건을 내가 지시했다고 과장하지 마세요. "
        "음악 취향 사건을 페널티 박스 이후 화해 사건과 혼동하지 마세요. "
        "사용자가 자신의 고민을 명시한 경우에만 상담으로 연결하세요. "
        "문서에 없는 행동은 만들지 마세요. "
        "RAG로 검색한 영화 사건도 당시의 놀람·걱정·안도감을 자연스럽게 살려 불안이답게 말하되, "
        "질문을 되묻지 말고 확인한 사실부터 정확히 말하세요.\n\n"
        f"현재 사용자 메시지:\n{current_question}"
        if context else (
            "영화 자료를 검색했지만 이 질문에 대한 확실한 근거를 찾지 못했습니다. "
            "영화 속 세부 사실을 지어내지 말고 근거를 확인할 수 없다고 솔직히 답하세요.\n\n"
            f"현재 사용자 메시지:\n{current_question}"
            if retrieval_attempted else current_question
        )
    )
    while True:
        messages = [
            {"role": "system", "content": runtime["system_prompt"]},
            *FEWSHOT_MESSAGES,
            *trimmed_history,
            {"role": "user", "content": current_user_content},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        if not (
            prompt.startswith("<|im_start|>system\n")
            and prompt.rstrip().endswith("<|im_start|>assistant")
        ):
            raise RuntimeError("Qwen ChatML framing unexpected")
        if token_count(tokenizer, prompt) <= MAX_PROMPT_TOKENS or not trimmed_history:
            return messages, prompt
        trimmed_history = trimmed_history[2:]


def generate_answer(question, history):
    """Direct vLLM generation: no Transformers model.generate / postprocessors."""
    import time
    from vllm import SamplingParams

    started_at = time.perf_counter()
    runtime = get_runtime()
    # Decide whether retrieving movie facts helps THIS message; keep
    # acknowledgments and general chat entirely RAG-free.
    ready_at = time.perf_counter()
    rag_query, rag_intent = choose_rag_query(runtime, question, history)
    # Query both original wording and the Qwen-rewritten search phrase in
    # Chroma+TF-IDF and fuse rankings. CHAT never touches VectorDB.
    context, sources = (retrieve_context(
        runtime, question, token_budget=950, search_queries=[rag_query],
    ) if rag_query else ("", []))
    retrieval_seconds = time.perf_counter() - ready_at  # includes LLM intent routing
    _, prompt = build_messages(
        runtime, question, history, context, retrieval_attempted=bool(rag_query),
    )
    prompt_tokens = token_count(runtime["tokenizer"], prompt)

    sampling_params = SamplingParams(
        temperature=SAMPLING_TEMPERATURE,
        top_p=0.9,
        repetition_penalty=1.0,
        max_tokens=MAX_GENERATION_TOKENS,
        seed=42,
        stop_token_ids=[runtime["tokenizer"].convert_tokens_to_ids("<|im_end|>")],
    )
    decoding_started_at = time.perf_counter()
    outputs = runtime["llm"].generate([prompt], sampling_params, use_tqdm=False)
    decode_seconds = time.perf_counter() - decoding_started_at
    reply = outputs[0].outputs[0].text.strip()
    if not reply:
        raise RuntimeError("vLLM 모델이 빈 답변을 반환했습니다.")

    # Multilingual output can occasionally contain stray Hanzi.
    # A single stray character is removed instantly; longer Chinese spans
    # get one low-temperature retry before the final display-time guard.
    if has_multiple_hanzi(reply):
        print("[language-guard] unexpected_hanzi; retrying_once=True", flush=True)
        retry_messages, _ = build_messages(
            runtime, question, history, context, retrieval_attempted=bool(rag_query),
        )
        retry_messages[0] = {
            **retry_messages[0],
            "content": retry_messages[0]["content"] + (
                "\n\n[출력 점검] 이번 답변에는 한국어만 사용하세요. "
                "중국어 문장이나 한자를 절대 출력하지 마세요."
            ),
        }
        retry_prompt = runtime["tokenizer"].apply_chat_template(
            retry_messages, tokenize=False, add_generation_prompt=True
        )
        retry_sampling = SamplingParams(
            temperature=SAMPLING_TEMPERATURE,
            top_p=0.9,
            repetition_penalty=1.0,
            max_tokens=MAX_GENERATION_TOKENS,
            seed=42,
            stop_token_ids=[runtime["tokenizer"].convert_tokens_to_ids("<|im_end|>")],
        )
        retry_outputs = runtime["llm"].generate(
            [retry_prompt], retry_sampling, use_tqdm=False
        )
        retried_reply = retry_outputs[0].outputs[0].text.strip()
        if retried_reply:
            reply = retried_reply

    reply = enforce_korean_output(reply)
    if not reply:
        raise RuntimeError("한국어 출력 검증 후 답변이 비었습니다.")

    # Retain only genuine user/assistant messages across Gradio turns.
    updated_history = completed_history(history)[-(MAX_HISTORY_EXCHANGES * 2 - 2):] + [
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": reply},
    ]
    source_rows = [
        {"source": doc["source"], "index": doc["index"],
         "score": round(doc["score"], 4),
         "lexical_score": round(doc["lexical_score"], 4)}
        for doc in sources
    ]
    generated_tokens = len(outputs[0].outputs[0].token_ids)
    print(
        f"[chat] engine=vllm prompt_tokens={prompt_tokens} "
        f"generated_tokens={generated_tokens} rag_intent={rag_intent} rag_docs={len(sources)} "
        f"rag_seconds={retrieval_seconds:.2f} decode_seconds={decode_seconds:.2f} "
        f"seconds={time.perf_counter() - started_at:.2f}",
        flush=True,
    )
    return reply, updated_history, source_rows


@app.function(
    image=web_image,
    timeout=30,
    min_containers=0,
    max_containers=1,
    scaledown_window=2,
    secrets=[modal_secret],
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def billing_api():
    from decimal import Decimal

    from fastapi import FastAPI, HTTPException, Request

    api = FastAPI(title="Anxiety Chatbot Billing API")

    @api.get("/billing")
    async def billing(request: Request):
        expected_token = os.environ.get("APP_API_TOKEN", "")
        auth_header = request.headers.get("Authorization", "")
        received_token = auth_header.removeprefix("Bearer ").strip()

        if not is_valid_api_token(received_token, expected_token):
            raise HTTPException(status_code=401, detail="Unauthorized")

        summary = await modal.Workspace.from_context().billing.summary.aio(
            "this month"
        )
        credit_total = Decimal("30")
        credit_adjustments = [
            -value
            for key, value in summary.adjustments.items()
            if "credit" in key.lower() and value < 0
        ]
        credit_used = sum(credit_adjustments, Decimal("0"))

        if credit_used == 0 and summary.metered_cost > 0:
            free_adjustments = sum(
                -value
                for key, value in summary.adjustments.items()
                if value < 0 and "credit" not in key.lower()
            )
            credit_used = max(
                Decimal("0"),
                summary.metered_cost - free_adjustments,
            )

        credit_used = min(credit_total, credit_used)
        credit_remaining = max(
            Decimal("0"),
            credit_total - credit_used,
        )
        return {
            "credit_total": str(credit_total),
            "credit_used": str(credit_used),
            "credit_remaining": str(credit_remaining),
            "billed_cost": str(summary.billed_cost),
        }

    return api


@app.function(
    image=modal_image,
    gpu=["L4", "A10"],
    cpu=4.0,
    memory=32768,
    timeout=180,
    startup_timeout=600,
    # Keep the interactive GPU worker ready. With scale-to-zero, the first
    # chat after an idle period waits for both Qwen 7B and BGE-M3 to reload.
    min_containers=1,
    max_containers=1,
    scaledown_window=180,
    volumes={"/root/.cache/huggingface": model_cache},
    secrets=[modal_secret],
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def modal_api():
    from fastapi import FastAPI, HTTPException, Request

    get_runtime()
    api = FastAPI(title="Anxiety Chatbot Fine-Tuned Model API")

    @api.get("/health")
    async def health():
        runtime = get_runtime()
        return {
            "status": "ok",
            "ready": bool(runtime.get("warmup_complete") and runtime.get("rag_ready") and runtime.get("warmup_tokens", 0) > 0),
            "warmup_complete": runtime["warmup_complete"],
            "rag_ready": runtime["rag_ready"],
            "warmup_tokens": runtime["warmup_tokens"],
            "warmup_seconds": round(runtime["warmup_seconds"], 2),
            "build": SERVING_BUILD,
            "model_id": MODEL_ID,
            "inference_type": "vllm_merged_finetuned_checkpoint",
            "uses_merged_checkpoint": True,
            "prompt_sha256": hashlib.sha256(
                runtime["system_prompt"].encode("utf-8")
            ).hexdigest()[:12],
            "engine": "vllm",
            "gpu_active": True,
            "gpu_name": runtime["gpu_name"],
            "cuda_graphs_enabled": not runtime["enforce_eager"],
            "startup_seconds": round(runtime["startup_seconds"], 2),
            "max_tokens": MAX_GENERATION_TOKENS,
            "rag_context_role": "qwen_route_then_chroma_bge3_plus_lexical_rrf_then_evidence_context",
            "rag_vector_store": "chroma_persistent",
            "rag_retrieval": "bge_m3_cosine_plus_char_tfidf_rrf",
            "rag_document_count": runtime["chroma_collection"].count(),
            "rag_router": "qwen_vllm_binary_route_no_keyword_override",
            "history_mode": "complete_user_assistant_pairs",
            "checkpoint_pinned": True,
            "fewshot_dialogs": 3,
            "history_max_exchanges": MAX_HISTORY_EXCHANGES,
            "temperature": SAMPLING_TEMPERATURE,
            "seed": 42,
            "do_sample": True,
            "repetition_penalty": 1.0,
            "top_p": 0.9,
        }

    @api.post("/chat")
    async def chat(request: Request):
        expected_token = os.environ.get("APP_API_TOKEN", "")
        auth_header = request.headers.get("Authorization", "")
        received_token = auth_header.removeprefix("Bearer ").strip()

        if not is_valid_api_token(received_token, expected_token):
            raise HTTPException(status_code=401, detail="Unauthorized")

        payload = await request.json()
        question = str(payload.get("message", "")).strip()
        history = payload.get("history", [])

        if not question:
            raise HTTPException(status_code=400, detail="Message is required")

        reply, updated_history, sources = generate_answer(
            question,
            history,
        )
        return {
            "reply": reply,
            "history": updated_history,
            "sources": sources,
            "build": SERVING_BUILD,
            "model_id": MODEL_ID,
            "engine": "vllm",
            "history_messages": len(updated_history),
            "rag_vector_store": "chroma_persistent",
        }

    return api


def call_modal(message, history):
    import requests

    base_url = os.environ.get("MODAL_API_URL", "").strip().rstrip("/")
    api_token = os.environ.get("MODAL_API_TOKEN", "").strip()

    if not base_url:
        raise RuntimeError("MODAL_API_URL이 설정되지 않았습니다.")

    if not api_token:
        raise RuntimeError("MODAL_API_TOKEN이 설정되지 않았습니다.")

    response = requests.post(
        f"{base_url}/chat",
        headers={"Authorization": f"Bearer {api_token}"},
        json={
            "message": message,
            "history": clean_history(history),
        },
        timeout=900,
        allow_redirects=True,
    )

    if response.status_code == 401:
        raise RuntimeError("Modal API 토큰이 일치하지 않습니다.")

    response.raise_for_status()
    result = response.json()
    if not (
        result.get("build") == SERVING_BUILD
        and result.get("model_id") == MODEL_ID
        and result.get("engine") == "vllm"
    ):
        raise RuntimeError(
            "연결된 Modal 서버가 이번 vLLM 배포판/모델과 다릅니다. "
            f"build={result.get('build')}, model={result.get('model_id')}, "
            f"engine={result.get('engine')}"
        )
    if not isinstance(result.get("reply"), str) or not isinstance(result.get("history"), list):
        raise RuntimeError("Invalid vLLM server response")
    return result["reply"], result["history"]


CREDIT_COIN_ICON = """
<span class="credit-coin" aria-hidden="true">
  <svg viewBox="0 0 40 40" focusable="false">
    <defs>
      <linearGradient id="coin-gold" x1="8" y1="5" x2="32" y2="35">
        <stop offset="0" stop-color="#fff4a7"/>
        <stop offset="0.42" stop-color="#ffc53d"/>
        <stop offset="1" stop-color="#e47a00"/>
      </linearGradient>
    </defs>
    <circle cx="20" cy="20" r="17" fill="#c66a00"/>
    <circle cx="20" cy="18.5" r="15.5" fill="url(#coin-gold)"/>
    <circle cx="20" cy="18.5" r="11.5" fill="none" stroke="#fff0a0" stroke-width="1.7"/>
    <text x="20" y="25" font-family="Arial, sans-serif" font-size="21" font-weight="900" text-anchor="middle" fill="#974900">$</text>
    <ellipse cx="14" cy="10.5" rx="4.1" ry="2.4" fill="#fff" opacity=".65" transform="rotate(-28 14 10.5)"/>
  </svg>
</span>
""".strip()


def credit_badge(content, muted=False, tooltip=None):
    # The coin is an embedded SVG; it will not break if a static file fails.
    from html import escape

    muted_class = " muted" if muted else ""
    title = escape(tooltip or (
        "Modal 사용 내역 기준 추정치입니다. 실제 남은 크레딧은 Modal Billing에서 확인하세요."
    ), quote=True)
    return (
        f'<div class="credit-pill{muted_class}" title="{title}">'
        f'{CREDIT_COIN_ICON}<span class="credit-values">{content}</span>'
        "</div>"
    )


def infer_billing_url(model_url):
    """Find the sibling billing endpoint from a deployed Modal API endpoint.

    Example:
    https://user--anxiety-chatbot-modal-api.us-east.modal.run
      -> https://user--anxiety-chatbot-billing-api.us-east.modal.run
    Explicit MODAL_BILLING_URL still takes priority for custom endpoints.
    """
    from urllib.parse import urlsplit, urlunsplit

    value = (model_url or "").strip().strip('"\'').rstrip("/")
    parts = urlsplit(value)
    if (parts.scheme != "https" or not parts.hostname
            or not parts.hostname.endswith(".modal.run")
            or "-modal-api." not in parts.hostname):
        return ""
    host = parts.hostname.replace("-modal-api.", "-billing-api.", 1)
    return urlunsplit(("https", host, "", "", ""))


def fetch_credit_status():
    import requests

    billing_url = os.environ.get("MODAL_BILLING_URL", "").strip().rstrip("/")
    if not billing_url:
        billing_url = infer_billing_url(os.environ.get("MODAL_API_URL", ""))
    api_token = os.environ.get("MODAL_API_TOKEN", "").strip()

    if not api_token:
        return credit_badge(
            "<strong>연결 필요</strong>", muted=True,
            tooltip="로컬 CMD에 MODAL_API_TOKEN을 등록하세요.",
        )
    if not billing_url:
        return credit_badge(
            "<strong>조회 설정 필요</strong>", muted=True,
            tooltip="MODAL_BILLING_URL을 설정하거나 실제 modal.run의 MODAL_API_URL을 설정하세요.",
        )

    try:
        response = requests.get(
            f"{billing_url}/billing",
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=20,
            allow_redirects=False,
        )
        response.raise_for_status()
        result = response.json()
        remaining = float(result["credit_remaining"])
        total = float(result["credit_total"])
        if not (0 <= remaining <= total):
            raise ValueError("Invalid estimated credit balance")
        return credit_badge(
            f'<strong>≈ ${remaining:.2f}</strong><small>/ ${total:.2f}</small>',
            tooltip=("이번 달 사용 내역으로 계산한 크레딧 추정치입니다. "
                     "실제 크레딧 지급·만료·청구 잔액과 다를 수 있습니다. "
                     "Modal Billing 페이지에서 실제 잔액을 확인하세요."),
        )
    except Exception as error:
        print(
            f"Modal billing error ({billing_url}): "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
        return credit_badge(
            "<strong>조회 불가</strong>", muted=True,
            tooltip="Modal 비용 조회에 실패했습니다. CMD 로그 및 Modal Billing 화면을 확인하세요.",
        )


# The startup overlay is dismissed only after Modal confirms GPU readiness.
def model_readiness_html(ready=False, error=False):
    if ready:
        return '<span class="readiness-complete" role="status">불안이 준비 완료</span>'

    avatar = image_data_uri(ASSET_DIR / "anxiety-chat-avatar.png")
    if error:
        heading = "모델 연결을 확인하고 있어"
        detail = "Modal 연결 정보와 배포 상태를 확인해 줘. 준비되면 자동으로 대화가 열릴 거야."
    else:
        heading = "불안이가 준비 중이야!"
        detail = "Qwen 모델과 RAG를 불러오는 중이야. 준비가 끝나면 바로 대화할 수 있어!"
    return (
        '<div class="readiness-screen" role="status" aria-live="polite" aria-busy="true">'
        '  <div class="readiness-dialog">'
        f'    <img class="readiness-avatar" src="{avatar}" alt="불안이 캐릭터">'
        '    <div class="readiness-spinner" aria-hidden="true"></div>'
        f'    <h2>{heading}</h2>'
        f'    <p>{detail}</p>'
        '    <span class="readiness-footnote">모델 준비 상태를 자동으로 확인하고 있어</span>'
        '  </div>'
        '</div>'
    )


def check_model_readiness():
    import gradio as gr
    import requests

    base_url = os.environ.get("MODAL_API_URL", "").strip().rstrip("/")
    api_token = os.environ.get("MODAL_API_TOKEN", "").strip()
    if not base_url or not api_token:
        return (
            model_readiness_html(error=True),
            gr.update(interactive=False, placeholder="모델 연결 설정을 확인해 줘..."),
            gr.update(interactive=False),
            False,
            gr.update(active=True),
        )
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        response.raise_for_status()
        payload = response.json()
        ready = (
            response.ok
            and payload.get("status") == "ok"
            and payload.get("ready") is True
            and payload.get("warmup_complete") is True
            and payload.get("rag_ready") is True
            and isinstance(payload.get("warmup_tokens"), int)
            and payload["warmup_tokens"] > 0
            and payload.get("build") == SERVING_BUILD
            and payload.get("engine") == "vllm"
            and payload.get("model_id") == MODEL_ID
            and payload.get("gpu_active") is True
            and payload.get("checkpoint_pinned") is True
            and payload.get("temperature") == SAMPLING_TEMPERATURE
        )
        if response.ok and not ready:
            print(
                f"[ready-check] unexpected backend model/engine: "
                f"build={payload.get('build')}, ready={payload.get('ready')}, "
                f"warmup={payload.get('warmup_complete')}, model={payload.get('model_id')}",
                flush=True,
            )
    except (requests.RequestException, ValueError) as error:
        # Model boot can be slow; keep retrying in the background and do not
        # send the user's first message to an uninitialized model.
        ready = False
    if ready:
        return (
            model_readiness_html(ready=True),
            gr.update(interactive=True, placeholder="걱정이나 고민을 입력해 줘..."),
            gr.update(interactive=True),
            True,
            gr.update(active=False),
        )
    return (
        model_readiness_html(),
        gr.update(interactive=False, placeholder="불안이가 모델을 준비하는 중이야..."),
        gr.update(interactive=False),
        False,
        gr.update(active=True),
    )


# #### UI Events: one queued request == one committed user/assistant pair.
def handle_send(message, display_history, model_history, ready):
    import gradio as gr

    rows = list(display_history or [])
    user_text = (message or "").strip()
    if not ready or not user_text:
        yield (
            gr.update(value=message or ""), rows, model_history,
            gr.update(interactive=bool(ready)), gr.update(interactive=True),
        )
        return

    # Disable new sends and resets until the backend returns the updated
    # history for THIS turn. Gradio event trigger_mode='once' also prevents
    # stacking duplicate requests while inference is running.
    rows.append({"role": "user", "content": user_text})
    yield (
        gr.update(value="", interactive=False), rows, model_history,
        gr.update(interactive=False), gr.update(interactive=False),
    )
    try:
        reply, new_history = call_modal(user_text, model_history)
        reply = reply.strip()
        if not reply:
            raise RuntimeError("Model returned an empty reply")
        if not isinstance(new_history, list) or len(new_history) < 2:
            raise RuntimeError("Backend returned no complete conversation history")
    except Exception as error:
        print(f"[chat-ui-error] {type(error).__name__}: {error}", flush=True)
        reply = (
            "으아, 잠깐! 모델 서버에서 답변을 받아오지 못했어. "
            "이 메시지는 기록하지 않을게. 연결을 확인하고 다시 보내줘!"
        )
        # No false model turn is saved on failure.
        new_history = model_history

    final_rows = rows + [{"role": "assistant", "content": reply}]
    yield (
        gr.update(value="", interactive=True), final_rows, new_history,
        gr.update(interactive=True), gr.update(interactive=True),
    )


def reset_chat():
    return ([{"role": "assistant", "content": INITIAL_GREETING}], [], "")


def image_data_uri(path):
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_decoration_html():
    return ""


ENTER_TO_SEND_JS = """
() => {
  const bindComposer = () => {
    const textarea = document.querySelector('#message-box textarea');
    const sendButton = document.querySelector('#send-button');

    if (!textarea || !sendButton || textarea.dataset.enterBound) return;

    textarea.dataset.enterBound = 'true';
    textarea.setAttribute('lang', 'ko');
    textarea.setAttribute('enterkeyhint', 'send');
    let composing = false;

    textarea.addEventListener('compositionstart', () => composing = true);
    textarea.addEventListener('compositionend', () => {
      setTimeout(() => composing = false, 80);
    });
    textarea.addEventListener('keydown', (event) => {
      const isIme = composing || event.isComposing || event.keyCode === 229;

      if (event.key === 'Enter' && !event.shiftKey && !isIme) {
        event.preventDefault();
        event.stopImmediatePropagation();
        if (!sendButton.disabled) sendButton.click();
      }
    }, true);
  };

  bindComposer();

  if (!window.__anxietyComposerObserver) {
    window.__anxietyComposerObserver = new MutationObserver(bindComposer);
    window.__anxietyComposerObserver.observe(
      document.body,
      {childList: true, subtree: true}
    );
  }
}
"""


def build_demo():
    import gradio as gr

    css = (APP_DIR / "style.css").read_text(encoding="utf-8")
    gradio_major = int(gr.__version__.split(".")[0])
    user_avatar_uri = image_data_uri(ASSET_DIR / "user-chat-avatar.png")
    character_avatar_uri = image_data_uri(
        ASSET_DIR / "anxiety-chat-avatar.png"
    )
    character_uri = image_data_uri(
        ASSET_DIR / "anxiety-character-cutout.png"
    )
    # Inline avatars also cover temporary/loading message rows. A file-cache
    # endpoint can briefly return an invalid URL during Gradio rerenders;
    # embedding the two small thumbnails prevents broken-image icons.
    css += f"""
#chatbot .message-row .avatar-container {{
  background: #fff center 34% / cover no-repeat url('{character_avatar_uri}') !important;
}}
#chatbot .message-row.user-row .avatar-container {{
  background: #fff center / cover no-repeat url('{user_avatar_uri}') !important;
}}
#chatbot .message-row .avatar-container img {{
  opacity: 0 !important;
}}
"""
    decoration_html = build_decoration_html()
    blocks_options = {
        "title": "불안이의 걱정 정리소",
        "fill_height": True,
    }

    if gradio_major < 6:
        blocks_options["css"] = css

    chatbot_options = {
        "value": [{"role": "assistant", "content": INITIAL_GREETING}],
        "height": 560,
        "autoscroll": True,
        "show_label": False,
        "layout": "bubble",
        "avatar_images": (
            {"path": str(ASSET_DIR / "user-chat-avatar.png"), "url": user_avatar_uri},
            {"path": str(ASSET_DIR / "anxiety-chat-avatar.png"), "url": character_avatar_uri},
        ),
        "line_breaks": True,
        "elem_id": "chatbot",
    }

    if gradio_major < 6:
        chatbot_options["type"] = "messages"

    with gr.Blocks(**blocks_options) as demo:
        gr.HTML(decoration_html, elem_id="decor-layer")

        model_status = gr.HTML(
            model_readiness_html(),
            elem_id="model-readiness",
        )

        with gr.Column(elem_id="chat-card"):
            gr.HTML(
                f"""
                <header id="title-block">
                  <h1><span>불안이</span>의 걱정 정리소</h1>
                  <p>불안은 많지만, 확실하게 정리해줄게.</p>
                  <span class="model-variant-label">{MODEL_VARIANT_LABEL}</span>
                </header>
                """
            )
            credit_status = gr.HTML(
                credit_badge(
                    "<strong>조회 중...</strong>",
                    muted=True,
                ),
                elem_id="credit-status",
            )
            gr.HTML(
                f"""
                <div id="character-perch" aria-hidden="true">
                  <img class="anxiety-character" src="{character_uri}" alt="">
                </div>
                """,
                elem_id="character-top",
            )

            with gr.Group(elem_id="dm-shell"):
                gr.HTML(
                    f"""
                    <div id="dm-header">
                      <div class="dm-avatar">
                        <img src="{character_uri}" alt="불안이 프로필">
                        <span class="online-dot" aria-label="대화 가능"></span>
                      </div>
                      <div class="dm-profile">
                        <strong>불안이</strong>
                        <small>같이 걱정을 정리하는 중</small>
                      </div>
                    </div>
                    """,
                    elem_id="dm-top",
                )
                chatbot = gr.Chatbot(**chatbot_options)

            model_history = gr.State([])
            model_ready = gr.State(False)
            model_ready_timer = gr.Timer(value=7.0, active=True)

            with gr.Row(elem_id="composer-row"):
                message_box = gr.Textbox(
                    placeholder="불안이가 모델을 준비하는 중이야...",
                    interactive=False,
                    lines=1,
                    max_lines=5,
                    autoscroll=True,
                    show_label=False,
                    container=False,
                    elem_id="message-box",
                    scale=12,
                    min_width=0,
                )
                send_button = gr.Button(
                    "",
                    elem_id="send-button",
                    interactive=False,
                    scale=1,
                    min_width=74,
                )

            clear_button = gr.Button(
                "대화 새로 시작하기",
                elem_id="clear-button",
            )

            send_event = send_button.click(
                fn=handle_send,
                inputs=[message_box, chatbot, model_history, model_ready],
                outputs=[message_box, chatbot, model_history, send_button, clear_button],
                queue=True,
                trigger_mode="once",
                show_progress="hidden",
            )
            clear_button.click(
                fn=reset_chat,
                outputs=[chatbot, model_history, message_box],
                queue=False,
            )
            # The local Gradio app is visible immediately, but sending stays
            # disabled until Modal reports that the GPU server is warmed up.
            readiness_outputs = [
                model_status, message_box, send_button,
                model_ready, model_ready_timer,
            ]
            demo.load(
                fn=check_model_readiness,
                outputs=readiness_outputs,
                queue=False,
                show_progress="hidden",
            )
            model_ready_timer.tick(
                fn=check_model_readiness,
                outputs=readiness_outputs,
                queue=False,
                show_progress="hidden",
            )
            demo.load(fn=None, js=ENTER_TO_SEND_JS, queue=False)
            demo.load(
                fn=fetch_credit_status,
                outputs=credit_status,
                queue=False,
            )

    return demo, css, gradio_major


if __name__ == "__main__":
    demo, css, gradio_major = build_demo()
    launch_options = {
        "server_name": "0.0.0.0",
        "server_port": int(os.environ.get("PORT", "7861")),
        "show_error": False,
    }

    if gradio_major >= 6:
        launch_options["css"] = css

    demo.queue(default_concurrency_limit=1).launch(**launch_options)
