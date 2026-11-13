# Anxiety Chatbot — Qwen2.5-7B + LoRA + RAG

불안이 캐릭터 챗봇의 Base / Fine-Tuned 모델을 비교하는 Serving 코드입니다. 두 앱은 Modal GPU(vLLM) 추론과 Gradio UI를 독립적으로 사용합니다.

| | Base | Fine-Tuned |
|---|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` | `somsom317/persona-qwen25-7b-merged-final` |
| Modal App | `anxiety-chatbot-base` | `anxiety-chatbot` |
| Modal Secret | `anxiety-chatbot-base-app-token` | `anxiety-chatbot-finetuned-app-token` |
| Local UI | `http://127.0.0.1:7860` | `http://127.0.0.1:7861` |

## Structure

- `Base/app.py` / `Fine-Tuned/app.py`: Modal vLLM API, RAG, Gradio UI 및 비용 조회
- `*/data/`: System Prompt와 RAG 문서 (두 모델 동일)
- `*/assets/`: UI에서 실제 사용하는 이미지 3종
- `*/style.css`: 채팅 화면 스타일
- `*/render.yaml`: 선택사항 — UI를 Render에 배포할 때 사용

## Deployment

Python 3.11 권장. 각 모델 폴더에서 별도로 실행합니다.

```bash
pip install -r requirements.txt
modal deploy app.py
```

Modal에서 위 표의 **Secret**을 만들어 `APP_API_TOKEN`과 `HF_TOKEN`을 설정합니다. 두 앱은 서로 다른 Secret을 사용합니다. Fine-Tuned 모델은 Hugging Face에 **병합된 체크포인트**가 업로드되어 있어야 합니다.

배포 후 Modal에서 생성된 `modal_api` 주소(끝에 `/chat`을 붙이지 않은 기본 주소)를 로컬 UI에 지정합니다. 예: **Windows CMD**

```bat
set MODAL_API_URL=https://YOUR-MODAL-ENDPOINT.modal.run
set MODAL_API_TOKEN=YOUR_APP_API_TOKEN
python app.py
```

PowerShell은 `set` 대신 다음처럼 설정합니다.

```powershell
$env:MODAL_API_URL="https://YOUR-MODAL-ENDPOINT.modal.run"
$env:MODAL_API_TOKEN="YOUR_APP_API_TOKEN"
python app.py
```

각 폴더에서 실행하며 `Base`와 `Fine-Tuned`의 URL과 토큰을 혼동하지 마세요. 실사용 토큰은 GitHub에 올리지 않고 Modal Secret 및 로컬 환경변수로만 관리합니다. `.env.example`은 예시이며 자동으로 읽히지 않습니다.

## Inference and latency

- Both: vLLM on Modal L4/A10 GPU, `temperature=0.8`, `top_p=0.9`, `repetition_penalty=1.0`, `max_tokens=384`, `seed=42`, Qwen chat template + `<|im_end|>`.
- Both: 같은 RAG 자료와 System Prompt 사용. LLM 라우터가 영화 자료 검색 필요 여부를 먼저 판단하며, 검색이 필요하고 관련 문서가 있을 때만 `<context>`를 붙입니다.
- **Cold start:** 두 모델 API 모두 `min_containers=1` (GPU worker)로 설정해, 유휴 후 첫 채팅이 Qwen 7B와 임베딩 모델 재로딩을 기다리던 문제를 없앰습니다. 배포 직후에는 상시 컨테이너가 한 번 준비될 때까지 모델 로딩이 진행됩니다.
- **CUDA Graph:** Base / Fine-Tuned 모두 기본값 ON (`enforce_eager=False`, `max_num_seqs=1`, `cudagraph_capture_sizes=[1]`). 시작 시 CUDA Graph 준비로 첫 로딩이 더 오래 걸릴 수 있지만, 모델 초기화 후 반복 생성에서 유리할 수 있습니다. GPU 메모리 부족/시작 실패 시 Modal 컨테이너의 `VLLM_ENFORCE_EAGER=1`로 비활성화한 뒤 재배포하세요. 로컬 CMD의 `set`만으로는 원격 Modal 환경변수가 변경되지 않습니다.
- **Non-streaming:** UI가 전체 결과를 받은 뒤 보여 주므로 생성 중에도 기다리는 화면이 나타납니다. 실제 속도 분석에는 Modal 로그의 `startup_seconds`, `vllm_init_seconds`, `rag_seconds`, `decode_seconds`를 참고합니다(로그 항목은 모델에 따라 일부 다름).
- `min_containers=1`은 채팅 지연을 없애는 대신 유휴 시간에도 GPU 비용이 발생합니다. 비용 우선 운영으로 되돌릴 때는 각 `app.py`의 GPU `modal_api` 설정을 `min_containers=0`으로 변경하세요.

### Check deployed model

브라우저 또는 HTTP GET으로 각 Modal API의 `/health` 확인:

- `model_id`: 위 모델 ID와 일치
- `engine`: `vllm`
- `cuda_graphs_enabled`: `true` (재배포 이후 기본 설정)
- `prompt_sha256`: `45fd98b4fc5e` (현재 두 앱 동일)

### Notes

GPU 및 실제 Modal 요청은 이 저장소의 오프라인 검증만으로 보장되지 않습니다. 인물/영화 관련 이미지의 공개 저장소 업로드 권한도 확인하세요.

### 응답 자연스러움 조정

기존 페르소나/RAG 지침을 보존하고, 직접 답변 우선·감탄사/질문 반복 조절 지침만 추가했습니다. Sampling 설정은 Base와 Fine-Tuned에 동일하게 적용했습니다. 실제 생성 결과는 입력 문맥과 모델별 가중치에 따라 달라질 수 있습니다.


### Qwen ChatML / 한국어 출력 점검

- `apply_chat_template(..., add_generation_prompt=True)`로 시스템 / 대화 이력 / 사용자 질문(필요 시 `<context>`) / assistant 생성 시작점을 구성합니다. ChatML 특수토큰을 System Prompt 텍스트에 직접 저장하지 않습니다.
- Part2 페르소나 원문을 유지하고, 답변 우선순위·한국어·Context 활용 지침을 끝에 추가했습니다.
- 드물게 한자가 섞이면 짧은 단일 글자는 출력 전에 제거하고, 연속적인 중국어 유출은 한 번 재생성한 뒤 최종 검사합니다. 예외적 경우 재생성으로 답변 대기 시간이 늘 수 있습니다.
- 정상 답변의 SamplingParams, vLLM, CUDA Graph, RAG 검색, 모델 경로는 변경하지 않았습니다.

### Model readiness before chatting

The Gradio UI starts with the composer disabled. It polls the Modal GPU
`/health` endpoint until Qwen 2.5, BGE-M3 retrieval, CUDA Graph setup, and
a two-token vLLM warm-up request have completed. The chat composer is enabled
only after the expected model ID is ready. On new deployments, GPU restarts,
or a changed Modal Secret, startup can still take time.

The GPU endpoint keeps `min_containers=1`, which incurs GPU charges while
the deployment is active. The readiness indicator does not stream tokens;
full-answer generation latency and a possible Korean-only retry remain.

## Conditional RAG routing (v7)

- Every meaningful user turn is first classified by the **already loaded vLLM Qwen checkpoint** with `temperature=0` for the *router only*. A quick acknowledgement such as `나도 그렇게 생각해!!!` is skipped without any extra generation or vector lookup.
- The router returns `CHAT` (no BGE embedding lookup or `<context>` tag) or `RAG` (context-aware rewritten search query), using the latest real user message and up to 3 recent completed exchanges.
- Only for `RAG` does the app embed the query with BGE-M3, score the cached movie-document vectors, and include relevant documents above the similarity threshold. If no document reaches the threshold, `<context>` is omitted.
- This is an **in-memory precomputed embedding index**, not a standalone Chroma/FAISS service. The cache is built at GPU container startup.
- The final response still uses `temperature=0.8`, `top_p=0.9`, `repetition_penalty=1.0`, `max_tokens=384`, `seed=42`, and the original user-supplied System Prompt, 3 demonstrations and up to 8 recent real conversation exchanges.
- `GET /health`: verify `build=qwen-eval-vllm-conversation-v7-llm-rag-router`, `rag_router=vllm_same_loaded_qwen_deterministic`, and the exact model ID before interacting.
- Modal logs: `[rag-router] route=... retrieve=...` and `[chat] ... rag_docs=...`. Route decisions are model outputs and may be imperfect; the similarity gate and deterministic ACK skip reduce accidental context injection.
- One extra vLLM classification call can add latency on messages not covered by the quick ACK guard; no additional checkpoint is loaded.



## 최신 Chroma + 하이브리드 RAG 변경 사항
`RAG_CHROMA_VERIFICATION.md`를 확인하세요.


### v11: 영화 사건을 묻지 않고 지적하는 경우 처리
- Qwen Router는 영화 장면에 대한 진술·지적(예: "너 그때 그랬잖아")도 사실 근거가 필요하면 RAG로 분류합니다. 물음표 유무로 분기하지 않습니다.
- 사실 질문은 정답 우선, 영화 사건 진술은 근거에 따라 본인 행동 인정/정정 우선, 사용자 고민은 조언 우선입니다.
- 영화 장면을 지적받았는데 사용자의 개인 고민이 없는 경우, 추측성 생활 조언으로 넘어가지 않습니다.
- 3개 Few-shot 대화 묶음 중 영화 사건 예시는 진술 → 자기 행동 인정 → 후속 공감으로 변경했습니다.
- Chroma/BGE-M3/문자 TF-IDF RRF 검색, vLLM, Model ID, Temperature 0.8, 대화 이력 8회 및 CUDA Graph는 유지합니다.
- 이 변경은 로컬 코드 검증이며 Modal에서 실제 Qwen 출력이 반드시 같음을 보장하지 않습니다.
