# Final conversation-aware vLLM serving (Base / Fine-Tuned)

## Confirmed in source code
- `/health` only succeeds after GPU vLLM (`LLM`) initialization, CPU BGE-M3 document embedding, and a real generation warm-up.
- Fine-Tuned model is pinned to `somsom317/persona-qwen25-7b-merged-final` (the merged LoRA model from Part3).
- Base model is pinned to `Qwen/Qwen2.5-7B-Instruct`.
- Main inference uses `runtime["llm"].generate(...)`; no Transformers `generate()` or OpenAI API generates chat responses.
- Part3 generation parameters: `temperature=0.8`, `top_p=0.9`, `repetition_penalty=1.0`, `max_tokens=384`, `seed=42`, and Qwen `<|im_end|>` stop token.
- Original system prompt text preserved exactly. Three ChatML demonstration dialogues and eight completed real conversational turns are supplied separately.
- Non-story conversation and acknowledgement messages do not trigger RAG retrieval; story questions and appropriate follow-ups may use BGE-M3 documents.
- The most recent user utterance is last in the model input. The Gradio event is a single, serialized generator: responses cannot be mixed with another in-flight turn.
- API `/chat` response includes `build`, `engine`, `model_id`, `history_messages`; the UI checks the first three and refuses a stale or incorrect backend.

## Deploy
From `Fine-Tuned/`: `py -m modal deploy app.py`, then `py app.py`.
If using Base, deploy and launch from `Base/` instead. The two apps have distinct names.
Ensure the environment `MODAL_API_URL` points to the selected version's **modal_api** endpoint, not billing_api or a prior deployment. `MODAL_API_TOKEN` must equal the backend Modal Secret `APP_API_TOKEN`. Fine-Tuned also needs `HF_TOKEN` authorized to read the private checkpoint.

## Live readiness checks
Open the deployed `MODAL_API_URL` plus `/health` in a browser. Check the JSON contains `ready:true`, `engine:vllm`, correct `model_id`, and `build:qwen-eval-vllm-conversation-v6`, and `gpu_active:true`.
Check the Modal logs for `[gpu]`, `[warmup]`, `[startup]` and `[chat]` entries. Never paste API tokens or Hugging Face credentials into screenshots.

## Multi-turn tests (ONE browser session; no reset between messages)
1. `라일리가 코치 사무실에 들어가기 전에 넌 무슨 생각을 했어?`
2. `나도 그렇게 생각해!!!` — MUST be treated as an acknowledgement; RAG log should show `rag_intent=acknowledgement rag_docs=0`. The reply must not reanswer turn #1.
3. `그런데 라일리가 왜 긴장했는지는 확실해?` — a new story question, retrieved context permitted, say unknown when necessary.
Also verify everyday advice: `내일 친구 만나는데 비가 온대.` -> `응, 나도 그게 좋겠어.`

## Test limitations
Source validation and mock inference tests cannot guarantee the real model wording. Actual GPU deployment, authentication, and multi-turn model outputs must be tested in your Modal account.


### v8: Router 및 직접 답변 개선
- 별도 키워드 강제 검색 없이 이미 로딩된 Qwen이 모든 사용자 발화를 RAG 또는 CHAT으로 판단합니다.
- 영화 사실 확인 질문은 RAG로 분류하도록 라우터 예시를 보강했습니다.
- 검색 실패 시 원래 질문으로 한 번 더 검색하고, 문서 근거가 없으면 추측하지 않도록 안내합니다.
- 답변은 질문을 반복하거나 되묻지 않고 이름·사실·이유를 먼저 말하도록 했습니다.
- 실제 GPU에서 Router 분류 성능은 재배포 후 로그로 검증해야 합니다.
