# 🚀 인사이드아웃2 불안이 Persona 챗봇 Fine-Tuning

> 영화 캐릭터 고유의 성격과 말투, 과거 경험을 반영한 Qwen LoRA Fine-Tuning 기반 페르소나 챗봇 개발

---

## 📌 1. 프로젝트 개요 (Overview)

- **개발 기간:** 2026.10.20 ~ 2026.10.27
- **프로젝트 유형:** 개인 프로젝트
- **개발 목적:** 영화 캐릭터의 정체성, 성격, 말투, 상호작용 방식과 과거 경험을 반영하여 일관된 페르소나 챗봇 구현
- **주요 결과:**
  - Persona Dialogue 및 Memory-Grounded Dialogue 학습 데이터 **총 436개 구축**
  - Qwen2.5-7B-Instruct 기반 LoRA Fine-Tuning 적용
  - 영화 에피소드 기반 RAG를 결합하여 캐릭터의 과거 경험을 반영한 답변 생성
  - Base 모델과 Fine-Tuned 모델의 대화 응답을 비교하여 캐릭터 말투 및 맥락 반영 수준 확인

영화 캐릭터와 실제로 대화하는 경험을 제공하기 위해, 캐릭터 고유의 성격과 말투뿐 아니라 영화 속 과거 경험까지 반영하는 페르소나 챗봇을 개발했습니다.

단순 프롬프팅을 넘어 **캐릭터 정체성을 학습하는 Fine-Tuning과 과거 경험을 검색하는 RAG를 결합**하여 일상적인 대화에서도 캐릭터의 일관성을 유지하는 것을 목표로 했습니다.

---

## 🛠️ 2. 기술 스택 (Tech Stack)

- **Language:** Python
- **LLM:** Qwen2.5-7B-Instruct, GPT-4o
- **Fine-Tuning:** LoRA, PEFT
- **RAG & Retrieval:** RAG, 영화 에피소드 기반 검색
- **Data Generation:** OpenAI API, Persona Prompt
- **Interface:** Gradio

---

## 🏗️ 3. 시스템 아키텍처 (Architecture)

### 전체 파이프라인

**Persona Prompt 설계 → 대화 데이터 생성 → 통합 학습 데이터 구축 → LoRA Fine-Tuning → 영화 에피소드 RAG 결합 → 캐릭터 챗봇**

### ① Persona Prompt Design

캐릭터의 고유한 특징을 일관되게 반영하기 위해 Persona Prompt를 3가지 요소로 구성했습니다.

- **Identity:** 캐릭터의 정체성과 역할
- **Personality:** 걱정 많고 예민하지만 관찰력과 계획성이 뛰어난 성격
- **Tone & Style:** 과장된 감정 표현, 빠른 반응, 다급한 말투와 해결책 제시

### ② Dataset Generation

**A. Persona Dialogue Dataset**

- 캐릭터 정체성·성격·말투를 반영한 Persona Prompt 작성
- GPT-4o를 활용하여 Single-Turn 및 Multi-Turn 대화 생성
- 일반적인 일상 대화에서도 캐릭터 특성이 나타나도록 데이터 구성

**B. Memory-Grounded Dataset**

- 영화 속 캐릭터와 과거 주요 에피소드 구축
- 에피소드 기반 질문 생성
- 관련 에피소드를 Context로 제공하여 GPT-4o로 답변 생성
- 캐릭터의 과거 경험을 반영하는 대화 데이터 구성

**통합 학습 데이터: 총 436 Samples**

| Dataset Type | Samples |
|---|---:|
| Single-Turn | 237 |
| Context-Grounded | 150 |
| Multi-Turn | 49 |
| **Total** | **436** |

### ③ LoRA Fine-Tuning

- **Base Model:** Qwen2.5-7B-Instruct
- **Fine-Tuning:** LoRA (Attention + MLP)
- **LoRA Rank (r):** 16
- **LoRA Alpha:** 32
- **LoRA Dropout:** 0.1

캐릭터 말투와 상호작용 방식뿐 아니라 과거 에피소드를 참고한 대화 패턴까지 학습하도록 구성했습니다.

### ④ RAG 기반 캐릭터 챗봇

- 영화 에피소드 관련 질문에 대해 RAG로 관련 문서 검색
- 검색된 과거 경험을 Context로 활용하여 답변 생성
- 일반적인 일상 대화와 영화 에피소드 질문을 모두 처리하는 챗봇 구현

---

## ✨ 4. 주요 기능 및 본인 역할 (Key Features & Contributions)

### ① 캐릭터 Persona Prompt 설계

- 영화 속 불안이 캐릭터의 정체성, 성격, 말투, 상호작용 방식 분석
- 미래의 위험을 먼저 걱정하고 여러 가능성을 예측하는 캐릭터 특성 정의
- 빠른 반응, 감탄사, 다급한 말투, 문제 해결 중심의 대화 스타일 구성

### ② Persona & Memory-Grounded 학습 데이터 구축

- GPT-4o 기반 대화 데이터 생성 파이프라인 구현
- Single-Turn, Multi-Turn, Context-Grounded 유형으로 데이터 구성
- 영화 속 과거 사건을 참고한 응답 생성 데이터 구축
- 서로 다른 데이터 유형을 통합하여 LoRA 학습에 활용

### ③ Qwen LoRA Fine-Tuning

- Qwen2.5-7B-Instruct 모델에 LoRA Fine-Tuning 적용
- Attention 및 MLP 모듈을 학습 대상으로 설정
- 캐릭터 고유의 말투와 과거 기억을 반영한 대화 패턴 학습

### ④ 캐릭터 챗봇 구현 및 비교 평가

- Gradio 기반 캐릭터 챗봇 인터페이스 구현
- 일반 대화와 영화 에피소드 기반 질문을 모두 처리
- Base 모델과 Fine-Tuned 모델의 동일 질문 응답 비교

**비교 평가에서 확인한 변화**

| 평가 항목 | Base Model | Fine-Tuned Model |
|---|---|---|
| Tone Consistency | 일반적인 어조 또는 과도하게 설명적인 응답 | 불안이 특유의 감탄사, 빠른 반응, 다급한 말투 강화 |
| Context Consistency | 캐릭터 맥락에서 벗어난 응답 발생 | 질문의 상황과 캐릭터 관점을 반영한 응답 생성 |
| Reference Similarity | 영화 에피소드와 관련 없는 응답 발생 | 관련 에피소드를 바탕으로 한 설명 및 답변 개선 |

※ 위 비교는 포트폴리오에 제시된 대화 예시를 바탕으로 한 정성적 비교이며, 별도의 정량 평가 수치는 아닙니다.

---

## 💡 5. 배운 점 (Troubleshooting)

### ① 프롬프팅과 Fine-Tuning의 역할 차이

초기에는 Persona Prompt를 통해 캐릭터의 정체성, 성격, 말투를 정의했지만, 프롬프트만으로 모든 대화에서 캐릭터의 특성을 일관되게 유지하는 데 한계가 있었습니다.

이를 보완하기 위해 캐릭터 특성을 반영한 대화 데이터를 직접 구축하고 LoRA Fine-Tuning을 적용했습니다. 이를 통해 **프롬프트 설계뿐 아니라 학습 데이터 구성이 모델의 응답 스타일에 영향을 준다는 점**을 경험했습니다.

### ② 캐릭터 말투와 과거 경험의 분리

캐릭터의 고유한 말투를 학습하는 것과 영화 속 과거 사건을 정확하게 언급하는 것은 서로 다른 문제였습니다.

이를 고려하여 Persona Dialogue와 Memory-Grounded Dialogue를 구분하여 구축하고, 챗봇에는 영화 에피소드 기반 RAG를 결합했습니다.

이를 통해 페르소나의 표현 방식은 Fine-Tuning으로, 외부 문서에 포함된 과거 경험은 RAG로 보완하는 구조를 설계했습니다.

### ③ 학습 데이터 유형의 다양성

Single-Turn 데이터만으로는 여러 차례 이어지는 대화나 과거 경험을 참고해야 하는 질문을 충분히 반영하기 어렵다고 판단했습니다.

이에 따라 Single-Turn, Multi-Turn, Context-Grounded 데이터를 통합하여 학습하고, 다양한 대화 상황에서도 캐릭터 특성이 드러나도록 구성했습니다.

---
