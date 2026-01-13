# M.A.R.S. 2025 SNUBH 의무기록 생성 데이터톤 - 예선

<p align="center">
  <strong>Medical Auto-documentation with Real-world Structuring: LLM Clinical Note Generation Challenge</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Result-Top%2010%20%2F%20100%20Teams-brightgreen" alt="Result">
  <img src="https://img.shields.io/badge/Score-72.31%20points-blue" alt="Score">
  <img src="https://img.shields.io/badge/Status-Finals%20Qualified-success" alt="Status">
</p>

---

## 📊 대회 결과

| 항목 | 점수 |
|------|------|
| **최종 점수** | 72.31점 |
| 정량 평가 + 정성 평가1 | 39.97 / 60점 |
| 정성 평가2 | 32.33 / 40점 |
| **순위** | 100팀 중 10위권 (본선 진출) |

---

## 👥 팀 소개: 우걱우걱

| 이름 | 역할 | 담당 업무 |
|------|------|----------|
| **이은비** (팀장) | Task A 리드 | 프로젝트 총괄, 입원 경과 요약, 프롬프트 엔지니어링 |
| **이승병** | Task C 리드 | ICD-10 코드 예측, 전체 파이프라인 아키텍처 설계 |
| **이승희** | Task B 리드 | 방사선 판독문 요약, 데이터 분석, 자체 평가 모듈 설계 |
| **박찬수** | 실험/검증 | 모델 실험, 성능 최적화, 자동화 파이프라인 구축 |

---

## 🎯 과제 개요

### Task A: 입원 경과 요약 (Brief Hospital Course)
> 복잡한 의무기록을 임상적으로 신뢰할 수 있는 요약문으로 생성

### Task B: 방사선 판독문 요약 (Radiology Impression)
> Findings/Comparison에서 핵심 소견을 추출하여 IMPRESSION 생성

### Task C: ICD-10 코드 예측
> 의료 텍스트에서 진단에 해당하는 ICD-10 코드 예측

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hybrid Pipeline Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│   │  전처리       │───▶│  LLM 추론    │───▶│  후처리       │      │
│   │ Preprocessing │    │ Llama 3.1   │    │ Postprocess  │      │
│   └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                    │                    │              │
│         ▼                    ▼                    ▼              │
│   • 노이즈 제거         • 역할 부여          • 형식 교정          │
│   • 약어 확장          • 제약 조건          • 용어 표준화         │
│   • 구조화             • 템플릿 강제        • 정보 보강          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 프로젝트 구조

```
📦 MARS-2025-Preliminary
├── 📄 예선.py                    # 메인 프로세서 코드
├── 📄 우걱우걱_예선제출.pdf       # 예선 결과 보고서
├── 📄 README.md                  # 프로젝트 설명
└── 📄 processor.py               # DatathonProcessor 베이스 클래스 (대회 제공)
```

---

## 🔧 기술 스택

| 구분 | 기술 |
|------|------|
| **모델** | `meta-llama/Llama-3.1-8B-Instruct` |
| **평가** | BERTScore, LLM Evaluator, F1 Score |
| **로컬 테스트** | Ollama (llama3.2 + gpt-oss:20b) |
| **언어** | Python 3.x |

---

## 📋 Task별 상세 접근법

### Task A: 입원 경과 요약

#### 핵심 전략: 지능형 하이브리드 파이프라인

```python
# 3단계 파이프라인
1️⃣ 지능형 전처리 (Intelligent Preprocessing)
   - 행정 노이즈 제거 (Dictated By, Electronically Signed By 등)
   - 의학 약어 확장 (HPI → History of Present Illness)
   - 민감 정보 마스킹 ([**...**] → [REDACTED])

2️⃣ 템플릿 프롬프트 (Template Injection Prompting)
   - ROLE: Chief Resident Physician
   - GOAL: 오직 제공된 의료 데이터에만 기반한 요약
   - CONSTRAINTS: 추론 금지, 대화체 금지, 사실성 최우선
   - OUTPUT: 구조화된 Brief Hospital Course

3️⃣ 가드레일 후처리 (Postprocessing with Guardrails)
   - 불필요한 서두 제거
   - 모호한 표현 교정 (appears to be → 명확한 표현)
   - 플레이스홀더 표준화 ([Patient Name] → [REDACTED])
```

#### 성공 요인
- **정량-정성 평가 트레이드오프 극복**
  - 프롬프트 → 구조 강제 → 정성 평가 점수 확보
  - 전처리 → 어휘 제어 → BERTScore 방어
  - 후처리 → 실수 보정 → 안정성 극대화

---

### Task B: 방사선 판독문 요약

#### 핵심 전략: 형식 표준화 + 어휘 보존

```python
# 출력 형식 (고정)
IMPRESSION:
1. [가장 중요한 소견 - 급성/생명위협 우선]
2. [부가 소견 - 선택적, 병합하여 한 문장으로]
3. [비교 문구 - 템플릿 고정]

# 비교 문구 허용 어휘 (정확 일치)
- "Improved compared to prior."
- "Worsened compared to prior."
- "Stable compared to prior."
- "No significant interval change."
- "No prior for comparison."
```

#### 프롬프트 설계 원칙
| 원칙 | 설명 |
|------|------|
| **사실 보존 (Fidelity)** | 원문에 없는 진단/원인/수치 생성 금지 |
| **형식 표준화 (Format Lock)** | 1-3문장, 8-20단어, 영어 전용 |
| **어휘 보존 (Lexical Preservation)** | 도메인 고유어 변형 없이 사용 |

---

### Task C: ICD-10 코드 예측

#### 핵심 전략: "Less is More" - 단순화 원칙

```python
# 성능 변화
초기 (EXAONE):     F1 0.6245
Llama 3.1 적용:    F1 0.7781  (+24.6%)
과적합 단계:       F1 0.6090  (과도한 규칙으로 하락)
최종 (단순화):     F1 0.9190  (+47.2% 최종 향상)
```

#### Few-shot 프롬프트 예시
```
Example 1 (복합 진단):
Hospital Course: "65M acute chest pain, EKG ST elevation, PCI performed. Diabetes history."
Result: I214, E119

Example 2 (폐렴):
Hospital Course: "42F cough and fever. Chest X-ray showed pneumonia."
Result: J189

Example 3 (외상 + 외부원인):
Hospital Course: "78M fell down stairs. Femoral neck fracture, surgery performed."
Result: S72001A, W1830XA
```

#### 후처리: 핵심 4가지 규칙
| 입력 | 출력 | 설명 |
|------|------|------|
| `I21` | `I214` | 심근경색 형식 통일 |
| `I48` | `I4891` | 심방세동 형식 통일 |
| `S06` | `S066X1A` | 외상 코드 7자리 형식 |
| `W18` | `W1830XA` | 외상 원인 7자리 형식 |

---

## 📈 실험 결과 요약

### Task A 성과
- BERTScore와 정성 평가 점수 동반 상승 달성
- 환각(Hallucination) 현저히 감소
- 일관된 구조의 요약문 생성

### Task B 성과
- BERTScore 0.8XXX 구간 달성
- 비교 문구 표준화로 어휘 일치도 향상
- 한글 혼입 문제 해결 (영어 전용 + 한글 스트립)

### Task C 성과
- F1 Score: 0.6245 → 0.9190 (47.2% 향상)
- 완전 불일치율 0% 달성
- 과적합 방지를 위한 최소 규칙 적용

---

## ⚠️ 한계점 및 향후 개선 방향

### 공통 한계
- 정규식 기반 로직의 비표준 형식 취약성
- 자동 보강의 잠재적 안전성 위험
- 데이터 분포 불균형에 따른 그룹별 성능 편차

### 향후 개선 계획
1. **의료 NER 도입**: 정규식 → 임상용 NER 모델 (medspaCy, clinical-BERT)
2. **Few-shot 프롬프팅 강화**: 임상적으로 전달력 우수한 예시 추가
3. **RAG 통합**: ICD-10 코딩 가이드라인 지식베이스 구축
4. **실시간 피드백 루프**: 의료진 피드백 반영 메커니즘

---

## 🚀 실행 방법

```python
# 기본 사용법 (대회 환경)
from 예선 import TaskAProcessor, TaskBProcessor, TaskCProcessor

# Task A: 입원 경과 요약
processor_a = TaskAProcessor()
result_a = await processor_a.process(medical_record_data)

# Task B: 방사선 판독문 요약
processor_b = TaskBProcessor()
result_b = await processor_b.process(radiology_report_data)

# Task C: ICD-10 코드 예측
processor_c = TaskCProcessor()
result_c = await processor_c.process(hospital_course_data)
```

---

## 📚 참고 자료

- [Llama 3.1 Model Card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [BERTScore Paper](https://arxiv.org/abs/1904.09675)
- [ICD-10-CM Official Guidelines](https://www.cdc.gov/nchs/icd/icd-10-cm.htm)

---

## 📜 라이선스

이 프로젝트는 M.A.R.S. 2025 데이터톤 참가작입니다.

---

<p align="center">
  <strong>🏆 100팀 중 10위권 본선 진출 🏆</strong>
</p>

<p align="center">
  주최/주관: 의료인공지능센터 | SNUBH 빅데이터센터 | AI NATION<br>
  후원: AWS 코리아 | 경기도경제과학진흥원 | 정보통신기획평가원 | 한국보건산업진흥원
</p>
