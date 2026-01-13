# 🏥 M.A.R.S. 2025 - SNUBH 의무기록 생성 데이터톤

> **Medical Auto-documentation with Real-world Structuring**  
> 분당서울대병원 LLM Clinical Note Generation Challenge

<p align="center">
  <img src="https://img.shields.io/badge/Preliminary-Top%2010%20%2F%20100%20Teams-brightgreen" alt="Preliminary">
  <img src="https://img.shields.io/badge/Finals-6th%20Place-blue" alt="Finals">
  <img src="https://img.shields.io/badge/Team-우걱우걱-orange" alt="Team">
</p>

---

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [대회 결과](#-대회-결과)
- [팀 소개](#-팀-소개)
- [과제 개요](#-과제-개요)
- [시스템 아키텍처](#-시스템-아키텍처)
- [기술 스택](#-기술-스택)
- [예선: Task별 상세 접근법](#-예선-task별-상세-접근법)
- [본선: 퇴원기록지 자동 생성](#-본선-퇴원기록지-자동-생성)
- [결과 및 성과](#-결과-및-성과)
- [실행 방법](#-실행-방법)
- [한계점 및 향후 개선 방향](#-한계점-및-향후-개선-방향)
- [참고 자료](#-참고-자료)

---

## 🎯 프로젝트 개요

### 대회 정보

| 항목 | 내용 |
|------|------|
| **대회명** | M.A.R.S. 2025 (Medical Auto-documentation with Real-world Structuring) |
| **주최/주관** | 분당서울대병원 의료인공지능센터, SNUBH 빅데이터센터, AI NATION |
| **후원** | AWS 코리아, 경기도경제과학진흥원, 정보통신기획평가원, 한국보건산업진흥원 |
| **과제** | 실제 임상 데이터를 활용한 의무기록 자동 생성 |

### 평가 기준

**예선**: 정량 평가(BERTScore, F1) + 정성 평가(LLM Evaluator)  
**본선**: 문서 품질(70점) + 전략 설계(30점)

---

## 📊 대회 결과

### 예선 (Preliminary Round)

| 항목 | 점수 |
|------|------|
| **최종 점수** | 72.31점 |
| 정량 평가 + 정성 평가1 | 39.97 / 60점 |
| 정성 평가2 | 32.33 / 40점 |
| **순위** | 🏆 100팀 중 10위권 (본선 진출) |

### 본선 (Finals)

| 항목 | 결과 |
|------|------|
| **최종 순위** | 🏆 6위 |
| 평균 생성 시간 | ~33초/건 |
| 평균 출력 길이 | ~2,000자 |
| 토큰 효율 | 입력 4,096 / 출력 1,500 제한 |

---

## 👥 팀 소개: 우걱우걱

| 이름 | 역할 | 담당 업무 |
|------|------|----------|
| **이은비** (팀장) | Task A 리드 | 입원 경과 요약, 프롬프트 엔지니어링 |
| **이승병** | Task C 리드 | 프로젝트 총괄, ICD-10 코드 예측, 전체 파이프라인 아키텍처 설계 |
| **이승희** | Task B 리드 | 방사선 판독문 요약, 데이터 분석, 자체 평가 모듈 설계 |
| **박찬수** | 실험/검증 | 모델 실험, 성능 최적화, 자동화 파이프라인 구축 |

---

## 📝 과제 개요

### 예선 과제 (3개 Task)

| Task | 과제명 | 설명 |
|------|--------|------|
| **A** | 입원 경과 요약 | 복잡한 의무기록 → 임상적으로 신뢰할 수 있는 Brief Hospital Course 생성 |
| **B** | 방사선 판독문 요약 | Findings/Comparison → IMPRESSION 핵심 소견 추출 |
| **C** | ICD-10 코드 예측 | 의료 텍스트 → 진단에 해당하는 ICD-10 코드 예측 |

### 본선 과제

| 과제 | 설명 |
|------|------|
| **퇴원기록지 자동 생성** | 실제 임상 데이터를 활용한 Discharge Summary 자동 생성 |

### 본선 데이터 구성

| 파일명 | 설명 |
|--------|------|
| `admission.csv` | 입원·퇴원일자, 재원일수, 응급실/ICU 정보 |
| `chief_complaint.csv` | 초진 시 주호소(Chief Complaint) |
| `diagnosis.csv` | 입원 기간 중 진단된 KCD 코드/명칭 |
| `medical_note.csv` | 초진, 입원경과, 타과의뢰/회신, 수술·마취 등 기록 |
| `nursing_note.csv` | 입원동기, 입원경로, 의식상태, 주증상 |
| `surgery_anesthesia.csv` | 수술일자, 수술코드(ICD9CM), 마취 종류 |

---

## 🏗 시스템 아키텍처

### 핵심 철학: "Extract First, LLM Minimal"

```
┌─────────────────────────────────────────────────────────────┐
│                    전략적 역할 분담                           │
├─────────────────────────────────────────────────────────────┤
│  📊 Rule-based (90%)          │  🤖 LLM (10%)              │
│  ─────────────────────────────┼────────────────────────────│
│  • 날짜/수치 추출             │  • 입원경과 서술 생성       │
│  • KCD/ICD 코드 매핑          │  • Patient Summary 작성    │
│  • 구조화된 데이터 파싱       │  • 자연스러운 문장 연결     │
│  • 가드레일 플래그            │                            │
└─────────────────────────────────────────────────────────────┘
```

### 왜 이 전략인가?

1. **사실 정확성 보장**: 수치·날짜·진단명은 규칙 기반으로 100% 정확도 확보
2. **환각(Hallucination) 최소화**: LLM이 "창작"할 수 있는 영역을 최소화
3. **처리 속도 향상**: LLM 토큰 사용량 63% 절감
4. **재현성 확보**: 동일 입력에 대해 일관된 출력 보장

### 전체 파이프라인 (예선 → 본선)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hybrid Pipeline Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│   │  전처리       │───▶│  LLM 추론    │───▶│  후처리       │      │
│   │ Preprocessing │    │   EXAONE    │    │ Postprocess  │      │
│   └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                    │                    │              │
│         ▼                    ▼                    ▼              │
│   • 노이즈 제거         • 역할 부여          • 형식 교정          │
│   • 약어 확장          • 제약 조건          • 용어 표준화         │
│   • 구조화             • 템플릿 강제        • 정보 보강          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 본선 상세 파이프라인

```
[Raw CSV Data]
      │
      ▼
┌─────────────────────────────────────────┐
│  Stage 1: 데이터 표준화                   │
│  • 컬럼명 정규화 (한글 → 영문)           │
│  • 날짜 형식 통일 (pandas datetime)      │
│  • encounter_id 생성 (patient_id + date) │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  Stage 2: 이벤트 통합 및 매칭             │
│  • 소스별 표준 스키마 변환               │
│  • encounter ↔ event 매칭               │
│  • 가드레일 플래그 부착                  │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  Stage 3: 경량 RAG (TF-IDF 기반)          │
│  • 섹션별 키워드 라우팅                  │
│  • 과별 가중치 부스팅                    │
│  • Top-K 스니펫 선택                     │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  Stage 4: LLM 생성 (EXAONE-3.5-7.8B)      │
│  • 구조화된 프롬프트 적용                │
│  • 과별 특화 힌트 주입                   │
│  • 금지 표현 필터링                      │
└─────────────────────────────────────────┘
      │
      ▼
[Discharge Summary (Markdown)]
```

---

## 🔧 기술 스택

| 구분 | 예선 | 본선 |
|------|------|------|
| **주요 모델** | `Llama-3.1-8B-Instruct` | `EXAONE-3.5-7.8B-Instruct` |
| **백업 모델** | - | `Llama-3.1-8B-Instruct` |
| **검색** | - | TF-IDF 기반 경량 RAG |
| **평가** | BERTScore, LLM Evaluator, F1 | 문서 품질 + 전략 설계 |
| **환경** | Ollama (로컬 테스트) | AWS EC2 g6e.8xlarge (L40S) |
| **언어** | Python 3.x | Python 3.12+ |

### 모델 비교 근거

| 모델 | 한국어 의료 텍스트 | 선택 이유 |
|------|------------------|----------|
| EXAONE-3.5-7.8B | ⭐⭐⭐⭐⭐ | 한국어 특화, 의료 용어 이해도 높음 |
| Llama-3.1-8B | ⭐⭐⭐ | 영문 기반, 한국어 의료 용어 약함 |

---

## 📋 예선: Task별 상세 접근법

### Task A: 입원 경과 요약 (Brief Hospital Course)

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
- 프롬프트 → 구조 강제 → 정성 평가 점수 확보
- 전처리 → 어휘 제어 → BERTScore 방어
- 후처리 → 실수 보정 → 안정성 극대화

---

### Task B: 방사선 판독문 요약 (Radiology Impression)

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

## 🏥 본선: 퇴원기록지 자동 생성

### 핵심 컴포넌트

#### 1. 데이터 표준화 레이어

```python
# 컬럼 매핑 구조 (예시)
COLUMN_MAPPING = {
    "환자번호": "patient_id",
    "입원일자": "admit_date", 
    "퇴원일자": "discharge_date",
    "입원시 진료과": "dept_admit",
}
```

#### 2. 이벤트 소스별 표준화 함수

| 소스 | 표준화 함수 | 주요 추출 필드 |
|------|------------|---------------|
| Chief Complaint | `std_from_cc()` | 주호소 텍스트, 작성일자 |
| Diagnosis | `std_from_dx()` | KCD 코드, 한글명 |
| Medical Note | `std_from_med()` | 서식명, 항목값, 진료과 |
| Nursing Note | `std_from_nurse()` | 항목명, 항목값 |
| Surgery | `std_from_surg()` | ICD9CM, 마취종류, ASA |

#### 3. 가드레일 시스템

```python
# 품질 관리 플래그
GUARDRAIL_FLAGS = {
    "out_of_range": "이벤트가 입원 구간을 벗어남",
    "conflict_same_day": "같은 날 상충되는 상태 기록",
    "mixed_status_same_day": "호전/악화 혼재",
    "measure_flags": "검사 수치 참조범위 이탈"
}
```

#### 4. 과별 키워드 부스팅

```python
# 과별 검색 가중치 (예시)
DEPT_KEYWORD_BOOSTS = {
    "신장내과": {"Cr": 1.5, "eGFR": 1.4, "BUN": 1.2},
    "신경외과": {"MRI": 1.4, "laminectomy": 1.6},
    "순환기내과": {"ECG": 1.4, "troponin": 1.4, "CAG": 1.2},
    "소화기내과": {"내시경": 1.4, "EGD": 1.2, "biopsy": 1.0},
}
```

#### 5. 경량 RAG (TF-IDF 기반)

임베딩 모델 대신 TF-IDF를 사용하여 **17배 빠른** 검색 속도 달성

```python
def lite_search(pool, keywords, topk=6):
    """키워드 가중치 기반 경량 검색"""
    scores = []
    for record in pool:
        text = record.get("text", "").lower()
        score = sum(
            weight for kw, weight in keywords.items() 
            if kw.lower() in text
        )
        scores.append((score, record))
    return [r for _, r in sorted(scores, reverse=True)[:topk]]
```

### 출력 형식 (가이드라인 준수)

```markdown
## 입원사유 및 병력요약
- **Chief Complaint(★)**: [주호소]
- **Present Illness(★)**: [현병력]
- **Past History(★)**: [과거력]

## 입원경과(★)
- [날짜] 주요 사건/검사/시술

## 입원결과(★)
- 치료 결과 및 퇴원 시 상태

## 검사결과(★)
- ① 검체검사 → ② 영상검사 → ③ 기능검사/시술 → ④ 병리

## Patient Summary(★)
- 5-10문장 요약
```

---

## 📈 결과 및 성과

### 예선 Task별 성과

| Task | 주요 성과 |
|------|----------|
| **Task A** | BERTScore + 정성 평가 동반 상승, 환각 현저히 감소 |
| **Task B** | BERTScore 0.8XX 달성, 비교 문구 표준화로 어휘 일치도 향상 |
| **Task C** | F1 Score 0.6245 → 0.9190 (47.2% 향상), 완전 불일치율 0% |

### 본선 성과

| 항목 | 결과 |
|------|------|
| **최종 순위** | 6위 |
| **ICD-10 예측 F1** | 0.919 (+47% 향상) |
| **LLM 토큰 사용량** | 63% 절감 |
| **환각률** | Near-zero (구조화 데이터) |

### 주요 개선 포인트

1. **가드레일 시스템**: 입원 구간 외 이벤트 자동 필터링
2. **과별 최적화**: 4개 진료과 특화 키워드 및 힌트
3. **금지 표현 필터**: "확진", "반드시", "절대" 등 과도한 확정 표현 제거
4. **동적 키워드 추출**: TF-IDF 기반 encounter별 핵심 키워드 자동 추출

---

## 💻 실행 방법

### 환경 요구사항

- **GPU**: NVIDIA L40S (AWS EC2 g6e.8xlarge)
- **모델**: LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
- **Python**: 3.12+

---

## ⚠️ 한계점 및 향후 개선 방향

### 공통 한계

- 정규식 기반 로직의 비표준 형식 취약성
- 자동 보강의 잠재적 안전성 위험
- 데이터 분포 불균형에 따른 그룹별 성능 편차

---

## 🛡️ 데이터 보안 준수사항

- ✅ 모든 작업은 지정된 AWS 환경 내에서만 수행
- ✅ 외부 API(ChatGPT, Claude, Gemini 등) 사용 금지
- ✅ 데이터 다운로드, 캡처, 외부 저장 금지
- ✅ 예시 문장은 재구성하여 실제 정보 미포함

---

## 📚 참고 자료

- [퇴원기록지 작성 가이드라인]
- [본선 자료 및 제출 안내]
- [EXAONE-3.5 모델 정보](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct)
- [Llama 3.1 Model Card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [BERTScore Paper](https://arxiv.org/abs/1904.09675)
- [ICD-10-CM Official Guidelines](https://www.cdc.gov/nchs/icd/icd-10-cm.htm)

---

## 📁 프로젝트 구조

```
📦 MARS-2025
├── 📂 preliminary/                # 예선
│   ├── 📄 예선.py                 # 메인 프로세서 코드
│   ├── 📄 우걱우걱_예선제출.pdf    # 예선 결과 보고서
│   └── 📄 processor.py            # DatathonProcessor 베이스 클래스
├── 📂 finals/                     # 본선
│   ├── 📄 submit.ipynb            # 본선 제출 노트북
│   └── 📂 docs/                   # 가이드라인 문서
└── 📄 README.md                   # 프로젝트 설명
```

---

<p align="center">
  <strong>🏆 예선 100팀 중 10위권 본선 진출 → 본선 6위 🏆</strong>
</p>

<p align="center">
  <strong>팀 우걱우걱</strong><br>
  "Extract First, LLM Minimal" - 의료 AI의 새로운 패러다임
</p>

<p align="center">
  주최/주관: 의료인공지능센터 | SNUBH 빅데이터센터 | AI NATION<br>
  후원: AWS 코리아 | 경기도경제과학진흥원 | 정보통신기획평가원 | 한국보건산업진흥원
</p>
