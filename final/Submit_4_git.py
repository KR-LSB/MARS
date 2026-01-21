# -*- coding: utf-8 -*-
"""
MARS 2025 본선 제출용 코드
팀: 우걱우걱

================================================================================
⚠️ 데이터 보안 안내
================================================================================
- 이 코드는 실제 환자 데이터를 포함하지 않습니다.
- 코드 내 모든 예시 문장은 형식 설명을 위한 것으로, 실제 진료 정보가 아닙니다.
- 실행 시 필요한 CSV 데이터(admission.csv 등)는 별도로 준비해야 합니다.
- S3 경로는 placeholder로 대체되어 있으며, 실제 사용 시 변경이 필요합니다.
================================================================================
"""

# ============================================================
# 셀 1
# ============================================================
import datetime as dt
NOTEBOOK_START_TIME = dt.datetime.now().isoformat()
# === CSV 로더 (===
import os
import pandas as pd

BASE_DIR = "./sample"

def read_csv(path: str) -> pd.DataFrame:
    """utf-8-sig → utf-8 → cp949 순서로 시도해서 읽기. 컬럼 공백 제거."""
    full = os.path.join(BASE_DIR, path)
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            df = pd.read_csv(full, encoding=enc)
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"CSV 읽기 실패: {full} | {last_err}")

# 파일명 매핑
df_admission = read_csv("admission.csv")          # 입원/퇴원/과 정보
df_cc        = read_csv("chief_complaint.csv")    # chief complaint
df_diagnosis = read_csv("diagnosis.csv")          # KCD 진단
df_medical   = read_csv("medical_note.csv")       # 의무기록(의사)
df_nursing   = read_csv("nursing_note.csv")       # 간호기록
df_surgery   = read_csv("surgery_anesthesia.csv") # 수술/마취


# ============================================================
# 셀 2
# ============================================================
# === 표준 이벤트 테이블 생성(표준화 + 가드레일 플래그, 경고/저장 이슈 해결판) ===
import os
import pandas as pd
import numpy as np
import re, json
from typing import List, Dict

pd.set_option("display.max_colwidth", 200)

# ---------------------------
# 0) 공통 유틸
# ---------------------------
def to_dt(s):
    """문자열 날짜 → pandas datetime. 변환 실패는 NaT."""
    return pd.to_datetime(s, errors="coerce")

# ---------------------------
# 1) admission 표준화 + encounter_id 만들기
# ---------------------------
adm = df_admission.rename(columns={
    "환자번호": "patient_id",
    "수진일자": "visit_date",
    "입원일자": "admit_date",
    "퇴원일자": "discharge_date",
    "입원시 진료과": "dept_admit",
    "수진 당시 나이": "age_at_visit",
    "성별": "sex",
    "수진(퇴원 포함) 진료과": "dept_visit",
}).copy()

adm["admit_date"]     = to_dt(adm["admit_date"])
adm["discharge_date"] = to_dt(adm["discharge_date"])
adm["visit_date"]     = to_dt(adm["visit_date"])
adm["patient_id"]     = adm["patient_id"].astype(str)

adm["encounter_id"] = adm["patient_id"] + "_" + adm["admit_date"].dt.strftime("%Y%m%d")

enc_core = adm[[
    "patient_id","encounter_id","admit_date","discharge_date",
    "dept_admit","dept_visit","age_at_visit","sex","visit_date"
]].drop_duplicates()

# ---------------------------
# 2) 소스별 → 표준 스키마
# ---------------------------
def std_from_cc(df):
    x = df.rename(columns={
        "환자번호": "patient_id",
        "수진일자": "visit_date",
        "작성일자": "event_datetime",
        "원내 CC명": "text",
    }).copy()
    x["visit_date"] = to_dt(x["visit_date"])
    x["event_datetime"] = to_dt(x["event_datetime"]).fillna(x["visit_date"])
    x["source_type"] = "chief_complaint"
    x["author_role"] = None; x["dept"] = None; x["code"] = None; x["display"] = None
    return x[["patient_id","visit_date","event_datetime","source_type","author_role","dept","text","code","display"]]

def std_from_dx(df):
    x = df.rename(columns={
        "환자번호": "patient_id",
        "수진일자": "visit_date",
        "진단일자": "event_datetime",
        "KCD 코드": "code",
        "KCD 한글명": "display",
    }).copy()
    x["visit_date"] = to_dt(x["visit_date"])
    x["event_datetime"] = to_dt(x["event_datetime"]).fillna(x["visit_date"])
    x["source_type"] = "dx"
    x["author_role"] = None; x["dept"] = None
    x["text"] = x["display"]
    return x[["patient_id","visit_date","event_datetime","source_type","author_role","dept","text","code","display"]]

def std_from_med(df):
    x = df.rename(columns={
        "환자번호": "patient_id",
        "수진일자": "visit_date",
        "서식 작성일자": "event_datetime",
        "서식명": "form_name",
        "서식 항목명": "item_name",
        "항목별 서식 값": "item_value",
        "서식 작성 진료과": "dept",
    }).copy()
    x["visit_date"] = to_dt(x["visit_date"])
    x["event_datetime"] = to_dt(x["event_datetime"]).fillna(x["visit_date"])
    x["text"] = (x["form_name"].fillna("") + " | " + x["item_name"].fillna("") + " : " + x["item_value"].fillna("")).str.strip(" |:")
    x["source_type"] = "medical_note"
    x["author_role"] = None; x["code"] = None; x["display"] = x["item_name"]
    return x[["patient_id","visit_date","event_datetime","source_type","author_role","dept","text","code","display"]]

def std_from_nurse(df):
    x = df.rename(columns={
        "환자번호": "patient_id",
        "수진일자": "visit_date",
        "간호기록 작성일자": "event_datetime",
        "항목명": "item_name",
        "항목값": "item_value",
    }).copy()
    x["visit_date"] = to_dt(x["visit_date"])
    x["event_datetime"] = to_dt(x["event_datetime"]).fillna(x["visit_date"])
    x["text"] = (x["item_name"].fillna("") + " : " + x["item_value"].fillna("")).str.strip(" :")
    x["source_type"] = "nursing_note"
    x["author_role"] = "nurse"; x["dept"] = None; x["code"] = None; x["display"] = x["item_name"]
    return x[["patient_id","visit_date","event_datetime","source_type","author_role","dept","text","code","display"]]

def std_from_surg(df):
    # 샘플에 없는 경우를 위한 안전 가드
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["patient_id","visit_date","event_datetime","source_type","author_role","dept","text","code","display"])
    x = df.rename(columns={
        "환자번호": "patient_id",
        "수술 일자": "event_datetime",
        "ICD9CM 코드": "code",
        "ICD9CM 명": "display",
        "[수술기록] 마취종류": "anesthesia",
        "[마취전 상태평가] ASA class": "asa",
        "[수술실 퇴실전] 출혈정도": "bleeding",
    }).copy()
    x["event_datetime"] = to_dt(x["event_datetime"])
    x["visit_date"] = x["event_datetime"].dt.normalize()
    def _mk_text(r):
        parts=[]
        if pd.notna(r.get("display")): parts.append(str(r["display"]))
        if pd.notna(r.get("anesthesia")): parts.append(f"마취:{r['anesthesia']}")
        if pd.notna(r.get("asa")): parts.append(f"ASA:{r['asa']}")
        if pd.notna(r.get("bleeding")): parts.append(f"출혈:{r['bleeding']}")
        return " | ".join(parts)
    x["text"] = x.apply(_mk_text, axis=1)
    x["source_type"] = "procedure"
    x["author_role"] = None; x["dept"] = None
    return x[["patient_id","visit_date","event_datetime","source_type","author_role","dept","text","code","display"]]

ev_cc, ev_dx = std_from_cc(df_cc), std_from_dx(df_diagnosis)
ev_med, ev_nurse, ev_surg = std_from_med(df_medical), std_from_nurse(df_nursing), std_from_surg(df_surgery)

# 빈 DF 제외, concat (FutureWarning 제거)
_ev_parts = [d for d in [ev_cc, ev_dx, ev_med, ev_nurse, ev_surg] if isinstance(d, pd.DataFrame) and not d.empty]
if _ev_parts:
    events = pd.concat(_ev_parts, ignore_index=True)
else:
    events = pd.DataFrame(columns=["patient_id","visit_date","event_datetime","source_type","author_role","dept","text","code","display"])

# 표준화: patient_id 문자열 통일 (encounter 매칭 안정)
events["patient_id"] = events["patient_id"].astype(str)

# ---------------------------
# 3) 이벤트 ↔ 입원(encounter) 매칭
# ---------------------------
def choose_enc(row, candidates):
    dt = row["event_datetime"]
    if pd.isna(dt) or candidates.empty:
        return pd.Series([np.nan, np.nan, np.nan])
    ok = candidates[(candidates["admit_date"] <= dt) & (dt <= candidates["discharge_date"])]
    if len(ok) > 0:
        best = ok.sort_values("admit_date").iloc[-1]
    else:
        tmp = candidates.copy()
        tmp["gap"] = (dt - tmp["admit_date"]).abs()
        best = tmp.sort_values("gap").iloc[0]
    return pd.Series([best["encounter_id"], best["admit_date"], best["discharge_date"]])

def assign_encounters_for_group(g: pd.DataFrame) -> pd.DataFrame:
    pid = g.name
    candidates = enc_core.loc[enc_core["patient_id"] == pid, ["encounter_id","admit_date","discharge_date"]]
    out = g.apply(lambda r: choose_enc(r, candidates), axis=1, result_type="expand")
    out.columns = ["encounter_id","admit_date","discharge_date"]
    return out

if not events.empty:
    events[["encounter_id","admit_date","discharge_date"]] = (
        events.groupby("patient_id", group_keys=False)
              .apply(assign_encounters_for_group, include_groups=False)
    )
else:
    events["encounter_id"] = ""
    events["admit_date"] = pd.NaT
    events["discharge_date"] = pd.NaT

# ---------------------------
# 4) 가드레일 플래그 1) 입원구간 벗어남
# ---------------------------
def flag_oor(r):
    dt, a, d = r["event_datetime"], r["admit_date"], r["discharge_date"]
    if pd.isna(dt) or pd.isna(a) or pd.isna(d): return False
    return not (a <= dt <= d)
events["out_of_range"] = events.apply(flag_oor, axis=1)

# ---------------------------
# 5) 수치/단위 추출 + 범위/단위 플래그
# ---------------------------
MEASURE_PAT = re.compile(r"(?P<name>[A-Za-z가-힣/]+)\s*[:=]?\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>%|mg/dL|mg/L|g/dL|mmHg|bpm|/µL|/uL|/L|U/L|µmol/L)?")
def extract_measures(text: str, topn: int = 5):
    res=[]
    if not isinstance(text, str): return res
    for m in MEASURE_PAT.finditer(text):
        name = m.group("name").strip()
        try: val = float(m.group("value"))
        except: continue
        unit = m.group("unit") if m.group("unit") else ""
        res.append({"name": name, "value": val, "unit": unit})
        if len(res) >= topn: break
    return res
events["measures"] = events["text"].apply(extract_measures)

REF_RANGE = {"Cr":{"unit":"mg/dL","low":0.6,"high":1.3},
             "CRP":{"unit":"mg/L","low":0.0,"high":5.0},
             "AST":{"unit":"U/L","low":0,"high":40},
             "ALT":{"unit":"U/L","low":0,"high":41}}
def flag_measures(ms: List[Dict]) -> Dict:
    flags=[]
    for m in ms:
        name_up = m["name"].upper().replace(" ","")
        key = next((k for k in REF_RANGE if k in name_up), None)
        if not key: continue
        rule = REF_RANGE[key]
        unit_ok = (m["unit"] in ["", rule["unit"]])
        if not unit_ok:
            flags.append(f"unit_suspect:{key}:{m['unit']}->expected:{rule['unit']}")
        v = m.get("value", None)
        if v is not None and (v < rule["low"] or v > rule["high"]):
            flags.append(f"out_of_ref:{key}:{v}{m['unit']}")
    return {"quality_flags": flags}
events["measure_flags"] = events["measures"].apply(flag_measures)

# ---------------------------
# 6) 같은 날 상태 요약(혼합 vs 충돌)
# ---------------------------
IMPROVE_PAT = re.compile(r"(호전|improv(ed|ement))", re.I)
WORSE_PAT   = re.compile(r"(ì•…í™"|worsen(ed|ing))", re.I)
def sentiment_tag(text: str) -> List[str]:
    if not isinstance(text, str): return []
    tags=[]
    if IMPROVE_PAT.search(text): tags.append("improved")
    if WORSE_PAT.search(text):   tags.append("worsened")
    return tags
events["tags"] = events["text"].apply(sentiment_tag)
events["event_date"] = events["event_datetime"].dt.date

SHORT_WINDOW_HOURS = 1.5
def summarize_day(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("event_datetime")
    seq=[]
    for _, r in g.iterrows():
        ts = r["tags"]
        if "improved" in ts: seq.append(("improved", r["event_datetime"], r["source_type"]))
        if "worsened" in ts: seq.append(("worsened", r["event_datetime"], r["source_type"]))
    if not seq:
        return pd.Series({"mixed_status_same_day": False, "conflict_same_day": False,
                          "day_first_tag": np.nan, "day_last_tag": np.nan, "day_span_hours": np.nan})
    first_tag, first_dt, _ = seq[0]
    last_tag,  last_dt,  _ = seq[-1]
    kinds = {s for s,_,_ in seq}
    span_hours = (last_dt - first_dt).total_seconds()/3600.0 if (pd.notna(first_dt) and pd.notna(last_dt)) else np.nan
    mixed = (kinds == {"improved","worsened"})
    had_proc = False
    if mixed and pd.notna(span_hours):
        mid_mask = (g["event_datetime"] >= first_dt) & (g["event_datetime"] <= last_dt)
        had_proc = any(g.loc[mid_mask, "source_type"].eq("procedure"))
    conflict = bool(mixed and (span_hours is not None) and (span_hours <= SHORT_WINDOW_HOURS) and (not had_proc))
    return pd.Series({"mixed_status_same_day": mixed, "conflict_same_day": conflict,
                      "day_first_tag": first_tag, "day_last_tag": last_tag, "day_span_hours": span_hours})

# 필요한 컬럼만 슬라이스해서 FutureWarning 제거 + 요약 병합
if not events.empty:
    daily_flags = (
        events[["encounter_id","event_date","event_datetime","source_type","tags"]]
          .groupby(["encounter_id","event_date"], as_index=False)
          .apply(summarize_day, include_groups=False)
          .reset_index(drop=True)
    )
else:
    daily_flags = pd.DataFrame(columns=["encounter_id","event_date","mixed_status_same_day",
                                        "conflict_same_day","day_first_tag","day_last_tag","day_span_hours"])

events = events.merge(daily_flags, on=["encounter_id","event_date"], how="left")

# 병합 후 안전 가드(비어 있을 때 컬럼 보장)
for c, default in [
    ("mixed_status_same_day", False),
    ("conflict_same_day", False),
    ("day_first_tag", ""),
    ("day_last_tag", ""),
    ("day_span_hours", np.nan),
]:
    if c not in events.columns:
        events[c] = default

# ---------------------------
# 7) encounter 내 순번 부여 → event_id 생성 → 저장 직전 dtype 정리
# ---------------------------
events = events.sort_values(["patient_id","encounter_id","event_datetime"], na_position="last").reset_index(drop=True)
events["row_no"]  = events.groupby(["patient_id","encounter_id"]).cumcount()+1
events["event_id"] = "e:" + events["encounter_id"].fillna("NA").astype(str) + ":" + events["row_no"].astype(str).str.zfill(4)

# --- Parquet 친화 타입 고정 ---
# 문자열 고정  (patient_id, encounter_id는 제외!)
to_string_cols = [
    "event_id","source_type","author_role","dept","text","code","display",
    "day_first_tag","day_last_tag"
]
for c in to_string_cols:
    if c in events.columns:
        events[c] = events[c].astype("string").fillna("")

# ID는 NaN 유지 + 문자열형만 강제 (fillna("") 금지)
if "patient_id" in events.columns:
    events["patient_id"] = events["patient_id"].astype("string")
if "encounter_id" in events.columns:
    events["encounter_id"] = events["encounter_id"].astype("string")


json_cols = ["measures","measure_flags","tags"]
for c in json_cols:
    if c in events.columns:
        def _dump(v):
            if isinstance(v, (list, dict)): return json.dumps(v, ensure_ascii=False)
            if pd.isna(v): return "[]" if c!="measure_flags" else "{}"
            return str(v)
        events[c] = events[c].apply(_dump)


for c in ["out_of_range","conflict_same_day","mixed_status_same_day"]:
    if c in events.columns:
        # 1) 먼저 pandas nullable boolean으로 변환
        events[c] = events[c].astype("boolean")
        # 2) 결측을 False로 채움
        events[c] = events[c].fillna(False)

for c in ["row_no","day_span_hours"]:
    if c in events.columns:
        events[c] = pd.to_numeric(events[c], errors="coerce")

# --- 저장 ---
os.makedirs("./events", exist_ok=True)
OUT_PATH = "./events/standardized_events.parquet"
events.to_parquet(OUT_PATH, index=False)


# ============================================================
# 셀 3
# ============================================================
# === RAG용 전체데이터 표준 이벤트 빌드 ===
import os

def _read_csv_data_dir(name):
    path = os.path.join("./data", name)
    return pd.read_csv(path, encoding="utf-8-sig")

try:
    df_adm_full      = _read_csv_data_dir("admission.csv")
    df_cc_full       = _read_csv_data_dir("chief_complaint.csv")
    df_dx_full       = _read_csv_data_dir("diagnosis.csv")
    df_med_full      = _read_csv_data_dir("medical_note.csv")
    df_nurse_full    = _read_csv_data_dir("nursing_note.csv")
    df_surg_full     = _read_csv_data_dir("surgery_anesthesia.csv")
except Exception as e:
    df_adm_full = None  # 전체데이터가 없으면 스킵
    print("./data 전체데이터를 찾지 못했음. (RAG는 샘플 파케로 동작) |", e)

if df_adm_full is not None:
    # --- admission 표준화 (셀2 로직과 동일) ---
    adm_full = df_adm_full.rename(columns={
        "환자번호": "patient_id",
        "수진일자": "visit_date",
        "입원일자": "admit_date",
        "퇴원일자": "discharge_date",
        "입원시 진료과": "dept_admit",
        "수진 당시 나이": "age_at_visit",
        "성별": "sex",
        "수진(퇴원 포함) 진료과": "dept_visit",
    }).copy()
    adm_full["admit_date"]     = to_dt(adm_full["admit_date"])
    adm_full["discharge_date"] = to_dt(adm_full["discharge_date"])
    adm_full["visit_date"]     = to_dt(adm_full["visit_date"])
    adm_full["encounter_id"] = adm_full["patient_id"].astype(str) + "_" + adm_full["admit_date"].dt.strftime("%Y%m%d")
    enc_core_full = adm_full[[
        "patient_id","encounter_id","admit_date","discharge_date",
        "dept_admit","dept_visit","age_at_visit","sex","visit_date"
    ]].drop_duplicates()

    # --- 소스 표준화(셀2 함수 재사용) ---
    ev_cc_full    = std_from_cc(df_cc_full)
    ev_dx_full    = std_from_dx(df_dx_full)
    ev_med_full   = std_from_med(df_med_full)
    ev_nurse_full = std_from_nurse(df_nurse_full)
    ev_surg_full  = std_from_surg(df_surg_full)

    events_full = pd.concat([ev_cc_full, ev_dx_full, ev_med_full, ev_nurse_full, ev_surg_full], ignore_index=True)

    # --- encounter 매칭 (셀2 로직과 동일) ---
    def _assign_encounters_full(g: pd.DataFrame) -> pd.DataFrame:
        pid = g.name
        candidates = enc_core_full.loc[enc_core_full["patient_id"] == pid, ["encounter_id","admit_date","discharge_date"]]
        out = g.apply(lambda r: choose_enc(r, candidates), axis=1, result_type="expand")
        out.columns = ["encounter_id","admit_date","discharge_date"]
        return out

    events_full[["encounter_id","admit_date","discharge_date"]] = (
        events_full.groupby("patient_id", group_keys=False)
                   .apply(_assign_encounters_full, include_groups=False)
    )

    # --- 가드/태깅(셀2 로직과 동일) ---
    events_full["out_of_range"] = events_full.apply(flag_oor, axis=1)
    events_full["measures"]      = events_full["text"].apply(extract_measures)
    events_full["measure_flags"] = events_full["measures"].apply(flag_measures)
    events_full["tags"]          = events_full["text"].apply(sentiment_tag)
    events_full["event_date"]    = events_full["event_datetime"].dt.date

    # --- 일일 요약(daily flags) (셀2와 동일) ---
    df_ok = events_full.dropna(subset=["event_datetime"]).copy()
    if not df_ok.empty:
        cols_for_day = ["encounter_id","event_date","event_datetime","source_type","tags"]
        daily_flags_full = (
            df_ok[cols_for_day]
              .groupby(["encounter_id","event_date"], as_index=False)
              .apply(summarize_day, include_groups=False)
              .reset_index(drop=True)
        )
        events_full = events_full.merge(daily_flags_full, on=["encounter_id","event_date"], how="left")
    else:
        events_full["mixed_status_same_day"] = False
        events_full["conflict_same_day"]     = False
        events_full["day_first_tag"]         = pd.NA
        events_full["day_last_tag"]          = pd.NA
        events_full["day_span_hours"]        = pd.NA

    # --- 정렬/ID/타입 고정 (셀2와 동일) ---
    events_full = events_full.sort_values(["patient_id","encounter_id","event_datetime"], na_position="last").reset_index(drop=True)
    events_full["row_no"]  = events_full.groupby(["patient_id","encounter_id"]).cumcount()+1
    events_full["event_id"] = "e:" + events_full["encounter_id"].fillna("NA").astype(str) + ":" + events_full["row_no"].astype(str).str.zfill(4)

    for c in ["event_id","patient_id","encounter_id","source_type","author_role","dept","text","code","display","day_first_tag","day_last_tag"]:
        if c in events_full.columns:
            events_full[c] = events_full[c].astype("string").fillna("")

    for c in ["measures","measure_flags","tags"]:
        if c in events_full.columns:
            def _dump(v):
                if isinstance(v, (list, dict)): return json.dumps(v, ensure_ascii=False)
                if pd.isna(v): return "[]" if c!="measure_flags" else "{}"
                return str(v)
            events_full[c] = events_full[c].apply(_dump)

    for c in ["out_of_range","conflict_same_day","mixed_status_same_day"]:
        if c in events_full.columns:
            events_full[c] = pd.Series(events_full[c], dtype="boolean").fillna(False).astype(bool)

    for c in ["row_no","day_span_hours"]:
        if c in events_full.columns:
            events_full[c] = pd.to_numeric(events_full[c], errors="coerce")

    os.makedirs("./events", exist_ok=True)
    OUT_PATH_FULL = "./events/standardized_events_full.parquet"
    events_full.to_parquet(OUT_PATH_FULL, index=False)
    print("(RAG용) 전체데이터 표준 이벤트 저장:", OUT_PATH_FULL, "| Encounter:", events_full["encounter_id"].nunique(), "| 이벤트:", len(events_full))
else:
    print("(RAG용) 전체데이터 파이프라인: 건너뜀")



# ============================================================
# 셀 4
# ============================================================
# === step 4: 인덱스 빌드 + 라이트검색(경량 RAG, 전 환자/전 encounter 대상으로) ===
import pandas as pd, numpy as np, json, re
from collections import defaultdict

EVENTS_PATH = "./events/standardized_events.parquet"
events = pd.read_parquet(EVENTS_PATH)

# --- JSON 문자열 -> Python 객체 (없으면 안전 기본값) ---
def _loads_maybe(s, default):
    if isinstance(s, str) and s.strip():
        try:
            return json.loads(s)
        except Exception:
            return default
    return default

for c, d in [("measures", []), ("measure_flags", {}), ("tags", [])]:
    if c in events.columns:
        events[c] = events[c].apply(lambda x: _loads_maybe(x, d))

# --- 섹션 라우팅 (필요시 조정) ---
SECTION_ROUTING = {
    "chief_complaint": ["chief_complaint","medical_note"],
    "major_diagnoses": ["dx","procedure"],
    "course_timeline": ["procedure","dx","medical_note","nursing_note","chief_complaint"],
}

def build_indices(enc_df: pd.DataFrame):
    enc_df = enc_df.copy()

    base_cols = [
        "event_id","event_datetime","source_type","text",
        "measures","out_of_range","mixed_status_same_day","conflict_same_day"
    ]
    # 타임라인(date → 이벤트 list)
    df_ok = enc_df.dropna(subset=["event_datetime"]).copy()
    if df_ok.empty:
        tl = {}
    else:
        cols = [c for c in base_cols if c in df_ok.columns]
        tl = (
            df_ok.sort_values("event_datetime")
                 .assign(_date=df_ok["event_datetime"].dt.date)
                 .groupby("_date", group_keys=False)[cols]
                 .apply(lambda g: g.to_dict("records"))
                 .to_dict()
        )

    # 섹션 후보뷰
    section_view = defaultdict(list)
    for r in enc_df.itertuples(index=False):
        st = getattr(r, "source_type", None)
        for sec, allowed in SECTION_ROUTING.items():
            if st in allowed:
                try:
                    section_view[sec].append(r._asdict())
                except AttributeError:
                    section_view[sec].append(dict(r._asdict()))

    return {
        "timeline": tl,
        "sections": dict(section_view),
        "all": enc_df.to_dict("records"),
    }

# encounter → index
enc_indices = {enc: build_indices(g.copy()) for enc, g in events.groupby("encounter_id", dropna=True)}
print(f"✅ 인덱스 빌드 완료 | encounters: {len(enc_indices)}")

# --- TF-IDF 활성화 확인 ---
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    USE_TFIDF = True
except Exception:
    USE_TFIDF = False

# --- 섹션 키워드 ---
SECTION_KEYWORDS = {
    "chief_complaint": {"흉통":1.0,"어지럼":0.8,"열":0.7,"설사":0.6,"pain":0.6,"fever":0.6},
    "major_diagnoses": {"진단":0.6,"KCD":0.9,"ICD":0.8,"폐렴":0.8,"저산소":0.6,"신부전":0.8},
    "course_timeline": {"수술":0.9,"시술":0.9,"검사":0.7,"impression":0.6,"plan":0.5,"투약":0.6},
}

# --- 과별 검색 가중치(경량 RAG 보정) ---
DEPT_KEYWORD_BOOSTS = {
    "신장내과": {
        "Cr": 1.5, "creatinine": 1.2, "eGFR": 1.4, "BUN": 1.2, "K": 1.0, "Na": 1.0,
        "요검사": 0.9, "proteinuria": 0.9, "albuminuria": 0.9, "신장": 0.6,
        "urinalysis": 0.9, "renal": 0.6,
    },
    "신경외과": {
        "laminectomy": 1.6, "IDEM": 1.5, "spine": 1.2, "척수": 1.2, "신경학": 1.0,
        "MRI": 1.4, "CT": 0.8, "dural": 0.8, "schwannoma": 1.2,
    },
    "순환기내과": {
        "ECG": 1.4, "ST elevation": 1.2, "troponin": 1.4, "BNP": 1.2, "CK-MB": 1.1,
        "Echo": 1.1, "CAG": 1.2, "PCI": 1.0, "angio": 1.0, "ischemia": 0.9,
    },
    "소화기내과": {
        "내시경": 1.4, "EGD": 1.2, "colonoscopy": 1.2, "biopsy": 1.0,
        "hepatic": 0.9, "AFP": 0.9, "PIVKA": 0.9, "담도": 0.9, "ERCP": 1.0,
    },
}

def _norm_dept(s: str) -> str:
    t = str(s).strip().lower()
    for ch in "()-_/[]{}|·•,":
        t = t.replace(ch, " ")
    return " ".join(t.split())

def canonicalize_dept_visit(dept_visit: str | None) -> str | None:
    if not dept_visit:
        return None
    t = _norm_dept(dept_visit)
    if "신경외과" in t:   return "신경외과"
    if "순환기내과" in t: return "순환기내과"
    if "소화기내과" in t: return "소화기내과"
    if "신장내과" in t:   return "신장내과"
    if "신경" in t:       return "신경외과"
    if "순환기" in t:     return "순환기내과"
    if "소화기" in t:     return "소화기내과"
    if "신장" in t:       return "신장내과"
    return None

DEPT_HINTS = {
    "신장내과": "병리검사 항목에 조직검사 결과를 구체적으로 작성.",
    "신경외과": "영상검사 항목에 수술 부위 관련 CT/MRI 소견 포함.",
    "순환기내과": "ECG/Echo/CAG/시술 결과를 빠짐없이 포함.",
    "소화기내과": "내시경 및 조직검사 소견을 영상/병리에 포함.",
}

def apply_dept_bias(base_kw: dict, dept: str | None) -> dict:
    if not dept or dept not in DEPT_KEYWORD_BOOSTS:
        return base_kw
    kw = dict(base_kw)
    for k, w in DEPT_KEYWORD_BOOSTS[dept].items():
        kw[k] = max(kw.get(k, 0.0), w)
    return kw

def _keyword_score(text: str, keywords: dict[str, float]) -> float:
    if not isinstance(text, str): 
        return 0.0
    t = text.lower()
    return sum(w for k, w in keywords.items() if k.lower() in t)

def lite_search(candidates: list[dict], query_keywords: dict[str, float], topk: int = 8):
    if not candidates:
        return []
    texts = [(c.get("text") or "") for c in candidates]
    kw_scores = [_keyword_score(tx, query_keywords) for tx in texts]

    tfidf_scores = [0.0]*len(candidates)
    if USE_TFIDF and len(texts) >= 2:
        vec = TfidfVectorizer(min_df=1, max_features=5000)
        X = vec.fit_transform(texts + [" ".join(query_keywords.keys())])
        sims = linear_kernel(X[-1], X[:-1]).ravel()
        tfidf_scores = sims.tolist()

    oor_pen = [0.0 if not c.get("out_of_range", False) else -0.25 for c in candidates]
    scores = [kw + 0.7*tf + pen for kw, tf, pen in zip(kw_scores, tfidf_scores, oor_pen)]
    order = np.argsort(scores)[::-1]

    picked, seen = [], set()
    for i in order:
        if len(picked) >= topk:
            break
        key = (str(candidates[i].get("event_datetime"))[:10], (candidates[i].get("text") or "")[:120])
        if key in seen:
            continue
        seen.add(key)
        picked.append(candidates[i])
    return picked

# --- 검사 카테고리 키워드 ---

TEST_CATEGORY_KEYWORDS = {
    "lab": {
        # 염증/혈액/간신장계
        "CRP": 1.0, "Procalcitonin": 0.9, "WBC": 0.8, "Neutrophil": 0.6,
        "AST": 0.7, "ALT": 0.7, "bilirubin": 0.6,
        "Cr": 1.0, "BUN": 0.9, "eGFR": 0.9, "Na": 0.8, "K": 0.8,
        # 심근표지자
        "Troponin": 1.0, "BNP": 0.9, "CK-MB": 0.8,
        
        "ABGA": 1.0, "pH": 0.6, "pCO2": 0.6, "pO2": 0.6, "HCO3": 0.6,
    },
    "imaging": {
        "X-ray":1.0, "CT":1.0, "MRI":1.0, "초음파":0.8, "내시경":0.7, "CAG":1.0, "angiography":0.8,
        "infiltrate":0.6, "consolidation":0.6, "lesion":0.6
    },
    "functional_or_procedure": {
        "ECG":1.0, "Echo":1.0, "PFT":1.0, "EEG":0.9, "PCI":1.0, "PTCA":1.0, "시술":0.8, "수술":0.8
    },
    "pathology": {
        "biopsy":1.0, "병리":1.0, "양성":0.8, "악성":0.8, "margin":0.8, "grade":0.8, "subtype":0.7
    },
}


def get_category_evidence(encounter_id: str, topk_per_cat: int = 3, sources=None) -> dict:
    idx = enc_indices.get(encounter_id)
    if not idx:
        return {"lab":[], "imaging":[], "functional_or_procedure":[], "pathology":[]}

    pool = idx.get("all", [])
    if sources:
        sset = set(sources)
        pool = [r for r in pool if r.get("source_type") in sset]

    out = {}
    for cat, kw in TEST_CATEGORY_KEYWORDS.items():
        picked = lite_search(pool, kw, topk=topk_per_cat)
        out[cat] = picked
    return out

# --- 전문과 힌트(요약용) ---
SPECIALTY_HINTS = {
    "cardio": {"trigger": ("Troponin","BNP","CAG","ST elevation","심초음파","ECG")},
    "nephro": {"trigger": ("Cr","eGFR","BUN","K","Na","요검사")},
    "pulmo":  {"trigger": ("ABGA","PFT","SpO2","저산소","폐렴","infiltrate")},
    "neurosurg":{"trigger": ("laminectomy","IDEM","spine","MRI","신경학적","근력")},
}

def guess_specialty(snippets: list[str]) -> str|None:
    t = " ".join([s.lower() for s in snippets if s]).strip()
    best, score = None, 0
    for sp, conf in SPECIALTY_HINTS.items():
        s = sum(1 for k in conf["trigger"] if k.lower() in t)
        if s > score:
            best, score = sp, s
    return best

# --- encounter → dept_visit 추출 (enc_core 사용) ---
def get_dept_visit(encounter_id: str) -> str | None:
    try:
        core = enc_core.loc[enc_core["encounter_id"] == encounter_id]
        if not core.empty and core["dept_visit"].notna().any():
            return str(core["dept_visit"].dropna().iloc[0])
    except NameError:
        pass
    # fallback: encounter 내 이벤트 dept 다수결
    df = events.loc[events["encounter_id"] == encounter_id, ["dept"]].dropna()
    if not df.empty:
        return df["dept"].astype(str).value_counts().idxmax()
    return None

# --- encounter 단위 evidence 수집 ---
def gather_evidence_for_encounter(encounter_id: str, topk_sec: int = 6, topk_cat: int = 3) -> dict:
    idx = enc_indices.get(encounter_id, {})
    if not idx:
        return {"encounter_id": encounter_id, "dept": None, "sections": {}, "tests": {}}

    dept_v = canonicalize_dept_visit(get_dept_visit(encounter_id))
    sections = {}
    for sec in ("chief_complaint","major_diagnoses","course_timeline"):
        cands = idx.get("sections", {}).get(sec, []) or idx.get("all", [])
        base_kw = SECTION_KEYWORDS.get(sec, {})
        biased_kw = apply_dept_bias(base_kw, dept_v)
        picked = lite_search(cands, biased_kw, topk=min(int(topk_sec), 6))
        sections[sec] = picked

    tests = get_category_evidence(encounter_id, topk_per_cat=topk_cat, sources=["dx","procedure","medical_note","nursing_note"])
    return {"encounter_id": encounter_id, "dept": dept_v, "sections": sections, "tests": tests}

# --- pid 단위 evidence 수집 (해당 pid의 모든 encounter) ---
def gather_evidence_for_pid(pid: str | int, topk_sec: int = 6, topk_cat: int = 3, max_encounters: int | None = None):
    pid_str = str(pid)
    # pid의 encounter 목록(최신순)
    encs = (
        enc_core.loc[enc_core["patient_id"].astype(str) == pid_str, ["encounter_id","admit_date"]]
        .sort_values("admit_date", ascending=False)
        .dropna(subset=["encounter_id"])
        ["encounter_id"]
        .tolist()
    )
    if max_encounters is not None:
        encs = encs[:max_encounters]

    out = []
    for enc in encs:
        out.append(gather_evidence_for_encounter(enc, topk_sec=topk_sec, topk_cat=topk_cat))
    return out

# === 모든 환자(pid) 대상: encounter별 경량 RAG 일괄 실행 ===
all_pids = sorted(enc_core["patient_id"].astype(str).unique())
ALL_EVIDENCE = {}  # {pid: [ {encounter_id, dept, sections{...}, tests{...}}, ... ]}

for pid in all_pids:
    ev_list = gather_evidence_for_pid(pid=pid, topk_sec=6, topk_cat=3, max_encounters=None)
    ALL_EVIDENCE[str(pid)] = ev_list

print("\n 전 환자/전 encounter 대상 경량 RAG 완료")



# ============================================================
# 셀 5
# ============================================================
# === Exaone 호출 어댑터 (경고/OOM 완화, CPU/4bit/토크나이저 가드) ===
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path("./models/EXAONE-3.5-7.8B-Instruct")

# ▼ 환경 스위치
USE_4BIT = False                 # 4bit 양자화 사용 시 True
PREFER_BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
DTYPE = torch.bfloat16 if PREFER_BF16 else torch.float16  # bf16 가능시 bf16, 아니면 fp16
MAX_INPUT_TOKENS = 4096          # 입력(프롬프트) 최대 토큰 (모델 템플릿 포함)
DEFAULT_MAX_NEW = 1500            # 생성 토큰 상한

_tokenizer = None
_model = None

def _safe_dtype():
    # CPU일 때는 float32로 강제 (fp16/bf16는 CPU에서 비효율/오류 가능)
    if not torch.cuda.is_available():
        return torch.float32
    return DTYPE

def _load_exaone():
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, use_fast=True, trust_remote_code=True
    )
    # pad 토큰 가드 (일부 토크나이저는 pad 토큰 미설정)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    common = dict(
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",   # flash-attn 없을 때 안정
    )

    if USE_4BIT:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_safe_dtype(),
            **common,
        )
    else:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            dtype=_safe_dtype(),
            **common,
        )

    _model.eval()
    return _tokenizer, _model

def _apply_chat_template(tokenizer, user_msg: str, system_msg: str|None=None) -> str:
    if getattr(tokenizer, "chat_template", None):
        msgs = []
        if system_msg:
            msgs.append({"role":"system","content":system_msg})
        msgs.append({"role":"user","content":user_msg})
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # 템플릿 없을 때 간단 결합
    return (system_msg + "\n\n" if system_msg else "") + user_msg

def _truncate_to_max(model_inputs, max_tokens: int, tokenizer):
    """입력 길이가 너무 길면 앞쪽(과거) 토큰을 잘라서 GPU OOM 방지."""
    ids = model_inputs["input_ids"][0]
    attn = model_inputs["attention_mask"][0]
    if ids.size(0) <= max_tokens:
        return model_inputs
    ids = ids[-max_tokens:]
    attn = attn[-max_tokens:]
    return {"input_ids": ids.unsqueeze(0).to(model_inputs["input_ids"].device),
            "attention_mask": attn.unsqueeze(0).to(model_inputs["attention_mask"].device)}

def exaone_generate(
    prompt: str,
    system: str | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW,
    do_sample: bool = False,
    temperature: float = 0.2,
    top_p: float = 0.9,
    stop_strings: list[str] | None = None,
    seed: int | None = None,
) -> str:
    tok, model = _load_exaone()
    text = _apply_chat_template(tok, user_msg=prompt, system_msg=system)

    # 시드 고정(옵션)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # 토크나이즈 + 길이 가드
    inputs = tok(text, return_tensors="pt", truncation=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    inputs = _truncate_to_max(inputs, MAX_INPUT_TOKENS, tok)

    # 생성 파라미터
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))

    # 생성
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    gen_text = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    # stop strings ì»·
    if stop_strings:
        cut_positions = [gen_text.find(s) for s in stop_strings if s in gen_text]
        if cut_positions:
            gen_text = gen_text[:min(cut_positions)]

    # 메모리 단속(긴 배치 반복 시)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return gen_text


# 전역 한 번만 출력 가드
PREVIEW_ONCE_SHOWN = False

def exaone_preview_input(prompt: str, system: str | None = None) -> str:
    tok, _ = _load_exaone()
    if getattr(tok, "chat_template", None):
        msgs = []
        if system:
            msgs.append({"role":"system","content":system})
        msgs.append({"role":"user","content":prompt})
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return (system + "\n\n" if system else "") + prompt



# ============================================================
# 셀 6
# ============================================================
# === RAG 동적 키워드 헬퍼 ===
def dyn_keywords_from_enc(idx, per_sec=8):
    """encounter의 전체 스니펫에서 TF-IDF 상위 키워드를 뽑아 약한 가중치로 섞어줌"""
    texts = [(r.get("text") or "") for r in (idx.get("all") or [])]
    texts = [t for t in texts if t.strip()]
    if len(texts) < 2:
        return {}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=2000, ngram_range=(1,2), min_df=1)
        X = vec.fit_transform(texts)
        mean_tfidf = X.mean(axis=0).A1
        feats = vec.get_feature_names_out()
        top = sorted(zip(feats, mean_tfidf), key=lambda x: x[1], reverse=True)[:per_sec]
        return {k: 0.4 for k,_ in top if len(k) >= 2}
    except Exception:
        # scikit-learn 미설치/에러 시 조용히 비활성화
        return {}



# ============================================================
# 셀 7
# ============================================================
# === 풀 컨텍스트 패키지 & 한 파일 프롬프트 ===
import textwrap

FORBIDDEN_PHRASES = ("확진","항상","반드시","절대","완치","100%","틀림없다")

# 과별 한 줄 힌트 (없으면 생략되도록)
DEPT_HINTS = globals().get("DEPT_HINTS", {
    "신장내과": "병리검사 항목에 조직검사 결과를 구체적으로 작성한다. 검체검사(특히 Cr/eGFR/BUN/K/Na)와 신장 생검 결과를 최우선으로 요약한다.",
    "신경외과": "영상검사 항목에 수술 부위 관련 CT/MRI 소견을 포함하고, 호르몬 연관 질환은 호르몬 수치/코일색전술 환자는 PRU 값 기재.",
    "순환기내과": "기능/시술 항목에 ECG, Echo, CAG, Peripheral angiography, T-SPECT 결과를 빠짐없이 포함한다.",
    "소화기내과": "내시경 및 조직검사 소견은 영상 또는 병리 항목에 포함한다.",
})

# 입원 경과 권장사항
def _hospital_course_line(dept: str | None) -> str:
    if not dept:
        return ""
    m = {
        "신경외과": "신경외과 체크: 수술 부위, 기구 사용 여부 등 구조적 정보를 명확히 포함, 신경학적 변화가 있다면 명확하게 기록, 종양 수술의 경우 수술 시 frozen biopsy 결과 기록",
        "소화기내과": "소화기내과 체크: 내시경/조직검사 소견 포함, 비정상 PE/Lab 강조",
    }
    return m.get(dept, "")

# 입원 결과 권장사항
def _discharge_dept_line(dept: str | None) -> str:
    if not dept:
        return ""
    m = {
        "신경외과": "신경외과 체크: Outcome, 잔여 신경학적 장애/증상, 그 원인, 치료 관련 합병증까지 명시.",
        "순환기내과": "순환기내과 체크: 퇴원 시 심기능 상태(Echo/심부전 증상), 시행 시술 결과(CAG/PCI 등), ECG/T-SPECT/지질 요약, 치료 중 합병증 포함.",
    }
    return m.get(dept, "")

def get_dept_visit(encounter_id: str) -> str | None:
    # enc_core ìš°ì" 
    try:
        core = enc_core.loc[enc_core["encounter_id"] == encounter_id]
        if not core.empty and core["dept_visit"].notna().any():
            return str(core["dept_visit"].dropna().iloc[0])
    except NameError:
        pass
    # fallback: encounter 내 이벤트 dept 다수결
    try:
        df = events.loc[events["encounter_id"] == encounter_id, ["dept"]].dropna()
        if not df.empty:
            return df["dept"].astype(str).value_counts().idxmax()
    except Exception:
        pass
    return None

# 토큰 예산 절약
_SNIP_LEN = 180
_MAX_SEC = 6
_MAX_CAT = 3

def build_context_package_full(encounter_id: str, topk: int = 6) -> dict:
    # enc_indices 안전 접근
    idx = enc_indices.get(encounter_id, {"sections": {}, "all": [], "timeline": {}})
    assert idx is not None, f"encounter {encounter_id} not found"

    # --- 유틸: 스니펫 길이 제한(토큰 예산 절약) ---
    def _shorten(txt: str, n=_SNIP_LEN):
        if not isinstance(txt, str):
            return ""
        return " ".join(txt.split())[:n]

    # --- 방문과(dept_visit) 추출 & 과별 힌트/부스트 준비 ---
    dept_visit_raw = get_dept_visit(encounter_id)        # 예: "신경외과(척추센터)"
    dept_visit     = canonicalize_dept_visit(dept_visit_raw) if 'canonicalize_dept_visit' in globals() else dept_visit_raw
    dept_hint_line = DEPT_HINTS.get(dept_visit or "", "")

    # --- 섹션 evidence (경량 검색 + 과별 키워드 부스트) ---
    ev_sec = {}

    # encounter 고유 동적 키워드 추출(헬퍼가 위 셀에 정의되어 있어야 함)
    dyn_kw = dyn_keywords_from_enc(idx)

    for sec in ("chief_complaint","major_diagnoses","course_timeline"):
        cands = idx.get("sections", {}).get(sec, []) or idx.get("all", [])

        base_kw = SECTION_KEYWORDS.get(sec, {}) if 'SECTION_KEYWORDS' in globals() else {}

        # 정적 + 동적 키워드 합치기 (동적은 약한 가중치로, 기존 값보다 크면 그 값 사용)
        mix_kw = dict(base_kw)
        for k, v in (dyn_kw or {}).items():
            mix_kw[k] = max(mix_kw.get(k, 0.0), v)

        # 과별 편향 적용은 섞은 키워드
        biased_kw = apply_dept_bias(mix_kw, dept_visit) if 'apply_dept_bias' in globals() else mix_kw

        # topk 클램프
        tk = max(1, min(int(topk), _MAX_SEC, len(cands))) if cands else 0

        # 1차 선택
        picked = lite_search(cands, biased_kw, topk=tk) if ('lite_search' in globals() and cands) else (cands[:tk] if tk else [])

        # 섹션별 fallback은 '루프 내부'에서 처리
        if not picked:
            base_all = idx.get("all", [])
            if base_all:
                base_all_sorted = sorted(base_all, key=lambda r: str(r.get("event_datetime") or ""), reverse=True)
                seen = set(); picked = []
                for r in base_all_sorted:
                    key = (str(r.get("event_datetime"))[:10], (r.get("text") or "")[:120])
                    if key in seen:
                        continue
                    seen.add(key)
                    picked.append(r)
                    # 2~3개 정도 최소 보장
                    if len(picked) >= max(2, min(int(topk)//2, 3)):
                        break

        # 스니펫 길이 제한
        for r in picked:
            r["text"] = _shorten(r.get("text", ""))

        ev_sec[sec] = picked

    # --- 카테 evidence (과별 재선택/재정렬 지원) ---
    ev_cat = {"lab": [], "imaging": [], "functional_or_procedure": [], "pathology": []}
    try:
        if callable(globals().get("get_category_evidence_deptaware")):
            base = get_category_evidence_deptaware(
                encounter_id,
                dept=dept_visit,
                topk_per_cat=min(_MAX_CAT, int(topk)),
                sources=["dx","procedure","medical_note","nursing_note"]
            ) or {}
        elif callable(globals().get("get_category_evidence")):
            base = get_category_evidence(
                encounter_id,
                topk_per_cat=min(_MAX_CAT, int(topk)),
                sources=["dx","procedure","medical_note","nursing_note"]
            ) or {}
        else:
            base = {}

        for cat in ev_cat.keys():
            arr = (base.get(cat) or [])[:_MAX_CAT]
            for r in arr:
                r["text"] = _shorten(r.get("text",""))
            ev_cat[cat] = arr
    except Exception:
        pass

    # --- 전문과 추정(있으면 사용) ---
    sp = None
    try:
        guess_sp = globals().get("guess_specialty")
        if callable(guess_sp):
            snips = []
            for arr in ev_sec.values(): snips += [(r.get("text") or "") for r in arr]
            for arr in ev_cat.values(): snips += [(r.get("text") or "") for r in arr]
            sp = guess_sp(snips)
    except Exception:
        sp = None

    # --- 입원 구간 ---
    try:
        enc_mask = events["encounter_id"].eq(encounter_id)
        admit = str(events.loc[enc_mask, "admit_date"].min())
        discharge = str(events.loc[enc_mask, "discharge_date"].max())
    except Exception:
        admit, discharge = "", ""

    return {
        "encounter_id": encounter_id,
        "admit": admit, "discharge": discharge,
        "evidence": {
            "chief_complaint": ev_sec.get("chief_complaint", []),
            "major_diagnoses": ev_sec.get("major_diagnoses", []),
            "course_timeline": ev_sec.get("course_timeline", []),
            "tests": ev_cat,
        },
        "specialty": sp,
        "dept_visit": dept_visit,
        "dept_hint_line": dept_hint_line,
    }

# === 과별 카테고리 evidence 선택기(경량 RAG용) ===
def get_category_evidence_deptaware(encounter_id: str, dept: str | None, topk_per_cat: int = 3, sources=None):
    base = get_category_evidence(encounter_id, topk_per_cat=topk_per_cat, sources=sources) if 'get_category_evidence' in globals() else {}
    KW = {
        "신장내과": {
            "lab": ("Cr","creatinine","eGFR","BUN","K","Na","요검사","proteinuria","albuminuria"),
            "imaging": ("renal","kidney","초음파","CT","MRI"),
            "functional_or_procedure": ("biopsy","생검"),
            "pathology": ("biopsy","glomerul","interstitial","IF","EM","병리"),
        },
        "신경외과": {
            "lab": (),
            "imaging": ("MRI","CT","spine","척수","IDEM","schwannoma"),
            "functional_or_procedure": ("laminectomy","수술","NRS","신경학"),
            "pathology": ("schwannoma","meningioma","병리"),
        },
        "순환기내과": {
            "lab": ("troponin","BNP","CK-MB"),
            "imaging": ("CAG","angiography","초음파","CT","MRI"),
            "functional_or_procedure": ("ECG","Echo","PCI","stent","ischemia","T-SPECT"),
            "pathology": (),
        },
        "소화기내과": {
            "lab": ("AST","ALT","bilirubin","amylase","lipase","AFP","PIVKA"),
            "imaging": ("CT","MRI","초음파","ERCP"),
            "functional_or_procedure": ("EGD","내시경","colonoscopy","ERCP"),
            "pathology": ("biopsy","조직","adenoma","carcinoma","병리"),
        },
    }
    def _score(txt, kws):
        if not isinstance(txt, str) or not txt: return 0.0
        t = txt.lower()
        return sum(1.0 for k in kws if k and k.lower() in t)

    if dept in KW and base:
        prefs = KW[dept]
        picked = {}
        for cat in ("lab","imaging","functional_or_procedure","pathology"):
            arr = base.get(cat, [])
            kws = prefs.get(cat, ())
            arr_scored = sorted(arr, key=lambda r: _score(r.get("text",""), kws), reverse=True)
            picked[cat] = arr_scored[:topk_per_cat]
        return picked
    return base or {"lab": [], "imaging": [], "functional_or_procedure": [], "pathology": []}

def render_prompt_freetext_full(pkg: dict) -> str:
    specialty = pkg.get("specialty")
    sp_str = specialty or "일반"
    cardio   = (specialty == "cardio")
    nephro   = (specialty == "nephro")
    pulmo    = (specialty == "pulmo")
    neurosurg= (specialty == "neurosurg")

    ev = pkg["evidence"]

    # Evidence가 없을 때는 빈 문자열(LLM에 '근거 없음' 문구가 들어가지 않게)
    def _fmt_evs(arr):
        if not arr: return ""
        lines=[]
        for r in arr[:_MAX_SEC]:
            dt = str(r.get("event_datetime",""))[:10]
            src= r.get("source_type","")
            txt= (r.get("text") or "").replace("\n"," ")
            lines.append(f"- {dt} [{src}] {txt[:_SNIP_LEN]}")
        return "\n".join(lines)

    def _fmt_tests(cat):
        arr = ev["tests"].get(cat, [])
        return _fmt_evs(arr)

    # 과별 권장 한 줄(있을 때만)
    course_hint = _hospital_course_line(pkg.get("dept_visit"))
    discharge_hint = _discharge_dept_line(pkg.get("dept_visit"))
    dept_hint_bullet = f"      4) {pkg.get('dept_hint_line')}" if pkg.get('dept_hint_line') else ""

    return textwrap.dedent(f"""
    [System]
    너는 분당서울대병원 퇴원요약 보조자다. 출력은 한국어 **Markdown 본문만** 작성하라.
    - 임의 추론 금지, 불명확하면 '미상'.
    - 과도한 확정 표현({', '.join(FORBIDDEN_PHRASES)}) 금지.
    - **환자 식별정보(나이/성별/ID 등)와 '기본정보(입·퇴원일/퇴원진단/수술명)'은 의료진이 직접 기입하므로 생성하지 않는다.**
    - '향후계획', '퇴원장소/결과' 등 의료진 별도 작성 항목은 **작성하지 않는다**.
    - 섹션 헤더는 ## 로, 하위 항목은 굵은 글씨 또는 불릿으로 간결히.
    - (전문화 힌트) 해당 케이스는 **{sp_str}** 추정. 검사결과 요약에서 **{ '심근표지자/ECG/Echo' if cardio else 'Cr/eGFR/K/Na' if nephro else 'ABGA/PFT/염증표지자' if pulmo else 'MRI/신경학적 소견' if neurosurg else '핵심 검사' }**를 우선 기술하라.
    - **Evidence에 없는 항목은 작성하지 말고 생략한다. 수치/날짜를 추측하거나 보정하지 않는다.**

    [Encounter Window – 참고만]
    - {pkg['admit']} ~ {pkg['discharge']}

    [Evidence (참고 전용, 인용문법 없이 내용만 활용)]
    ### chief_complaint
    {_fmt_evs(ev["chief_complaint"])}

    ### major_diagnoses
    {_fmt_evs(ev["major_diagnoses"])}

    ### course_timeline
    {_fmt_evs(ev["course_timeline"])}

    ### tests: laboratory
    {_fmt_tests("lab")}
    ### tests: imaging
    {_fmt_tests("imaging")}
    ### tests: functional_or_procedure
    {_fmt_tests("functional_or_procedure")}
    ### tests: pathology
    {_fmt_tests("pathology")}

    [Output]
    아래 **다섯 개 섹션**을 **이 순서로** 한 파일로 작성하라.
    (환자 식별/기본정보/향후계획/퇴원장소는 작성 금지)

    ## 입원사유 및 병력요약
    - **Chief Complaint(★)**: 한 문장
    - **Present Illness(★)**: 발현 시점·진행·동반증상 1–2문장
    - **Past History(★)**: 기저질환/수술력/복용약 1문장(없으면 '미상')
    - **Physical Exam/Assessment**: 관련 소견이 **확실히 있을 때만** 핵심만을 기술.

    ## 입원경과(★)
    - 날짜 중심 핵심 사건만 **시간순** 불릿 3–8개 나열:
      - [YYYY-MM-DD] 증상 변화 / 주요 검사 / 수술·시술 / 치료반응 / 합병증
      - 과별 체크(있으면 내용에만 반영, **'신경외과 체크' 등의 라벨 문구는 출력 금지**.):
      {('  - ' + course_hint) if course_hint else ''}

    ## 입원결과(★)
    - 입원 사유와 치료 결과 요약(1–2줄)
    - 퇴원 시 상태(증상/기능) 및 약물 변화
    - 과별 체크(있으면 내용에만 반영, **'신경외과 체크' 등의 라벨 문구는 출력 금지**.):
      {('- ' + discharge_hint) if discharge_hint else ''}

    ## 검사결과(★)
    - 작성 원칙:
      1) **진단·치료 결정에 영향 준 검사만 포함** (정상/불명확/미상은 생략하라).
      2) **[YYYY-MM-DD] 검사명: 결과** 한 줄 요약, 주요 수치는 **입원→퇴원 추세로 표시**. 예) CRP 12.4→4.8
      3) **같은 검사명이 24–48시간 내 동일하면 최신만 기재**, 이전은 "(지속)"으로 표기. 
        
        [변화 있는 예] ※ 아래는 실제 데이터가 아닌 형식 예시입니다
        - YYYY-MM-DD: 검사명 - 항목1 X.XXX, 항목2 XX.X
        - YYYY-MM-DD: 검사명 - 항목1 X.XXX, 항목2 XX.X (**호전**)
        
        [나쁜 예(금지)] ※ 아래는 실제 데이터가 아닌 형식 예시입니다
        - YYYY-MM-DD: 검사명 - 항목1 X.XXX, 항목2 XX.X
        - YYYY-MM-DD: 검사명 - 항목1 X.XXX, 항목2 XX.X **(지속)** ← 변화 없으면 중복 기재 금지

      {dept_hint_bullet}
    
    - 고정 순서(있을 때만 작성): ① 검체검사(Lab) → ② 영상검사(Imaging) → ③ 기능검사/시술 → ④ 병리(Pathology)
    - **수치/날짜 조작 금지**: Evidence에 값/날짜가 없으면 정량 문장 쓰지 말고, **핵심 소견 1줄**만.


    ## Patient Summary(★, 5–10문장)
    - 목적: 입원 전후 전체 흐름을 5–10문장으로 간결히 요약. 불명확하면 '미상'.
    - 포함 순서(반드시 지킬 것):
      1) 과거력: 주요 기저질환/수술력/복용약(모르면 '미상'). **나이/성별/ID는 작성 금지**.
      2) 입원 사유: 증상/발현 시점/악화·완화 요인(1문장).
      3) 진단 및 주요 검사 결과: 확정/의심 진단, 결정적 검사·영상·수치(필요시 추세).
      4) 치료 및 시술 내용: 시행 치료·수술/시술, 반응·합병증 요지.
      5) 퇴원 상태 및 향후 계획: 현재 상태(증상/기능), 착용/복약/주의, 검사·외래 추적(있으면).
    - 문체: 의무기록 요약체. 과도한 확정 표현(확진, 100% 등) 금지. 영문 병기 허용(X-ray, MRI 등).
    """).strip()



# ============================================================
# 셀 8
# ============================================================
# === 일괄 생성 러너(모든 encounter 대상, CSV 저장) ===
import os, time, datetime as dt
from pathlib import Path
import pandas as pd
from pathlib import Path
import datetime as dt

# 필수 가드: 필요한 함수/객체가 있는지 확인
for name in ["enc_indices", "build_context_package_full", "render_prompt_freetext_full", "exaone_generate"]:
    assert name in globals(), f"필요 함수/객체 누락: {name}. 앞 셀(3/R2/4)을 먼저 실행하세요."

OUT_DIR_FULL = Path("./events/outputs_final"); OUT_DIR_FULL.mkdir(parents=True, exist_ok=True)

def generate_one(encounter_id: str, topk: int = 6, max_new_tokens: int = 900):
    """단일 encounter 요약 생성 + 파일 저장 → (encounter_id, text, path)"""
    # 1) 컨텍스트/프롬프트 준비
    pkg = build_context_package_full(encounter_id, topk=topk)
    prompt = render_prompt_freetext_full(pkg)
    system_text = "심사위원이 바로 읽을 수 있는 한국어 Markdown으로만 출력."  # 먼저 정의

    import time
    secs = 0.0
    text = ""  # 혹시 상위 로직에서 미리 채워줄 때 대비
    if not text:
        t0 = time.time()
        text = exaone_generate(
            prompt,
            system=system_text,
            max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.6, top_p=0.9,
            stop_strings=["```","<eos>"],
        ).strip()
        secs = time.time() - t0
        
    
    # 모델 호출
    text = exaone_generate(
        prompt,
        system=system_text,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        stop_strings=["```","<eos>"],
    ).strip()

    if not text:
        text = exaone_generate(
            prompt,
            system=system_text,
            max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.6, top_p=0.9,
            stop_strings=["```","<eos>"],
        ).strip()

    # 파일 저장

    OUT_DIR_FULL = Path("./events/outputs_final"); OUT_DIR_FULL.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR_FULL / f"{encounter_id}__FULL_{ts}.md"
    out_path.write_text(text, encoding="utf-8")

    return encounter_id, text, str(out_path), secs


def run_batch_for_all_encounters(topk: int = 6, max_new_tokens: int = 1500, limit: int | None = None):
    assert isinstance(enc_indices, dict) and len(enc_indices) > 0, "enc_indices가 비어있음. 셀3 실행 확인."

    enc_list = sorted(enc_indices.keys())
    if limit is not None:
        enc_list = enc_list[:int(limit)]

    rows = []
    errs = 0
    total_model_sec = 0.0
    start_ts = globals().get("NOTEBOOK_START_TIME", dt.datetime.now().isoformat())  # ✅ 전역 변수 사용

    for i, enc in enumerate(enc_list, 1):
        try:
            eid, text, path, secs = generate_one(enc, topk=topk, max_new_tokens=max_new_tokens)
            total_model_sec += secs
            rows.append({"encounter_id": eid, "discharge_summary": text})
            print(f"[{i}/{len(enc_list)}] ok: {eid} (len={len(text)}) -> {path} | model_runtime_sec={secs:.2f}")
        except Exception as e:
            errs += 1
            print(f"[{i}/{len(enc_list)}] FAIL: {enc} | {e}")

    # 제출 CSV (멀티라인 안전)
    df_out = pd.DataFrame(rows, columns=["encounter_id", "discharge_summary"])
    df_out.to_csv("./result.csv", index=False, encoding="utf-8-sig", quoting=1)  # QUOTE_NONNUMERIC

    # runtime.txt
    end_time = dt.datetime.now()
    start_time = dt.datetime.fromisoformat(start_ts)
    total_seconds = (end_time - start_time).total_seconds()
  
    with open("./runtime.txt", "w", encoding="utf-8") as f:
        f.write(f"start_ts={start_ts}\n")
        f.write(f"end_ts={dt.datetime.now().isoformat()}\n")
        f.write(f"{total_seconds:.2f}\n")


    print("✅ 저장 완료: ./result.csv, ./runtime.txt")

    # S3 업로드
    import subprocess
    try:
        subprocess.run([
            "aws", "s3", "cp", "result.csv", 
            "s3://your-bucket/submissions/your-team/result.csv"
        ], check=True)
        subprocess.run([
            "aws", "s3", "cp", "runtime.txt",
            "s3://your-bucket/submissions/your-team/runtime.txt"
        ], check=True)
        print("S3 업로드 완료")
    except Exception as e:
        print(f"S3 업로드 실패: {e}")
    return df_out

    sz = os.path.getsize(out_csv) if os.path.exists(out_csv) else 0
    print(f"✅ 저장 완료: {out_csv} ({sz} bytes) / rows={len(df_out)}")
    if len(df_out) > 0:
        print(df_out.head(2))
    return df_out

# ===== 실행 =====
_ = run_batch_for_all_encounters(topk=6, max_new_tokens=1500, limit=None)
