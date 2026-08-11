import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.optimize import minimize
import sqlite3
import json
import os
import time
from datetime import datetime
from groq import Groq
import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import WorksheetNotFound

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

# ── 임시 비번 Google Sheets 기반 영구 저장 함수 ──────────────────────
# (기존 로컬 JSON 파일(temp_pwd_store.json) 방식은 Streamlit Cloud 재부팅 시
#  컨테이너 파일시스템이 초기화되면서 함께 사라지는 문제가 있어 Google Sheets로 이전)
_GSHEET_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
_TEMP_PWD_WORKSHEET = "EDNC-AI-A_temp_pwd"

def _get_temp_pwd_worksheet():
    """secrets에 등록된 서비스 계정으로 인증 후, 임시 비번 저장용 워크시트를 반환.
    시트가 없으면(최초 1회) 새로 만들고 헤더를 기록함."""
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]), scopes=_GSHEET_SCOPES
    )
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(st.secrets["temp_pwd_sheet_id"])
    try:
        ws = sh.worksheet(_TEMP_PWD_WORKSHEET)
    except WorksheetNotFound:
        ws = sh.add_worksheet(title=_TEMP_PWD_WORKSHEET, rows=200, cols=3)
        ws.append_row(["pwd", "expires", "created"])
    return ws


def _load_temp_pwds():
    """Google Sheets에서 임시 비번 목록 로드 — 앱 재부팅과 무관하게 유지됨"""
    try:
        ws = _get_temp_pwd_worksheet()
        records = ws.get_all_records()  # [{'pwd':..., 'expires':..., 'created':...}, ...]

        if not records:
            # 시트가 비어있으면(최초 실행) 기본 임시 비번(ednc1234, 7일 만료)으로 초기화
            from datetime import timedelta
            default_expires = datetime.now() + timedelta(days=7)
            default_created = datetime.now()
            _save_temp_pwds({"ednc1234": {"expires": default_expires, "created": default_created}})
            return {"ednc1234": {"expires": default_expires, "created": default_created}}

        result = {}
        for row in records:
            pwd = str(row.get("pwd", "")).strip()
            if not pwd:
                continue
            exp_raw = row.get("expires")
            cre_raw = row.get("created")
            result[pwd] = {
                "expires": datetime.fromisoformat(exp_raw) if exp_raw else None,
                "created": datetime.fromisoformat(cre_raw) if cre_raw else datetime.now()
            }
        return result
    except Exception:
        # Sheets 연결 일시 장애 시 세션에 남아있던 값이라도 유지 (완전 초기화 방지)
        return st.session_state.get('temp_pwd_list', {})


def _save_temp_pwds(pwd_dict):
    """임시 비번 목록을 Google Sheets에 저장.
    중요: 먼저 새 데이터를 기록해 성공을 확인한 뒤에만 남은 이전 행을 지웁니다.
    (기존 코드는 clear() 후 update()를 했기 때문에, update()가 실패하면
     시트가 빈 채로 남아 다음 로드 시 기본 비번으로 초기화되며 데이터가
     사라지는 문제가 있었습니다.)"""
    try:
        ws = _get_temp_pwd_worksheet()
        rows = [["pwd", "expires", "created"]]
        for pwd, info in pwd_dict.items():
            exp = info.get("expires")
            cre = info.get("created")
            rows.append([
                pwd,
                exp.isoformat() if isinstance(exp, datetime) else (exp if isinstance(exp, str) else ""),
                cre.isoformat() if isinstance(cre, datetime) else (cre if isinstance(cre, str) else str(datetime.now()))
            ])

        # values를 명시적 키워드로 전달 (gspread 버전 간 인자 순서 차이에 안전)
        ws.update(values=rows, range_name="A1")

        # 새 데이터가 이전 데이터보다 짧을 경우에 대비해, 기록 후 남는 잔여 행만 정리.
        # update()가 위에서 이미 성공했으므로 이 단계에서 실패해도 데이터 유실은 없음.
        existing_row_count = ws.row_count
        last_needed_row = len(rows)
        if existing_row_count > last_needed_row:
            ws.batch_clear([f"A{last_needed_row + 1}:C{existing_row_count}"])

        st.session_state['_temp_pwd_save_error'] = None
        return True
    except Exception as e:
        # 실패를 숨기지 않고 세션에 기록 — 사이드바에서 owner에게 표시됨
        st.session_state['_temp_pwd_save_error'] = str(e)
        return False

#  AI 리포트 핵심 조치 사항 개수 — 원하는 숫자로 변경하세요
NUM_ACTIONS = 3

#  핵심 조치 사항에 미리 포함할 내용 — 원하는 지시문을 자유롭게 입력하세요
# 사용하려면 앞의 # 을 제거하고, 빈 리스트([])로 두면 AI가 자동 생성합니다.
# 리포트는 항상 이 한국어 원문을 기준으로 1회 생성된 뒤 필요 시 영어로 번역되므로,
# 영어 버전을 별도로 관리할 필요가 없습니다.
PRESET_ACTIONS_KO = [
    "가장 불량 가능성이 높고, 50% 이상이고, 상위 3개의 조건을 모두 만족하는 불량에 대한 객관적 분석으로 해결책 도출하세요.",
    "가장 불량 가능성이 높고, 50%이상이고, 상위 3개의 3개 조건을 만족하는 불량들 간의 Trade Off에 대한 분석을 하세요.",
    "이러한 불량에 대한 유변학적이나 이론적 기술을 설명하세요."
]

import re

def _choose_regularization(n_pos, n_neg):
    """샘플이 극히 적은 타겟일수록 과적합 위험이 크므로 규제(C)를 더 강하게 줍니다.
    (알고리즘은 그대로 LogisticRegression을 쓰되, 규제 강도만 표본 수에 맞게 조정)"""
    min_class = min(n_pos, n_neg)
    if min_class < 5:
        return 0.1   # 표본이 매우 적음 → 강한 규제
    elif min_class < 15:
        return 0.5   # 표본이 다소 적음 → 중간 규제
    return 1.0       # sklearn 기본값


def _cross_val_reliability(X, y, n_pos, n_neg, C):
    """학습 정확도가 아니라 K-fold 교차검증 정확도로 모델 신뢰도를 측정합니다.
    표본이 너무 적어 유의미한 교차검증이 불가능하면 None(측정 불가)을 반환합니다."""
    min_class = min(n_pos, n_neg)
    if min_class < 2:
        return None  # 각 폴드에 최소 1개씩도 배정 못 함 → 교차검증 불가
    n_splits = min(5, min_class)
    try:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(
            LogisticRegression(max_iter=1000, C=C), X, y, cv=cv, scoring='accuracy'
        )
        return float(np.mean(scores))
    except Exception:
        return None


def _auto_select_best_model(X, y, n_pos, n_neg, C, algo_status_fn=None):
    """[추가] LR / RF / XGB / LGBM 4종을 교차검증 정확도로 비교해 가장 좋은 모델을 반환.
    표본이 너무 적거나 패키지가 없으면 가능한 후보 중 최선을 선택합니다.
    반환: (fitted_model, algo_name, cv_score, feature_importances_or_None)"""
    min_class = min(n_pos, n_neg)
    n_splits  = max(2, min(5, min_class))

    candidates = {
        'LogisticRegression': LogisticRegression(max_iter=1000, C=C),
        'RandomForest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    if _HAS_XGB:
        candidates['XGBoost'] = XGBClassifier(
            n_estimators=100, learning_rate=0.1, use_label_encoder=False,
            eval_metric='logloss', random_state=42, verbosity=0
        )
    if _HAS_LGBM:
        candidates['LightGBM'] = LGBMClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42,
            verbose=-1, n_jobs=-1
        )

    best_name  = 'LogisticRegression'
    best_score = -1.0
    best_model = candidates['LogisticRegression']

    if min_class >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        algo_list = list(candidates.items())
        for a_idx, (name, clf) in enumerate(algo_list):
            # [추가] 콜백으로 현재 시도 중인 알고리즘 표시
            if algo_status_fn:
                algo_status_fn(name, a_idx + 1, len(algo_list))
            try:
                scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
                mean_score = float(np.mean(scores))
                if mean_score > best_score:
                    best_score = mean_score
                    best_name  = name
                    best_model = clf
            except Exception:
                continue

    # 최종 전체 데이터로 재학습
    if algo_status_fn:
        algo_status_fn(f"Final fit: {best_name} ✓", len(candidates), len(candidates))
    best_model.fit(X, y)

    # Feature Importance 추출 (LR은 coef_ 사용)
    if hasattr(best_model, 'feature_importances_'):
        fi = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        fi = np.abs(best_model.coef_[0])
    else:
        fi = None

    cv_score = best_score if best_score >= 0 else None
    return best_model, best_name, cv_score, fi


def _fit_model_fixed(X, y, algo_status_fn=None):
    """LogisticRegression(max_iter=1000) 단일 모델로 고정 학습.
    (표본 수에 따른 규제 자동 조정이나 RF/XGB/LGBM 등 타 알고리즘과의 비교선택은 하지 않음.
    현장에서 검증된 방식)
    반환: (fitted_model, algo_name, feature_importances_or_None)"""
    if algo_status_fn:
        algo_status_fn("Final fit: LogisticRegression ✓", 1, 1)

    model = LogisticRegression(max_iter=1000).fit(X, y)
    fi = np.abs(model.coef_[0]) if hasattr(model, 'coef_') else None

    return model, 'LogisticRegression', fi


def _build_fact_block(defect_results):
    """[분석 결과]를 LLM이 스스로 판단하게 두지 않고, 파이썬이 이미 임계치(DEFECT_THRESHOLD)
    기준으로 고위험/저위험을 판정해서 '사실'로 정리해 넘깁니다. 모델은 이 판정을 그대로
    신뢰하기만 하면 되므로, 실측치와 무관하게 위험도를 과장/추측하는 할루시네이션을 줄입니다."""
    if not isinstance(defect_results, dict) or not defect_results:
        return (
            f"[분석 결과]: {defect_results}\n"
            "(진단/최적화 데이터가 없어 신뢰할 수 있는 분석을 수행할 수 없습니다. "
            "이 경우 절대 추측하지 말고 '데이터 부족으로 판단 불가'라고만 답하세요.)"
        )

    threshold_pct = int(DEFECT_THRESHOLD * 100)
    high_risk = {k: v for k, v in defect_results.items() if v >= DEFECT_THRESHOLD}
    low_risk = {k: v for k, v in defect_results.items() if v < DEFECT_THRESHOLD}

    def _fmt(d):
        return ", ".join(
            f"{TARGET_VARS.get(k, k)}: {v*100:.1f}%"
            for k, v in sorted(d.items(), key=lambda x: -x[1])
        )

    lines = [
        f"[사실 확인 - 아래는 파이썬이 이미 {threshold_pct}% 기준으로 판정을 마친 결과입니다. "
        f"이 판정을 그대로 신뢰하고, 절대 재해석하거나 다른 결론을 새로 만들지 마세요]"
    ]
    if high_risk:
        lines.append(f"- 위험(≥{threshold_pct}%) 불량 {len(high_risk)}개: {_fmt(high_risk)}")
    else:
        lines.append(
            f"- 위험(≥{threshold_pct}%) 불량: 없음 (전체 불량이 기준치 미만입니다. "
            "위험이 있는 것처럼 서술하거나 잠재적 위험을 추측하지 마세요.)"
        )
    if low_risk:
        lines.append(f"- 안전(<{threshold_pct}%) 불량 {len(low_risk)}개: {_fmt(low_risk)}")
    return "\n".join(lines)



def run_blocking_task(task_key, run_fn, running_msg, done_msg=None, trigger=False, show_spinner=True):
    """
    5번째 사진 스타일 모달 박스:
    - 진행 중: 회전 아이콘 + 파란 메시지 + 진행 바 + % (하나의 박스)
    - 완료 후: 녹색 체크 + 메시지 + 확인 버튼 (하나의 박스)
    - backdrop으로 배경 클릭 완전 차단
    """
    _is_en_task = st.session_state.get('lang', 'en') == 'en'
    if done_msg is None:
        done_msg = "Done! Click OK to view the result." if _is_en_task else "완료되었습니다! 확인을 누르면 결과가 표시됩니다."
    _state_key  = f"_modal_state_{task_key}"
    _result_key = f"_modal_result_{task_key}"
    st.session_state.setdefault(_state_key, "idle")
    if trigger:
        st.session_state[_state_key] = "running"
    _state = st.session_state[_state_key]
    if _state == "idle":
        return None
    if _state == "confirmed":
        st.session_state[_state_key] = "idle"
        return st.session_state.pop(_result_key, None)

    _uid = task_key.replace("-", "_").replace(".", "_")
    _BOX_W = 380

    # ── 공통 스타일 + backdrop ──────────────────────────────────
    st.markdown(f"""<style>
    /* backdrop: body::before로 구현해 별도 div 태그 불필요 */
    body::before {{
        content:'';
        position:fixed;top:0;left:0;width:100vw;height:100vh;
        background:rgba(4,9,18,0.88);z-index:99990;
        display:block;
    }}
    #mbox_{_uid} {{
        position:fixed;top:50%;left:50%;
        transform:translate(-50%,-50%);
        z-index:99995;
        background:#0d1525;
        border:1px solid #1e3a5f;
        border-radius:12px;
        box-shadow:0 14px 45px rgba(0,0,0,0.95);
        width:{_BOX_W}px;max-width:80vw;
        box-sizing:border-box;
        padding:24px 20px 34px 20px;
        text-align:center;
    }}
    .modal_icon_{_uid} {{
        font-size:1.4rem;
        margin-bottom:8px;
        display:block;
    }}
    .modal_msg_{_uid} {{
        font-weight:700;
        font-size:0.78rem;
        line-height:1.4;
        margin-bottom:2px;
    }}
    .modal_sub_{_uid} {{
        color:#64748b;
        font-size:0.6rem;
        margin-top:5px;
    }}
    .mprog_track_{_uid} {{
        width:100%;height:5px;
        background:#1e293b;
        border-radius:20px;
        overflow:hidden;
        margin:10px 0 5px 0;
    }}
    .mprog_fill_{_uid} {{
        height:100%;border-radius:20px;
        background:linear-gradient(90deg,#00e5ff,#10b981);
        transition:width 0.4s ease;
    }}
    .mpct_{_uid} {{
        color:#94a3b8;font-size:0.6rem;
        margin-bottom:0;
    }}
    /* 확인 버튼 위치 고정 */
    [data-modal-ok="{_uid}"] {{
        position:fixed !important;
        top:calc(50% + 44px) !important;
        left:50% !important;
        transform:translateX(-50%) !important;
        width:{_BOX_W - 40}px !important;
        max-width:calc(80vw - 40px) !important;
        z-index:99996 !important;
    }}
    [data-modal-ok="{_uid}"] button {{
        width:100% !important;
        padding:8px !important;
        font-size:0.75rem !important;
        font-weight:700 !important;
        border-radius:8px !important;
        background:linear-gradient(135deg,#10b981,#059669) !important;
        border:none !important;
    }}
    @keyframes mspin_{_uid} {{
        0%   {{ transform:rotate(0deg);   }}
        100% {{ transform:rotate(360deg); }}
    }}
    .mspinicon_{_uid} {{
        display:inline-block;
        animation:mspin_{_uid} 1.4s linear infinite;
    }}
    </style>
    """, unsafe_allow_html=True)

    if _state == "waiting_confirm":
        # ── 완료: 모달 박스 (아이콘 + 메시지) ──────────────────────
        _ok_label = "OK" if _is_en_task else "확인"
        _ok_key   = f"_modal_ok_{task_key}"

        # 모달 박스: 메시지만 (버튼 영역은 아래 여백으로 비워둠)
        st.markdown(f"""
        <div id="mbox_{_uid}">
            <div class="modal_msg_{_uid}" style="color:#10b981; margin-bottom:70px;">{done_msg}</div>
        </div>
        <style>
        /* Streamlit 확인 버튼을 모달 박스 하단 위에 완전히 겹치도록 고정 */
        div[data-testid="stButton"]:has(> button[kind="primary"]) {{
            position: fixed !important;
            top: calc(50% + 6px) !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
            width: {_BOX_W - 40}px !important;
            max-width: calc(80vw - 36px) !important;
            z-index: 99999 !important;
            pointer-events: all !important;
        }}
        div[data-testid="stButton"]:has(> button[kind="primary"]) > button {{
            width: 100% !important;
            padding: 9px 0 !important;
            font-size: 0.78rem !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
            background: linear-gradient(135deg, #10b981, #059669) !important;
            border: none !important;
            cursor: pointer !important;
        }}
        </style>
        """, unsafe_allow_html=True)
        _confirmed = st.button(_ok_label, type="primary",
                               use_container_width=True, key=_ok_key)
        if _confirmed:
            st.session_state[_state_key] = "confirmed"
            st.rerun()
        return None
    else:
        # ── 진행 중: 회전 아이콘 + 메시지 + 진행 바 ──────────────
        _pct_val = st.session_state.get(f"_modal_pct_{task_key}", 0)
        _sub_msg = st.session_state.get(f"_modal_sub_{task_key}",
                   "Please wait..." if _is_en_task else "잠시 기다려 주세요...")
        st.markdown(f"""
        <div id="mbox_{_uid}">
            <span class="modal_icon_{_uid}">
                <span class="mspinicon_{_uid}">🔄</span>
            </span>
            <div class="modal_msg_{_uid}" style="color:#38bdf8;">{running_msg}</div>
            <div class="mprog_track_{_uid}">
                <div class="mprog_fill_{_uid}" style="width:{_pct_val}%;"></div>
            </div>
            <div class="mpct_{_uid}">{_pct_val}%</div>
            <div class="modal_sub_{_uid}">{_sub_msg}</div>
        </div>
        """, unsafe_allow_html=True)
        _slot = st.empty()
        with _slot:
            if show_spinner:
                with st.spinner(" "):
                    _result = run_fn()
            else:
                _result = run_fn()
        st.session_state[_result_key] = _result
        st.session_state[_state_key] = "waiting_confirm"
        st.rerun()
        return None


def generate_ai_report(defect_results, optimized_params, num_actions=3, lang="ko", is_optimized=True):
    if not GROQ_API_KEY:
        return ("⚠️ GROQ_API_KEY가 설정되지 않았습니다. "
                "Streamlit Cloud의 App settings → Secrets에 GROQ_API_KEY를 등록한 뒤 다시 시도해주세요.")
    try:
        client = Groq(api_key=GROQ_API_KEY)

        # 사전 지정 조치 사항 (한국어 원문을 기준으로 사용)
        preset_text = ""
        if PRESET_ACTIONS_KO:
            preset_text = "\n\n[사전 지정 분석 항목 - 아래 항목들을 번호 순서대로 빠짐없이 먼저 분석하여 답변하세요]:\n"
            preset_text += "\n".join(f"{i+1}. {item}" for i, item in enumerate(PRESET_ACTIONS_KO))

        fact_block = _build_fact_block(defect_results)

        # [파라미터]가 최적화 결과인지, 아직 최적화하지 않은 현재 진단 조건인지 명시
        # (혼동 방지: 진단 조건을 최적화된 추천값처럼 서술하지 않도록)
        params_label = "[최적화된 추천 파라미터]" if is_optimized else "[현재 진단 조건 파라미터 - 아직 최적화되지 않은 값입니다]"
        params_note = (
            "" if is_optimized else
            "\n(주의: 아래 파라미터는 '조건 최적화'를 실행하기 전, 현재 설정된 진단 조건입니다. "
            "최적화된 추천값이 아니므로, 리포트에서 이를 '최적화 결과'나 '추천 조건'처럼 서술하지 말고 "
            "'현재 조건' 또는 '진단 조건'이라고 명확히 구분해서 표현하세요.)"
        )

        prompt_ko = f"""당신은 20년 경력의 사출 성형 공정 전문가입니다.

{fact_block}

{params_label}: {optimized_params}{params_note}{preset_text}

<중요 - 반드시 지켜야 할 원칙>
- 위 [사실 확인]에 없는 내용을 추측하거나 새로 만들어내지 마세요.
- 확실하지 않은 내용은 "데이터 부족으로 판단 불가"라고 답하세요.
- 본문에 등장하는 모든 수치(%, 값)는 위 [사실 확인]/{params_label}에 있는 값과 정확히 일치해야 하며,
  임의로 새로운 수치를 만들거나 반올림 외의 방식으로 바꾸지 마세요.
- 위험(≥{int(DEFECT_THRESHOLD*100)}%) 불량이 "없음"이라고 되어 있다면, 잠재적 위험이나 향후 가능성을
  추측해서 서술하지 말고 사실 그대로만 전달하세요.
- 사전 지정 분석 항목이 특정 조건(예: 위험도 {int(DEFECT_THRESHOLD*100)}% 이상인 불량의 존재)을
  전제로 한다면, 실제로 그런 불량이 없을 경우 "해당 조건을 만족하는 불량이 없습니다"라고 명확히
  밝히고 억지로 답을 만들어내지 마세요.

위 [사전 지정 분석 항목]이 있다면 그 항목들을 번호 순서대로 절대 빠짐없이 전부 분석하여 답변한 뒤,
마지막에 별도 섹션으로 "현장 작업자를 위한 핵심 조치 사항"을 정확히 {num_actions}개 작성하세요.
(사전 지정 분석 항목과 최종 핵심 조치 사항은 서로 다른 별개의 항목이므로, 핵심 조치 사항이 {num_actions}개라고 해서
사전 지정 분석 항목을 생략하거나 그 개수에 맞춰 줄이지 마세요.)
반드시 한국어로만 답하세요. 생각 과정(thinking)은 출력하지 말고 최종 답변만 작성하세요."""

        # ---------------------------------------------------------
        # [변경] 언어별로 매번 독립적으로 리포트를 생성하면, 같은 데이터를 넣어도
        # 모델이 호출마다 다시 추론하기 때문에 한국어/영어 리포트의 "결론 자체"가
        # 달라질 수 있습니다(실제 확인됨). 이를 막기 위해 항상 한국어로 리포트를
        # 1회만 생성한 뒤, 영어가 필요하면 그 결과를 그대로 "번역"만 하여 반환합니다.
        # → 두 언어의 수치·결론·구조가 항상 100% 일치합니다.
        # ---------------------------------------------------------
        response_ko = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 한국어로만 대화하는 사출 성형 전문가입니다. 반드시 한국어로만 답변하세요. "
                        "영어를 절대 사용하지 마세요. 제공된 데이터에 없는 사실을 추측하거나 지어내지 마세요."
                    )
                },
                {"role": "user", "content": prompt_ko}
            ],
            temperature=0,
        )
        report_ko = response_ko.choices[0].message.content

        if lang != "en":
            return report_ko

        translate_prompt = f"""다음은 사출 성형 공정 AI 분석 리포트(한국어 원문)입니다. 이 내용을 영어로 정확하게 번역하세요.
내용을 추가하거나 재해석하거나 결론을 바꾸지 마세요. 원문의 수치·사실·구조(표, 번호 목록, 소제목 등)를 그대로 유지한 채 번역만 하세요.

[한국어 원문]
{report_ko}"""

        response_en = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional technical translator. Translate the given Korean text "
                        "into English precisely and faithfully. Do not add, omit, reinterpret, or change "
                        "any facts, numbers, or conclusions. Preserve the original structure (tables, "
                        "numbered lists, headings) exactly. Respond only in English."
                    )
                },
                {"role": "user", "content": translate_prompt}
            ],
            temperature=0,
        )
        report_en = response_en.choices[0].message.content
        return report_en
    except Exception as e:
        err_prefix = "Report generation error: " if lang == "en" else "리포트 생성 오류: "
        return f"{err_prefix}{str(e)}"


# --- i18n 언어 사전 정의 ---
LANG_DICT = {
    "en": {
        "page_title": "Total Injection Defect AI Solution System",
        "access_title": "Injection Molding AI System Access",
        "enter_pwd": "Enter Password",
        "connect_sys": "Connect System",
        "invalid_pwd": "Invalid Password. Please try again.",
        "data_mgmt": "Data Management",
        "upload_1": "1. Current Optimal Conditions Data",
        "upload_2": "2. Historical Cumulative Data",
        "upload_3": "3. CAE Analysis Data",
        "run_ai": "Run AI Learning & Solution",
        "algo_mode_label": "4. AI Algorithm",
        "algo_mode_auto": "Intelligent Auto-Select (LR / RF / XGBoost / LightGBM comparison)",
        "algo_mode_light": "Lightweight Fixed Model (single LogisticRegression)",
        "algo_mode_help": "Auto-Select compares multiple algorithms via cross-validation and picks the best one per defect. Lightweight Fixed Model always uses a single LogisticRegression — often more stable on small datasets.",
        "algo_guide_title": "❔ Which one should I choose?",
        "algo_guide_auto": "Best for larger datasets (roughly 100+ rows) with complex, non-linear patterns. Compares 4 algorithms and automatically picks the most accurate one per defect.",
        "algo_guide_light": "Best for small datasets (roughly 50 rows or fewer). A single, stable LogisticRegression — lower overfitting risk.",
        "algo_reco_prefix": "Detected data:",
        "algo_reco_unit": " rows",
        "algo_reco_suffix": "Recommended:",
        "algo_badge_prefix": "Model used:",
        "algo_badge_auto": "Intelligent Auto-Select",
        "algo_badge_light": "Lightweight Fixed (IMS-TOTAL-Ver. 4)",
        "err_load": "Error loading file: ",
        "err_vars": "Could not find 10 defect variables in the uploaded data.",
        "warn_upload": "Please upload the Current Data (1) and either Historical (2) or CAE (3) data.",
        "main_title_1": "Total Injection ",
        "main_title_2": "",
        "main_title_3": "AI Solution System",
        "main_title_4": "",
        "main_desc_txt": "Comprehensive Defect Diagnostic & Multi-Objective Optimization System (10 Key Defects)",
        "main_desc": "Comprehensive Defect Diagnostic & Multi-Objective Optimization System v6.6 (10 Key Defects)",
        "m_status": "System Status",
        "m_vars": "Analyzed Variables",
        "m_reliability": "Expert Reliability",
        "m_opt": "Optimization Status / Algorithm",
        "status_active": "Operational",
        "status_standby": "Standby",
        "info_standby": "Please upload the converted data in the sidebar and start AI learning.",
        "tab_diag": "▶  Diagnostic & Optimization",
        "tab_master": "▶  Master Data & Analytics",
        "sec_a": "A. Current Conditions & Adjust",
        "btn_reset_initial": "↺ Reset to Initial Conditions",
        "lbl_initial_val": "Initial",
        "sec_b_expert": "B. Expert Recommended Condition Settings",
        "sec_c_result": "C. Optimized Process Conditions",
        "sec_d_diag": "C. Optimized Process Conditions",
        "sec_d_weight": "D. Defect Weights",
        "sec_e": "E. Feature Importance-based Diagnosis & AI Expert Report",
        "sec_c": "B. Defect Weights",
        "sec_c_sub2": "C. Expert Recommended Condition Settings",
        "lbl_constant": "Select Variables to Keep Constant",
        "lbl_target": " Target",
        "lbl_target_range": " Safe Range",
        "lbl_expert_rel": "Expert Guideline Reliability (%)",
        "sec_d": "D. Optimization & Intelligent Diagnosis",
        "btn_diagnose": "Diagnose Current Risk",
        "btn_optimize": "Optimize Conditions",
        "opt_converged": "Converged",
        "opt_failed": "Failed",
        "dash_title": "AI Intelligent Dashboard",
        "opt_success_msg": "AI Recommendation Derived Successfully",
        "diag_success_msg": "Diagnosis Complete (based on current, not-yet-optimized conditions)",
        "warn_need_diagnose": "⚠ To generate the AI Expert Report, please run 'Diagnose Current Risk' or 'Optimize Conditions' above first.",
        "btn_ai_report": "▸ Generate AI Expert Report",
        "report_box_title": "AI Expert Report",
        "spinner_analyzing": "Analyzing...",
        "btn_download": "Download Optimal Parameters (.csv)",
        "db_save_empty": "No data available to save. Please run AI Learning first.",
        "db_pc_download": "↓ Download Saved DB File to PC Directly",
        "db_export_title": "▤ External Database Export",
        "db_prepare_btn": "▸ Generate & Save DB Snapshot",
        "db_prepared_msg": "Prepared File: ",
        "db_current_latest": "✓ The file contains the latest data state.",
        "learning_progress": "Target Model Learning",
        "opt_progress": "Algorithm Search in Progress",
        "opt_step_local": "Fine-tuning in progress",
        "opt_step_global": "Global re-search in progress",
        "opt_current_risk": "Current Risk",
        "reliability_label": "Model Reliability (cross-validated)",
        "reliability_na": "Not measurable (too few samples)",
        "reliability_low_sample": "Low sample warning",
        "reliability_samples": "samples",
        "opt_step_label": "Step",
        "btn_feature_guide": "Generate Feature Importance-based Process Diagnosis Guide",
        "guide_title": "Process Improvement Guide (Result Diagnosis Report)",
        "guide_subtitle": "Optimization Result Diagnosis Report",
        "guide_pred_rel": "Prediction Reliability",
        "guide_unachievable": "Unachievable",
        "guide_out_of_spec": "Out of Spec",
        "guide_normal": "Normal Achievement",
        "guide_all_success": "All Targets Achieved Normally",
        "guide_success_msg": "Based on the analysis of the current input data, all valid quality targets are predicted to perfectly reach the target specification range you set. You can apply the currently derived process conditions directly to the field.",
        "guide_partial_msg": "Based on the analysis, some quality targets have a high risk of defect. Please review the recommended process conditions."
    },
    "ko": {
        "page_title": "통합 사출 불량 AI 솔루션 시스템",
        "access_title": "<span class='title-blue'>AI 머신 러닝</span><span class='title-white'>을 통한 </span><span class='title-blue'>사출</span><span class='title-white'> 공정 조건 최적화 시스템</span>",
        "enter_pwd": "비밀번호 입력",
        "connect_sys": "시스템 연결",
        "invalid_pwd": "비밀번호가 올바르지 않습니다. 다시 시도해 주세요.",
        "data_mgmt": "데이터 관리",
        "upload_1": "1. 현재 최적 조건 데이터",
        "upload_2": "2. 누적 이력 데이터",
        "upload_3": "3. CAE 해석 데이터",
        "run_ai": "학습 초기화 및 데이터 통합 학습 실행",
        "algo_mode_label": "4. AI학습 알고리즘 선택",
        "algo_mode_auto": "다중 모델 비교 선택",
        "algo_mode_light": "기준 모델 비교 선택",
        "algo_mode_help": "다중 모델 비교 선택은 여러 알고리즘을 교차검증으로 비교해 불량별로 가장 좋은 모델을 고릅니다. 기준 모델 비교 선택은 항상 단일 모델만 사용하며, 현장에서 검증된 방식과 동일합니다 — 표본이 적을 때 더 안정적일 수 있습니다.",
        "algo_guide_title": "AI 학습 알고리즘 선택 기준",
        "algo_guide_auto": "다중 모델 비교 : 데이터가 많고(대략 100건 이상) 조건별 변화 패턴이 복잡할 때 적합합니다. LogisticRegression, Random Forest, XGBoost, LightGBM 4가지 알고리즘을 교차검증(cross-validation)으로 비교해서, 불량 항목마다 가장 정확도가 높은 모델을 자동으로 찾아내는 최적화된 선택 방식입니다. 조건과 불량 사이의 복잡하고 비선형적인 관계까지 폭넓게 포착할 수 있어서, 데이터가 쌓일수록 예측 정확도를 최대한으로 끌어올릴 수 있습니다.",
        "algo_guide_light": "기준 모델 비교 : 데이터가 적을 때(대략 50건 이하) 적합합니다. LogisticRegression 단일 모델을 고정된 설정으로 학습해서, 속도가 빠르고 결과가 항상 안정적으로 재현됩니다. 모델 구조가 단순한 만큼 적은 데이터로도 과적합(overfitting) 위험이 낮고, 현장에서 이미 오랜 기간 검증된 신뢰도 높은 방식과 동일한 결과를 냅니다. 데이터가 아직 충분히 쌓이지 않은 초기 단계에서 특히 강점을 발휘합니다.",
        "algo_reco_prefix": "감지된 데이터",
        "algo_reco_unit": "건",
        "algo_reco_suffix": "추천:",
        "algo_badge_prefix": "적용 모델:",
        "algo_badge_auto": "다중 모델 비교 선택",
        "algo_badge_light": "기준 모델 비교 선택",
        "err_load": "파일 로드 오류: ",
        "err_vars": "업로드된 데이터에서 10대 불량 변수를 찾을 수 없습니다.",
        "warn_upload": "현재 데이터(1)와 함께 이력 데이터(2) 또는 CAE 데이터(3)를 업로드해 주세요.",
        "main_title_1": "AI 머신 러닝",
        "main_title_2": "을 통한 ",
        "main_title_3": "사출",
        "main_title_4": " 공정 조건 최적화 시스템",
        "main_desc_txt": "종합 불량 진단 및 다목적 최적화 시스템 (10대 핵심 불량)",
        "main_desc": "종합 불량 진단 및 다목적 최적화 시스템 v6.6 (10대 핵심 불량)",
        "m_status": "시스템 상태",
        "m_vars": "분석된 변수",
        "m_reliability": "전문가 신뢰도",
        "m_opt": "최적화 상태 / 사용 알고리즘",
        "status_active": "가동 중",
        "status_standby": "대기 중",
        "info_standby": "사이드바에 변환된 데이터를 업로드하고 AI 학습을 시작해 주세요.",
        "tab_diag": "▶  진단 및 최적화",
        "tab_master": "▶  마스터 데이터 & 분석",
        "sec_a": "A. 현재 사출 조건 & 조건 변경",
        "btn_reset_initial": "↺ 초기 조건으로 되돌리기",
        "lbl_initial_val": "초기값",
        "sec_b_expert": "B. 전문가 추천 조건 설정",
        "sec_c_result": "C. 최적 공정 조건",
        "sec_d_diag": "C. 최적 공정 조건",
        "sec_d_weight": "D. 불량 가중치",
        "sec_e": "E. Feature Importance 기반 공정 진단 가이드 & AI 전문가 진단",
        "sec_c": "B. 불량 가중치",
        "sec_c_sub2": "C. 전문가 추천 조건 설정",
        "lbl_constant": "고정 상태를 유지할 변수 선택",
        "lbl_target": " 목표치",
        "lbl_target_range": " 허용범위",
        "lbl_expert_rel": "전문가 가이드라인 신뢰도 (%)",
        "sec_d": "D. 최적화 및 지능형 진단",
        "btn_diagnose": "현재 리스크 진단",
        "btn_optimize": "역추론 최적화 탐색 실행",
        "opt_converged": "수렴 완료",
        "opt_failed": "최적화 실패",
        "dash_title": "AI 지능형 대시보드",
        "opt_success_msg": "AI 추천 조건 도출 완료",
        "diag_success_msg": "진단 완료 (아직 최적화되지 않은 현재 조건 기준)",
        "warn_need_diagnose": "⚠ AI 전문가 리포트를 생성하려면 먼저 위의 '현재 리스크 진단' 또는 '조건 최적화'를 실행해 주세요.",
        "btn_ai_report": "▸ AI 전문가 리포트 생성",
        "report_box_title": "AI 전문가 리포트",
        "spinner_analyzing": "분석 중...",
        "btn_download": "최적 파라미터 다운로드 (.csv)",
        "db_save_empty": "저장할 데이터가 없습니다. 먼저 데이터 업로드 후 AI 가동을 완료해 주세요.",
        "db_pc_download": "↓ 내보낸 DB 파일 PC로 직접 다운로드",
        "db_export_title": "▤ 데이터베이스 외부 내보내기",
        "db_prepare_btn": "▸ DB 스냅샷 생성 및 서버 저장",
        "db_prepared_msg": "준비된 파일: ",
        "db_current_latest": "✓ 최신 데이터 상태가 파일에 이미 반영되어 있습니다.",
        "learning_progress": "타겟 값 모델 학습 중",
        "opt_progress": "알고리즘 탐색 중",
        "opt_step_local": "미세 정밀 조정 중",
        "opt_step_global": "전역 재탐색 중",
        "opt_current_risk": "현재 위험도",
        "reliability_label": "모델 신뢰도(교차검증)",
        "reliability_na": "측정 불가(표본 부족)",
        "reliability_low_sample": "표본 부족 경고",
        "reliability_samples": "개 표본",
        "opt_step_label": "Step",
        "btn_feature_guide": "Feature Importance 기반 공정 진단 가이드 생성",
        "guide_title": "공정 개선 가이드 (결과 진단 리포트)",
        "guide_subtitle": "최적화 결과 진단 리포트",
        "guide_pred_rel": "예측 신뢰도",
        "guide_unachievable": "달성 불가",
        "guide_out_of_spec": "스펙 이탈",
        "guide_normal": "정상 달성",
        "guide_all_success": "전체 타겟 정상 달성",
        "guide_success_msg": "현재 입력된 데이터를 바탕으로 분석한 결과, 모든 유효 품질 타겟(타겟 값)이 설정하신 목표 스펙 범위 내에 완벽하게 도달할 수 있는 것으로 예측되었습니다. 현재 도출된 공정 조건(추천 공정 스펙)을 현장에 바로 적용하셔도 좋습니다.",
        "guide_partial_msg": "분석 결과 일부 품질 타겟에서 불량 위험도가 높게 나타났습니다. 도출된 공정 조건을 바탕으로 현장 적용 전 세밀한 검토가 필요합니다."
    }
}

# 0. 페이지 설정
st.set_page_config(layout="wide", page_title="Total Injection Defect AI Solution System")

if "lang" not in st.session_state:
    st.session_state.lang = "ko"

L = LANG_DICT[st.session_state.lang]

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "temp_pwd_list" not in st.session_state:
    st.session_state.temp_pwd_list = _load_temp_pwds()
if "is_owner" not in st.session_state:
    st.session_state.is_owner = False

if not st.session_state.authenticated:
    st.markdown("""
        <style>
        .stApp { background-color: #0b0c10 !important; color: #e1e1e1 !important; }
        h2 { color: #1e88e5 !important; font-weight: 800 !important; }
        .stTextInput label p { color: #FFFFFF !important; font-size: 1.1rem !important; font-weight: 600 !important; }
        .stButton>button {
            background: linear-gradient(180deg, #1e88e5 0%, #1565c0 100%) !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            border: 1px solid #1976d2 !important;
        }
        /* [추가] 언어 선택 드롭다운(KO) 크기/스타일 고정: 메인 화면 버튼과 완전히 동일한 값 사용 */
        .st-key-lang_sel_auth {
            width: 90px !important;
            max-width: 90px !important;
            flex: none !important;
        }
        .st-key-lang_sel_auth > div {
            width: 90px !important;
            max-width: 90px !important;
        }
        .st-key-lang_sel_auth [data-baseweb="select"] {
            width: 90px !important;
            max-width: 90px !important;
            flex: none !important;
        }
        .st-key-lang_sel_auth [data-baseweb="select"] > div {
            box-sizing: border-box !important;
            width: 90px !important;
            max-width: 90px !important;
            min-height: 30px !important;
            height: 30px !important;
            background: #262730 !important;
            border: 1px solid rgba(250,250,250,0.2) !important;
            border-radius: 8px !important;
            color: #fafafa !important;
            font-size: 0.78rem !important;
            flex: none !important;
            padding-left: 12px !important;
            padding-right: 8px !important;
        }
        .st-key-lang_sel_auth [data-baseweb="select"] svg {
            margin-left: 0 !important;
        }
        /* [추가] 타이틀 색상 지정용 클래스 (h2의 전역 파란색 규칙보다 우선 적용) */
        h2 .title-blue { color: #00e5ff !important; }
        h2 .title-white { color: #ffffff !important; }
        </style>
    """, unsafe_allow_html=True)

    col_space, col_lang = st.columns([9, 1])
    with col_lang:
        _lang_opt = ["KO", "EN"]
        _cur_lang = "KO" if st.session_state.lang == "ko" else "EN"
        _sel_lang = st.selectbox("🌐", _lang_opt,
            index=_lang_opt.index(_cur_lang),
            label_visibility="collapsed", key="lang_sel_auth")
        if (_sel_lang == "KO") != (st.session_state.lang == "ko"):
            st.session_state.lang = "ko" if _sel_lang == "KO" else "en"
            st.rerun()

    _, center, _ = st.columns([0.5, 2, 0.5])
    with center:
        # 로그인 화면 로고 — 왼쪽 정렬
        _login_logo = "<img src='data:image/png;base64,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' style='width:160px;object-fit:contain;'>"
        st.markdown('<div style="text-align:left;margin-bottom:12px;padding-top:8px;">' + _login_logo + '</div>', unsafe_allow_html=True)
        st.markdown(f"<br><h2 style='text-align: center; color:#FFFFFF; font-size: 1.7rem;'>{L['access_title']}</h2>", unsafe_allow_html=True)
        # ── 비밀번호 설정 ──────────────────────────────────
        OWNER_PWD  = "nt1234"          # 소유자 비번 (항상 유효)

        # 임시 비번은 Google Sheets에 영구 저장됨 (앱 재부팅과 무관하게 유지)
        # st.session_state.temp_pwd_list는 그 값을 캐싱해두는 용도
        if 'temp_pwd_list' not in st.session_state:
            st.session_state.temp_pwd_list = {}  # {비번: 만료일(datetime)}

        def _check_temp_pwd(p):
            """임시 비번 유효성 검사 — 파일에서 항상 최신 목록 확인"""
            # 파일에서 최신 목록 로드하여 세션 동기화
            _fresh = _load_temp_pwds()
            st.session_state.temp_pwd_list = _fresh
            info = _fresh.get(p)
            if info is None:
                return False
            if info['expires'] is None:  # 만료일 없음 = 무기한
                return True
            return datetime.now() < info['expires']

        _pwd_l, _pwd_mid, _pwd_r = st.columns([1, 4, 1])
        with _pwd_mid:
            col_pwd, col_btn = st.columns([4, 1], vertical_alignment="bottom")
            with col_pwd:
                pwd = st.text_input(L['enter_pwd'], type="password")
            with col_btn:
                _connect_clicked = st.button(L['connect_sys'], use_container_width=True)
        if _connect_clicked:
            if pwd == OWNER_PWD:
                st.session_state.authenticated = True
                st.session_state.is_owner = True
                st.rerun()
            elif _check_temp_pwd(pwd):
                st.session_state.authenticated = True
                st.session_state.is_owner = False
                st.session_state.logged_temp_pwd = pwd
                st.rerun()
            else:
                st.error(L['invalid_pwd'])


    st.stop()


# 10대 사출 불량 마스터 변수 정의
TARGET_VARS = {
    'Short_Shot': 'Short Shot (Short_Shot)',
    'Flash': 'Flash / Burr (Flash)',
    'Sink_Mark': 'Sink Mark / Shrinkage (Sink_Mark)',
    'Weld': 'Weld Line (Weld)',
    'Flow_Mark': 'Flow Mark (Flow_Mark)',
    'Silver_Streak': 'Silver Streak (Silver_Streak)',
    'Jetting': 'Jetting (Jetting)',
    'Burn_Mark': 'Burn Mark (Burn_Mark)',
    'Void': 'Void (Void)',
    'Warpage': 'Warpage (Warpage)'
}

# [추가] 공정변수 접두어별 설명 (예: SP_0 -> SP_0 (스크류 위치 0) / SP_0 (Screw Position 0))
_VAR_DESC_KO = {
    'SP': '스크류 위치', 'IV': '사출 속도', 'VP': 'VP', 'PP': '보압', 'PT': '보압 시간',
    'CT': '냉각 시간', 'NT': '노즐 온도', 'MTU': '금형 온도 상측', 'MTD': '금형 온도 하측',
    'MTUM': '금형 온도 상측 실측', 'MTDM': '금형 온도 하측 실측',
    'BP': '배압', 'BV': '배압 속도', 'SB': '흘름방지',
    'HP': '히트 파이프', 'HR': '핫 런너',
}
_VAR_DESC_EN = {
    'SP': 'Screw Position', 'IV': 'Injection Velocity', 'VP': 'VP', 'PP': 'Packing Pressure',
    'PT': 'Packing Time', 'CT': 'Cooling Time', 'NT': 'Nozzle Temp.', 'MTU': 'Mold Temp. Upper',
    'MTD': 'Mold Temp. Down', 'MTUM': 'Mold Temp Upper Measure', 'MTDM': 'Mold Temp Down Measure',
    'BP': 'Back Pressure', 'BV': 'Back Velocity', 'SB': 'Suck Back',
    'HP': 'Heat Pipe', 'HR': 'Hot Runner',
}

def _var_label(var_name, lang):
    """변수명을 'SP_0 (스크류 위치 0)' / 'SP_0 (Screw Position 0)' 형태로 변환.
    매핑에 없는 변수(예: HP_1)는 원래 이름 그대로 반환."""
    desc_map = _VAR_DESC_EN if lang == 'en' else _VAR_DESC_KO
    if var_name in desc_map:
        return f"{var_name} ({desc_map[var_name]})"
    if '_' in var_name:
        prefix, _, suffix = var_name.partition('_')
        if prefix in desc_map:
            return f"{var_name} ({desc_map[prefix]} {suffix})"
    return var_name

def _var_label_html(var_name, lang):
    """_var_label과 동일하지만, 괄호로 묶인 설명 부분만 2/3 크기로 렌더링하는 HTML 반환.
    (커스텀 라벨을 st.markdown으로 슬라이더 위에 그리고, st.slider 자체 라벨은 숨길 때 사용)"""
    desc_map = _VAR_DESC_EN if lang == 'en' else _VAR_DESC_KO
    desc = None
    if var_name in desc_map:
        desc = desc_map[var_name]
    elif '_' in var_name:
        prefix, _, suffix = var_name.partition('_')
        if prefix in desc_map:
            desc = f"{desc_map[prefix]} {suffix}"
    if desc is None:
        return var_name
    return (
        f"{var_name} "
        f"<span style='font-size:0.67em;'>({desc})</span>"
    )

OLD_TO_NEW_MAP = {
    'Y_Melt_Short': 'Short_Shot',
    'Y_Flash': 'Flash',
    'Y_Sink_Mark': 'Sink_Mark',
    'Y_Weld': 'Weld',
    'Y_Flow_Mark': 'Flow_Mark',
    'Y_Silver_Streak': 'Silver_Streak',
    'Y_Jetting': 'Jetting',
    'Y_Burn_Mark': 'Burn_Mark',
    'Y_Void': 'Void',
    'Y_Warpage': 'Warpage'
}

DEFECT_THRESHOLD = 0.5

# 세션 상태 초기화
if 'models' not in st.session_state:
    st.session_state.update({
        'models': {},
        'scalers': {},
        'model_reliability': {},
        'model_algo_names': {},      # [추가] 타겟별 선택된 알고리즘 이름
        'feature_importance': {},    # [추가] 타겟별 Feature Importance
        'df_injection': pd.DataFrame(),
        'ui_display_vars': [],
        'global_process_vars': [],
        'global_bounds': {},
        'expert_constraints': {},
        'current_inputs': {},
        'defect_weights': {k: 1.0 for k in TARGET_VARS.keys()},
        'defect_switches': {k: True for k in TARGET_VARS.keys()},
        'ver': 0,
        'initial_inputs': {},
        'expert_reliability': 0.0,
        'last_res_val': None,
        'last_defect_risks': {},
        'last_opt_df': None,
        'optimization_success': "N/A",
        'selected_algorithm': "N/A",
        'prepared_db_file': None,
        'data_changed_since_save': False,
        'show_feature_guide': False,
        'algo_summary': {}   # [추가] 학습 완료 후 알고리즘 선택 결과 요약
    })

# [수정] 슬라이더/숫자입력 위젯이 이미 이번 실행(run)에서 생성된 뒤에는
# st.session_state[key]를 직접 재할당할 수 없으므로(StreamlitAPIException),
# 최적화/재학습 직후에는 '대기용' 딕셔너리에만 값을 담아두고 st.rerun()으로 넘어가서,
# 위젯이 아직 생성되지 않은 이번 스크립트 최상단(여기)에서 안전하게 반영한다.
_pending_sync = st.session_state.pop('_pending_ni_sync', None)
if _pending_sync:
    for _pv, _pval in _pending_sync.items():
        _pb = st.session_state.get('global_bounds', {}).get(_pv, (0.0, 100.0))
        _pmin, _pmax = float(_pb[0]), float(_pb[1])
        if _pmin == _pmax:
            _pmax = _pmin + 1.0
        _pclamped = float(max(_pmin, min(float(_pval), _pmax)))
        st.session_state[f"ni_a_{_pv}"] = _pclamped
        st.session_state[f"sl_{_pv}_{st.session_state['ver']}"] = _pclamped

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Noto+Sans+KR:wght@300;400;700&display=swap');
    .stApp {
        background-color: #0b0c10 !important;
        color: #e1e1e1 !important;
        font-family: 'Inter', sans-serif;
    }
    /* ── 위젯 간격 전역 축소 ── */
    /* 슬라이더/체크박스 등 위젯 컨테이너 상하 패딩 제거 */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
        gap: 0rem !important;
    }
    div[class*="stSlider"] {
        padding-top: 0px !important;
        padding-bottom: 0px !important;
        margin-top: 0px !important;
        margin-bottom: 0px !important;
    }
    div[class*="stCheckbox"] {
        padding-top: 2px !important;
        padding-bottom: 0px !important;
        margin-bottom: 0px !important;
    }
    /* 불량 가중치 슬라이더만 label='' 타겟: weight_ key를 가진 슬라이더 */
    div[class*="stSlider"]:has(input[aria-label=""]) {
        margin-top: -28px !important;
    }
    /* expander 사이 간격 */
    div[data-testid="stExpander"] {
        margin-top: 2px !important;
        margin-bottom: 2px !important;
    }
    /* Streamlit 기본 block 간격 제거 */
    .element-container {
        margin-bottom: 0px !important;
    }
    /* hr 구분선 숨김 */
    hr { display: none !important; }
    [data-testid="stSidebar"] {
        background-color: #12141d !important;
        border-right: 1px solid #1f222e;
    }
    [data-testid="stSidebar"] label { color: #FFFFFF !important; font-weight: 400 !important; }
    .stSlider label,
    .stNumberInput label,
    [data-testid="stWidgetLabel"] p {
        color: #FFFFFF !important;
        font-weight: 400 !important;
        font-size: 1.05rem !important;
    }
    .metric-container {
        background-color: #1a1c24;
        border: 1px solid #2d3142;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .metric-label { color: #cbd5e1; font-size: 0.85rem; margin-bottom: 5px; }
    .metric-value { color: #ffffff; font-size: 1.2rem; font-weight: 700; }
    .section-title {
        display: flex;
        align-items: center;
        color: #00e5ff !important;
        font-weight: 600 !important;
        font-size: 1.3rem;
        margin-bottom: 0.3rem;
        margin-top: 0.8rem;
        padding-top: 0.8rem;
        border-top: 1px solid #2d3142;
    }
    .square-icon {
        width: 18px;
        height: 18px;
        background-color: #00e5ff;
        margin-right: 14px;
        display: inline-block;
        flex-shrink: 0;
    }
    .optimized-table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
        font-size: 0.9rem;
        background-color: #1a1c24;
        border-radius: 8px;
        overflow: hidden;
    }
    .optimized-table th {
        background-color: #2d3142;
        color: #FFFFFF !important;
        font-weight: 700;
        padding: 12px;
        border: 1px solid #3f445e;
    }
    .optimized-table td {
        color: #FFFFFF !important;
        padding: 12px;
        text-align: center;
        border: 1px solid #3f445e;
        font-weight: 500;
    }
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        background: linear-gradient(180deg, #10b981 0%, #059669 100%);
        color: white !important;
        font-weight: 700;
        border: 1px solid #047857;
        padding: 0.7rem;
        transition: all 0.3s ease;
    }
    /* [추가] 상단 언어 선택 드롭다운: 로그인 화면 드롭다운(KO)과 완전히 동일한 크기/스타일 */
    .st-key-lang_sel_main {
        width: 90px !important;
        max-width: 90px !important;
        flex: none !important;
    }
    .st-key-lang_sel_main > div {
        width: 90px !important;
        max-width: 90px !important;
    }
    .st-key-lang_sel_main [data-baseweb="select"] {
        width: 90px !important;
        max-width: 90px !important;
        flex: none !important;
    }
    .st-key-lang_sel_main [data-baseweb="select"] > div {
        box-sizing: border-box !important;
        width: 90px !important;
        max-width: 90px !important;
        min-height: 30px !important;
        height: 30px !important;
        background: #262730 !important;
        border: 1px solid rgba(250,250,250,0.2) !important;
        border-radius: 8px !important;
        color: #fafafa !important;
        font-size: 0.78rem !important;
        flex: none !important;
        padding-left: 12px !important;
        padding-right: 8px !important;
    }
    .st-key-lang_sel_main [data-baseweb="select"] svg {
        margin-left: 0 !important;
    }
    .stDownloadButton>button {
        background: linear-gradient(180deg, #2e7d32 0%, #1b5e20 100%) !important;
        border: 1px solid #2e7d32 !important;
    }
    h1 { color: #ffffff !important; font-weight: 800 !important; letter-spacing: -0.04em; }
    /* [추가] 타이틀 색상 지정용 클래스 (span 인라인 style이 걸러지는 문제를 피하기 위해
       클래스+스타일블록 방식으로 대체) */
    .title-blue { color: #00e5ff !important; }
    .title-white { color: #ffffff !important; }
    /* 탭 opacity 전역 차단 — Streamlit 기본 테마가 비활성 탭을 흐리게 하는 것 방지 */
    [data-baseweb="tab"] { opacity: 1 !important; }
    [data-baseweb="tab-highlight"] { background-color: #00e5ff !important; }
    .custom-progress-container {
        width: 100%;
        background-color: #1f222e;
        border-radius: 20px;
        margin: 10px 0;
        height: 22px;
        position: relative;
        overflow: hidden;
        border: 1px solid #2d3142;
    }
    .custom-progress-bar {
        height: 100%;
        border-radius: 20px;
        transition: width 0.8s ease-in-out;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 800;
        font-size: 0.85rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
        min-width: 2.5rem;
    }
    div[data-testid="stCheckbox"] label p {
        color: #00e5ff !important;
        font-weight: 600 !important;
    }
    /* Tab 버튼 시인성 개선 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        border-bottom: 2px solid #2d3142;
        padding-bottom: 0;
    }
    /* 비활성 탭 — 가능한 모든 선택자 병기 */
    .stTabs [data-baseweb="tab"],
    .stTabs button[data-baseweb="tab"],
    .stTabs [role="tab"],
    .stTabs button[role="tab"] {
        background-color: #252840 !important;
        border: 1px solid #4a5070 !important;
        border-bottom: none !important;
        border-radius: 8px 8px 0 0 !important;
        color: #f8fafc !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 10px 24px !important;
        opacity: 1 !important;
        transition: all 0.2s ease;
    }
    /* 탭 내부 모든 텍스트 요소 강제 적용 */
    .stTabs [data-baseweb="tab"] *,
    .stTabs [role="tab"] * {
        color: #f8fafc !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }
    /* 활성 탭 */
    .stTabs [aria-selected="true"],
    .stTabs button[aria-selected="true"] {
        background-color: #00e5ff22 !important;
        border-color: #00e5ff !important;
        color: #00e5ff !important;
        opacity: 1 !important;
    }
    .stTabs [aria-selected="true"] *,
    .stTabs button[aria-selected="true"] * {
        color: #00e5ff !important;
        opacity: 1 !important;
    }
    /* hover */
    .stTabs [data-baseweb="tab"]:hover,
    .stTabs button[data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background-color: #2e3455 !important;
        opacity: 1 !important;
    }
    .stTabs [data-baseweb="tab"]:hover *,
    .stTabs button[data-baseweb="tab"]:hover * {
        color: #ffffff !important;
        opacity: 1 !important;
    }
    /* ── Expander 전체 스타일 ── */
    /* 컨테이너 */
    [data-testid="stExpander"] {
        border: 1px solid #2d3142 !important;
        border-radius: 8px !important;
        background-color: #1a1c24 !important;
        margin-bottom: 2px !important;
        margin-top: 2px !important;
        overflow: hidden !important;
    }
    /* details 태그 자체 (접힘/펼침 공통) */
    [data-testid="stExpander"] details {
        background-color: #1a1c24 !important;
        border: none !important;
    }
    /* 펼침 상태 details[open] */
    [data-testid="stExpander"] details[open] {
        background-color: #1a1c24 !important;
    }
    /* 헤더 summary (접힌 상태) */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] details summary {
        background-color: #1a1c24 !important;
        border: none !important;
        border-radius: 8px !important;
        color: #e1e1e1 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 12px 16px !important;
        transition: background 0.2s ease, color 0.2s ease !important;
        list-style: none !important;
    }
    /* 펼침 상태 헤더 */
    [data-testid="stExpander"] details[open] summary,
    .streamlit-expanderHeader[aria-expanded="true"] {
        background-color: #1e2235 !important;
        color: #00e5ff !important;
        border-radius: 8px 8px 0 0 !important;
        border-bottom: 1px solid #2d3142 !important;
    }
    /* hover */
    .streamlit-expanderHeader:hover,
    [data-testid="stExpander"] details summary:hover {
        background-color: #23263a !important;
        color: #00e5ff !important;
    }
    /* 포커스 빨간 테두리 완전 제거 */
    .streamlit-expanderHeader:focus,
    .streamlit-expanderHeader:focus-visible,
    [data-testid="stExpander"] *:focus,
    [data-testid="stExpander"] *:focus-visible,
    [data-testid="stExpander"] details:focus,
    [data-testid="stExpander"] summary:focus,
    [data-testid="stExpander"] summary:focus-visible {
        outline: none !important;
        box-shadow: none !important;
        border-color: #2d3142 !important;
    }
    /* 컨텐츠 영역 */
    .streamlit-expanderContent,
    [data-testid="stExpander"] details > div {
        border: none !important;
        border-top: 1px solid #2d3142 !important;
        border-radius: 0 0 8px 8px !important;
        background-color: #12141d !important;
        padding: 12px 4px !important;
    }
    /* 화살표(chevron) 색상 */
    .streamlit-expanderHeader svg,
    [data-testid="stExpander"] summary svg {
        color: #00e5ff !important;
        fill: #00e5ff !important;
        stroke: #00e5ff !important;
    }
    /* 로딩 스텝 박스 */
    .load-step {
        background: #1a1c24;
        border: 1px solid #2d3142;
        border-left: 3px solid #00e5ff;
        border-radius: 6px;
        padding: 6px 12px;
        margin: 4px 0;
        font-size: 0.82rem;
        color: #cbd5e1;
    }
    .load-step.done { border-left-color: #10b981; color: #10b981; }
    .load-step.active { border-left-color: #00e5ff; color: #e1e1e1; }
    </style>
""", unsafe_allow_html=True)


# =================================================================
# 1. 사이드바 (데이터 관리)
# =================================================================
with st.sidebar:
    st.markdown(f"<h2 style='color:#FFFFFF; font-size:1.5rem;'>{L['data_mgmt']}</h2>", unsafe_allow_html=True)

    # ── 임시 비번 로그인 시 유효기간 표시 ────────────────────────
    if not st.session_state.get('is_owner', False):
        _is_ko_exp = st.session_state.lang != 'en'
        _logged_pwd = st.session_state.get('logged_temp_pwd', '')
        _tp_info = st.session_state.temp_pwd_list.get(_logged_pwd, {})
        _exp_dt = _tp_info.get('expires')
        if _exp_dt is None:
            _exp_msg   = "🟢 " + ("무기한 사용 가능" if _is_ko_exp else "No Expiry")
            _exp_color = "#10b981"
        else:
            _remain  = _exp_dt - datetime.now()
            _total_h = int(_remain.total_seconds() // 3600)
            _days_r  = _total_h // 24
            _hrs_r   = _total_h % 24
            if _remain.total_seconds() > 0:
                if _days_r > 0:
                    _time_str = f"{_days_r}일 {_hrs_r}시간" if _is_ko_exp else f"{_days_r}d {_hrs_r}h"
                else:
                    _time_str = f"{_hrs_r}시간" if _is_ko_exp else f"{_hrs_r}h"
                _icon      = "🟡" if _total_h < 24 else "🟢"
                _exp_color = "#ffab00" if _total_h < 24 else "#10b981"
                _exp_msg   = _icon + " " + ("잔여 사용기간: " if _is_ko_exp else "Expires in: ") + _time_str
            else:
                _exp_msg   = "🔴 " + ("사용 기간 만료" if _is_ko_exp else "Access Expired")
                _exp_color = "#ff5252"
        st.markdown(
            f"<div style='background:#0d1525;border:1px solid #1e3a5f;border-radius:8px;"
            f"padding:8px 12px;margin-bottom:10px;font-size:0.82rem;"
            f"color:{_exp_color};font-weight:600;'>{_exp_msg}</div>",
            unsafe_allow_html=True
        )

    u1 = st.sidebar.file_uploader(L['upload_1'], type=['csv', 'xlsx'], key='u1_uploader')
    u2 = st.sidebar.file_uploader(L['upload_2'], type=['csv', 'xlsx', 'db'], key='u2_uploader')
    u3 = st.sidebar.file_uploader(L['upload_3'], type=['csv', 'xlsx'], key='u3_uploader')

    # [추가] 업로드된 데이터 건수를 가볍게 미리 훑어서 알고리즘 추천 문구 표시
    def _quick_row_count(f):
        """실제 학습 전, 미리보기용으로만 행 수를 셈. 실패해도 조용히 None 반환."""
        if not f:
            return None
        try:
            f.seek(0)
            if f.name.endswith('.db'):
                return None  # DB 파일은 미리보기 생략
            elif f.name.endswith('csv'):
                _df_prev = pd.read_csv(f)
            else:
                _df_prev = pd.read_excel(f)
            f.seek(0)  # 실제 학습 시 다시 읽을 수 있도록 포인터 원위치
            return len(_df_prev)
        except Exception:
            try:
                f.seek(0)
            except Exception:
                pass
            return None

    _row_counts = [c for c in (_quick_row_count(u1), _quick_row_count(u2), _quick_row_count(u3)) if c is not None]
    _total_rows = sum(_row_counts) if _row_counts else None

    if _total_rows is not None:
        if _total_rows <= 50:
            _reco_icon, _reco_label = "🟢", L['algo_mode_light']
        else:
            _reco_icon, _reco_label = "🔵", L['algo_mode_auto']
        st.sidebar.markdown(
            f"<div style='background:#0d1525;border:1px solid #1e3a5f;border-radius:8px;"
            f"padding:6px 10px;margin:6px 0 2px 0;font-size:0.76rem;color:#cbd5e1;'>"
            f"{_reco_icon} {L['algo_reco_prefix']} <b>{_total_rows}{L['algo_reco_unit']}</b> → "
            f"{L['algo_reco_suffix']} <b style='color:#00e5ff;'>{_reco_label}</b></div>",
            unsafe_allow_html=True
        )

    with st.sidebar:
        # [추가] 라디오 옵션 글자색을 아래 가이드 점 색상과 맞춤
        # (순서: 1번째=기준 모델 비교 선택=녹색, 2번째=다중 모델 비교 선택=파란색)
        st.markdown("""
            <style>
            div[data-testid="stRadio"] div[role="radiogroup"] label:nth-of-type(1) p {
                color: #10b981 !important;
            }
            div[data-testid="stRadio"] div[role="radiogroup"] label:nth-of-type(2) p {
                color: #3b82f6 !important;
            }
            </style>
        """, unsafe_allow_html=True)
        algo_mode_choice = st.radio(
            L['algo_mode_label'],
            options=['light', 'auto'],
            format_func=lambda x: L['algo_mode_auto'] if x == 'auto' else L['algo_mode_light'],
            index=0,  # 기본값: 기준 모델 비교 선택
            help=L['algo_mode_help'],
            key='algo_mode_radio'
        )
        with st.expander(L['algo_guide_title'], expanded=False):
            st.markdown(f"""
                <div style='font-size:0.78rem; line-height:1.55;'>
                    <div style='color:#cbd5e1; margin-bottom:10px;'>🟢 {L['algo_guide_light']}</div>
                    <div style='color:#cbd5e1;'>🔵 {L['algo_guide_auto']}</div>
                </div>
            """, unsafe_allow_html=True)
        sub_btn = st.button(L['run_ai'], key='run_ai_btn', use_container_width=True)

    if sub_btn:
        def load_data(f):
            if not f:
                return None
            try:
                if f.name.endswith('.db'):
                    temp_db = "temp_uploaded.db"
                    with open(temp_db, "wb") as t:
                        t.write(f.getvalue())
                    conn = sqlite3.connect(temp_db)
                    df_temp = pd.read_sql_query("SELECT vars FROM production_log", conn)
                    conn.close()
                    if os.path.exists(temp_db):
                        os.remove(temp_db)
                    df_res = pd.json_normalize([json.loads(x) for x in df_temp['vars']])
                    if 'vars' in df_res.columns:
                        df_res = df_res.drop(columns=['vars'], errors='ignore')
                    return df_res
                elif f.name.endswith('csv'):
                    return pd.read_csv(f)
                else:
                    return pd.read_excel(f)
            except Exception as e:
                st.sidebar.error(f"{L['err_load']}{e}")
                return None

        # 데이터 로딩/학습 진행 상황 → 화면 중앙 모달로 표시
        _train_modal_slot = st.empty()

        def _show_train_modal(pct, msg, detail=""):
            _train_modal_slot.markdown(f"""
            <style>
            /* backdrop: body::before로 대체 */
            #train_modal_box {{
                position:fixed;top:50%;left:50%;
                transform:translate(-50%,-50%);
                z-index:99999;background:#0d1525;
                border:1px solid #1e3a5f;border-radius:12px;
                box-shadow:0 12px 40px rgba(0,0,0,0.9);
                width:250px;max-width:80vw;
                box-sizing:border-box;
                padding:20px 20px 18px 20px;
                text-align:center;pointer-events:all;
            }}
            .train_prog_track {{
                width:100%;height:5px;background:#1e293b;
                border-radius:20px;overflow:hidden;margin:8px 0 5px 0;
            }}
            .train_prog_fill {{
                height:100%;border-radius:20px;
                background:linear-gradient(90deg,#00e5ff,#10b981);
                transition:width 0.4s ease;
            }}
            @keyframes train_modal_spin {{
                0%   {{ transform:rotate(0deg);   }}
                100% {{ transform:rotate(360deg); }}
            }}
            .train_modal_spin_icon {{
                display:inline-block;
                animation:train_modal_spin 0.9s linear infinite;
            }}
            </style>
            <div id="train_modal_box">
                <div style="font-size:1.0rem;margin-bottom:7px;">
                    <span class="train_modal_spin_icon">🔄</span>
                </div>
                <div style="font-weight:700;color:#00e5ff;font-size:0.7rem;margin-bottom:4px;">
                    {msg}
                </div>
                <div class="train_prog_track">
                    <div class="train_prog_fill" style="width:{pct}%;"></div>
                </div>
                <div style="color:#94a3b8;font-size:0.62rem;margin-bottom:4px;">{pct}%</div>
                <div style="color:#64748b;font-size:0.6rem;">{detail}</div>
            </div>
            """, unsafe_allow_html=True)

        # 사이드바 진행바도 유지 (보조)
        load_prog    = st.sidebar.progress(0)
        load_status  = st.sidebar.empty()
        load_detail  = st.sidebar.empty()

        def _step(pct, msg, detail="", done_steps=None):
            load_prog.progress(pct, text=f"{pct}%")
            load_status.markdown(
                f"<div style='background:#1a1c24;border:1px solid #2d3142;border-left:3px solid #00e5ff;"
                f"border-radius:
