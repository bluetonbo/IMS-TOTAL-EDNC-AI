import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from scipy.optimize import minimize
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
import io
import sqlite3
import json
import os
import re
from groq import Groq
from datetime import datetime
import time

def _r2_score(y_true, y_pred):
    """sklearn.metrics 없이 R² 계산 (Python 3.14 segfault 회피)"""
    ya = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    ss_tot = float(np.sum((ya - ya.mean()) ** 2))
    ss_res = float(np.sum((ya - yp) ** 2))
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

def auto_select_best_model(X, y):
    """
    데이터 특성을 분석하여 회귀 문제에 최적인 알고리즘을 자동 선택합니다.
    - LinearRegression / Ridge: 소규모·선형 데이터
    - RandomForest / ExtraTrees: 중규모·비선형 데이터
    - GradientBoosting / XGBoost: 대규모·복잡한 패턴
    교차검증(CV) R² 기준으로 최고 성능 모델을 반환합니다.
    """
    n_samples, n_features = X.shape
    cv_folds = min(3, max(2, n_samples // 5))

    candidates = {}

    # 항상 포함: 선형 모델 (기준선)
    candidates['Linear'] = LinearRegression()
    candidates['Ridge'] = Ridge(alpha=1.0)

    # 샘플 수에 관계없이 항상 트리 모델 포함 (소규모 데이터 대응)
    n_est = min(100, max(10, n_samples // 3))
    candidates['RandomForest'] = RandomForestRegressor(
        n_estimators=n_est, random_state=42, n_jobs=-1
    )
    candidates['ExtraTrees'] = ExtraTreesRegressor(
        n_estimators=n_est, random_state=42, n_jobs=-1
    )
    candidates['GradientBoosting'] = GradientBoostingRegressor(
        n_estimators=n_est, random_state=42
    )
    if XGBOOST_AVAILABLE:
        candidates['XGBoost'] = XGBRegressor(
            n_estimators=n_est, random_state=42,
            verbosity=0, eval_metric='rmse'
        )

    best_name  = 'Linear'
    best_score = -999
    best_model = LinearRegression()

    for name, clf in candidates.items():
        try:
            scores = cross_val_score(
                clf, X, y, cv=cv_folds, scoring='r2', error_score=-999
            )
            mean_score = float(scores.mean())
            if mean_score > best_score:
                best_score = mean_score
                best_name  = name
                best_model = clf
        except Exception:
            continue

    return best_model, best_name, round(best_score, 3)


if 'df_caulking' not in st.session_state: st.session_state['df_caulking'] = pd.DataFrame()
if 'scaler' not in st.session_state: st.session_state['scaler'] = None
if 'opt_result_x' not in st.session_state: st.session_state['opt_result_x'] = None
if 'sim_result_x' not in st.session_state: st.session_state['sim_result_x'] = None
if 'sim_confidence' not in st.session_state: st.session_state['sim_confidence'] = 0.0
if 'feature_importance' not in st.session_state: st.session_state['feature_importance'] = {}
if 'algo_summary' not in st.session_state: st.session_state['algo_summary'] = {}

# 1. 페이지 설정
st.set_page_config(
    layout="wide", 
    page_title="JOINT AI - Process Optimization Suite",
)

# 2. 콘솔 스타일 CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        background-color: #090d16 !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif;
    }

    /* 메인 컨텐츠 영역이 사이드바를 제외한 전체 폭을 항상 사용하도록 강제 */
    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 100% !important;
        width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0f1524 !important;
        border-right: 1px solid #1e293b;
        min-width: 360px !important;
    }
    
    .scrollable-box {
        max-height: 400px;
        overflow-y: auto;
        padding: 15px;
        background-color: #0f1524;
        border: 1px solid #223154;
        border-radius: 6px;
        color: #e2e8f0;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
        color: #f1f5f9 !important;
    }
    
    .glass-card {
        background: #131b2e;
        border: 1px solid #223154;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    
    .glass-card-title {
        color: #38bdf8;
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #1e293b;
    }

    .stButton>button, .stDownloadButton>button {
        height: 2.8rem !important;
        font-size: 0.9rem !important;
        border-radius: 4px !important;
        background: #10b981 !important;
        color: #ffffff !important;
        font-weight: 600;
        border: none !important;
        transition: all 0.2s ease;
        width: 100%;
    }

    /* 0. 설명/보조 문구 가독성 보정 (너무 어둡지 않게, 과하지 않게) */
    label, .stTextInput label, .stSelectbox label, .stSlider label,
    .stNumberInput label, .stRadio label, .stFileUploader label,
    [data-testid="stWidgetLabel"] p {
        color: #b6c2d9 !important;
    }
    ::placeholder {
        color: #8291ab !important;
        opacity: 1 !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #a9b6cc !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #a9b6cc !important;
    }

    /* 진행바 문구 및 info/warning/error/success 알림 문구 밝기 보정 (사이드바 포함) */
    [data-testid="stProgress"] p,
    [data-testid="stProgress"] span,
    [data-testid="stProgress"] div,
    [data-testid="stProgress"] *,
    [data-testid="stSidebar"] [data-testid="stProgress"] * {
        color: #e7ecf6 !important;
    }
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] div,
    [data-testid="stAlert"] *,
    [data-testid="stSidebar"] [data-testid="stAlert"] *,
    [data-testid="stAlertContentInfo"] p,
    [data-testid="stAlertContentWarning"] p,
    [data-testid="stAlertContentError"] p,
    [data-testid="stAlertContentSuccess"] p {
        color: #eef2f9 !important;
    }
    [data-testid="stSidebar"] h3 {
        color: #00e5ff !important;
    }

    /* 파일 업로더 드롭존 배경을 다크 테마 색상으로 강제 지정 (라이트 배경 방지) */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #131b2e !important;
        border: 1px solid #223154 !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        height: 2.1rem !important;
        min-height: unset !important;
        padding: 0 14px !important;
        font-size: 0.8rem !important;
        width: auto !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] p,
    [data-testid="stFileUploaderDropzoneInstructions"] svg {
        color: #a9b6cc !important;
        fill: #a9b6cc !important;
    }
    [data-testid="stFileUploaderFile"] {
        background-color: #131b2e !important;
        border: 1px solid #223154 !important;
        border-radius: 6px !important;
    }

    /* 2. 회색조 보조 문구 전반 시인성 개선 */
    [data-testid="stFileUploaderFileName"],
    [data-testid="stFileUploaderFile"] span,
    [data-testid="stFileUploaderFile"] small,
    [data-testid="stFileUploaderFileErrorMessage"] {
        color: #d3dbec !important;
    }
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] svg {
        color: #dbe3f2 !important;
    }
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] {
        color: #b6c2d9 !important;
    }
    .stTabs [aria-selected="true"] p {
        color: #ffffff !important;
    }
    [data-testid="stMarkdownContainer"] small,
    .stMarkdown small {
        color: #b6c2d9 !important;
    }

    /* ── 탭 시인성 ── */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #1e3a5f; gap: 8px; }
    .stTabs [data-baseweb="tab"], .stTabs button[data-baseweb="tab"], .stTabs [role="tab"] {
        background-color: #0d1b2e !important; border: 1px solid #1e3a5f !important;
        border-bottom: none !important; border-radius: 8px 8px 0 0 !important;
        color: #e2e8f0 !important; font-weight: 700 !important; opacity: 1 !important;
        padding: 10px 22px !important;
    }
    .stTabs [data-baseweb="tab"] * { color: #e2e8f0 !important; opacity: 1 !important; }
    .stTabs [aria-selected="true"], .stTabs button[aria-selected="true"] {
        background-color: #003d5c44 !important; border-color: #38bdf8 !important; color: #38bdf8 !important;
    }
    .stTabs [aria-selected="true"] * { color: #38bdf8 !important; }
    [data-baseweb="tab"] { opacity: 1 !important; }
    /* ── Expander ── */
    [data-testid="stExpander"] { border: 1px solid #1e3a5f !important; border-radius: 8px !important; background: #0a1628 !important; margin-bottom: 6px !important; overflow: hidden !important; }
    [data-testid="stExpander"] details { background: #0a1628 !important; }
    [data-testid="stExpander"] details[open] { background: #0a1628 !important; }
    .streamlit-expanderHeader, [data-testid="stExpander"] details summary { background: #0a1628 !important; color: #cbd5e1 !important; font-weight: 600 !important; border: none !important; border-radius: 8px !important; padding: 12px 16px !important; }
    [data-testid="stExpander"] details[open] summary { background: #0d1f3c !important; color: #38bdf8 !important; border-bottom: 1px solid #1e3a5f !important; border-radius: 8px 8px 0 0 !important; }
    .streamlit-expanderHeader:focus, [data-testid="stExpander"] *:focus, [data-testid="stExpander"] summary:focus-visible { outline: none !important; box-shadow: none !important; }
    .streamlit-expanderContent, [data-testid="stExpander"] details > div { background: #060e1a !important; border-top: 1px solid #1e3a5f !important; border-radius: 0 0 8px 8px !important; }

    /* 전체 슬라이더 Min/Max 박스 위로 올리기 */
    [data-testid="stSlider"] + div [data-testid="stNumberInput"],
    div[data-testid="column"]:nth-child(2) [data-testid="stNumberInput"],
    div[data-testid="column"]:nth-child(3) [data-testid="stNumberInput"] {
        margin-top: -18px !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. 다국어 사전
LANG_DICT = {
    "KO": {
        "title": "JOINT 설계 & 공정 최적화   V1.5",
        "console": "데이터 컨트롤",
        "upload_title": "입력 데이터",
        "upload_help": "신규 입력 데이터 파일 업로드 (CSV, XLSX, DB)",
        "upload_hist_help": "기존 누적 DB 파일 업로드 (선택)",
        "init_btn": "학습 초기화 및 데이터 통합 학습 실행",
        "tab1": "Spec.(품질) 타겟팅 최적화",
        "tab2": "설계 & 공정 변수 타겟팅 최적화",
        "tab3": "마스터 데이터 & 분석",
        "tab4": "실시간 최적화 결과 예측",
        "bound_title": "경계 조건 최적화 도구",
        "bound_mode": "안전 경계 제한 모드",
        "kpi_title": "목표 품질 타겟 값 범위 설정",
        "run_opt": "역추론 최적화 탐색 실행",
        "pred_title": "예측 성능 분석",
        "rec_title": "추천 공정 스펙 조건 (34개 입력 변수)",
        "live_input": "실시간 가상 타겟 범위 설정 (What-If)",
        "run_sim": "가상 역최적화 파라미터 도출",
        "sim_title": "시뮬레이션 역산 도출 결과 (34개 공정 변수)",
        "sim_pred_title": "도출 조건 기준 최종 예측 품질",
        "engine_inactive": "학습 비활성화: 사이드바를 통해 입력 데이터를 업로드하십시오.",
        "best_algo": "최적 선택 알고리즘",
        "opt_conf": "목표 최적화 신뢰도",
        "dl_format": "내보내기 파일 포맷 선택",
        "dl_btn_spec": "추천 스펙 데이터 다운로드",
        "dl_btn_pred": "예측 성능 데이터 다운로드",
        "db_export_title": "💾 데이터베이스 외부 내보내기",
        "db_prepare_btn": " DB 스냅샷 생성 및 서버 저장",
        "db_current_latest": " 최신 데이터 상태가 파일에 이미 반영되어 있습니다.",
        "db_prepared_msg": "준비된 파일: ",
        "db_pc_download": "📥 내보낸 DB 파일 PC로 직접 다운로드",
        "db_save_empty": "저장할 데이터가 없습니다. 먼저 데이터 업로드 후 엔진 초기화를 완료해 주세요.",
        "ai_title": " JOINT AI 공정 인사이트 가이드",
        "ai_btn": "LLM 기반 공정 가이드라인 생성",
        "ai_loading": "최적화 변수와 품질 타겟 값 데이터를 분석하여 팩토리 가이드를 생성 중입니다..."
    },
    "EN": {
        "title": "JOINT PROCESS INTELLIGENCE",
        "console": "CONTROL CONSOLE",
        "upload_title": "Master Data Stream",
        "upload_help": "Upload New Log File (CSV, XLSX, DB)",
        "upload_hist_help": "Upload Existing History DB File (Optional)",
        "init_btn": "RUN ENGINE INIT & DATA MERGE",
        "tab1": "QUALITY SPEC. TARGETING",
        "tab2": "DESIGN & PROCESS VAR. TARGETING",
        "tab3": "MASTER DATA & ANALYTICS",
        "tab4": "REAL-TIME OPTIMIZATION PREDICTION",
        "bound_title": "Boundary Condition Optimizer",
        "bound_mode": "Safety Bound Limit Mode",
        "kpi_title": "Target Quality KPIs Range Configurator",
        "run_opt": "RUN INVERSE INFERENCE SEARCH",
        "pred_title": "Predicted Performance Analysis",
        "rec_title": "Recommended Process Specifications (34 Variables)",
        "live_input": "Real-time Virtual Target Range Configurator (What-If)",
        "run_sim": "EXECUTE VIRTUAL INVERSE OPTIMIZATION",
        "sim_title": "Inversed Simulation Results (34 Process Variables)",
        "sim_pred_title": "Final Predicted Quality via Inversed Specs",
        "engine_inactive": "CORE ENGINE INACTIVE: Please upload log data via sidebar.",
        "best_algo": "Selected Best Algorithm",
        "opt_conf": "Target Optimization Confidence",
        "dl_format": "Select Export File Format",
        "dl_btn_spec": "DOWNLOAD RECOMMENDED SPECS",
        "dl_btn_pred": "DOWNLOAD PREDICTED PERFORMANCE",
        "db_export_title": "💾 External Database Export",
        "db_prepare_btn": " Generate & Save DB Snapshot",
        "db_current_latest": " The file contains the latest data state.",
        "db_prepared_msg": "Prepared File: ",
        "db_pc_download": "📥 Download Saved DB File to PC Directly",
        "db_save_empty": "No data available to save. Please run Engine Initialization first.",
        "ai_title": " JOINT AI Process Insight Guidance",
        "ai_btn": "Generate LLM-based Process Guidelines",
        "ai_loading": "Analyzing optimized variables and quality KPI data to generate factory guidance..."
    }
}

if "lang" not in st.session_state:
    st.session_state["lang"] = "KO"

LOGIN_TEXT = {
    "KO": {
        "badge": "JOINT AI SYSTEM",
        "title": "AI 머신러닝을 이용한 JOINT 설계 & 공정 최적화",
        "pwd_label": "비밀번호 입력",
        "auth_btn": "시스템 접속",
        "invalid": "비밀번호가 올바르지 않습니다."
    },
    "EN": {
        "badge": "JOINT AI SYSTEM",
        "title": "Joint Design and Process Optimization using AI Machine Learning",
        "pwd_label": "Enter Password",
        "auth_btn": "Authenticate",
        "invalid": "Invalid credentials."
    }
}

# 4. 인증 시스템
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    _, center, _ = st.columns([1, 1.8, 1])
    with center:
        st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)

        # 언어 선택 - 배경 밝은 회색, 글자 검정
        st.markdown("""<style>
            div[data-testid="stSelectbox"] > div > div {
                background-color: #e2e8f0 !important;
                color: #1e293b !important;
                border-color: #cbd5e1 !important;
            }
            div[data-testid="stSelectbox"] > div > div > div {
                color: #1e293b !important;
            }
            div[data-baseweb="select"] > div {
                background-color: #e2e8f0 !important;
                border-color: #cbd5e1 !important;
            }
            div[data-baseweb="select"] span {
                color: #1e293b !important;
            }
        </style>""", unsafe_allow_html=True)

        _, lang_select_col = st.columns([5, 1.1])
        with lang_select_col:
            lang_display_options = ["KO", "EN"]
            current_display = st.session_state["lang"]
            lang_choice_login = st.selectbox(
                "Language", lang_display_options,
                index=lang_display_options.index(current_display),
                label_visibility="collapsed",
                key="login_lang_select"
            )
            new_lang = lang_choice_login
            if new_lang != st.session_state["lang"]:
                st.session_state["lang"] = new_lang
                st.rerun()

        LT = LOGIN_TEXT[st.session_state["lang"]]

        # 제목 박스 - 세로 2/3 축소 (padding 44px→22px)
        st.markdown(
            f"""<div class='glass-card' style='text-align:center; padding:22px 36px; margin-top:12px;'>
                <div style='color:#38bdf8; font-size:0.78rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:8px;'>{LT['badge']}</div>
                <h2 style='color:#f1f5f9; font-size:1.35rem; font-weight:600; line-height:1.4; margin:0 0 4px 0;'>{LT['title']}</h2>
                <div style='width:56px; height:2px; background:#10b981; margin:12px auto 0 auto;'></div>
            </div>""",
            unsafe_allow_html=True
        )

        _pw_col, _btn_col = st.columns([4, 1])
        with _pw_col:
            pwd = st.text_input(LT['pwd_label'], type="password",
                                label_visibility="collapsed",
                                placeholder=LT['pwd_label'])
        with _btn_col:
            _login_btn = st.button(LT['auth_btn'], type="primary",
                                   use_container_width=True)
        if _login_btn:
            if pwd == "iljin1234":
                st.session_state.authenticated = True
                st.rerun()
            else: st.error(LT['invalid'])
    st.stop()

col_title, col_lang = st.columns([8, 1])
with col_lang:
    lang_choice = st.selectbox("🌐 Lang", ["KO", "EN"], index=0 if st.session_state["lang"] == "KO" else 1, label_visibility="collapsed")
    if lang_choice != st.session_state["lang"]:
        st.session_state["lang"] = lang_choice
        st.rerun()

L_G = LANG_DICT[st.session_state["lang"]]

# 5. 입력 변수 및 타겟 정의 (ABAMS, RBAMS 추가됨)
X_list = [
    'BD', 'CID', 'BH', 'CIH', 'CITH', 'COHB', 'CD1', 'CD2', 'CD3', 'CD4', 'CD5', 
    'CH1', 'CH2', 'CH3', 'CH4', 'CGW', 'CGD', 'CD6', 'CR', 'CF', 'SH1', 'SR', 
    'SH2', 'SH3', 'SH4', 'SH5', 'SH6', 'SID', 'SOD', 'SH7', 'SH8', 'SH9', 'SR2', 'COHA'
]
target_vars = ['BT', 'RT', 'AGB', 'RGB', 'AGA', 'RGA', 'AGI', 'RGI', 'ABAMS', 'RBAMS']

# 기본 Spec 가이드 텍스트 (ABAMS, RBAMS 스펙 25.0 ~ 100.0 수정)
def generate_combined_report(process_specs, predicted_kpis, feasibility_info,
                              confidence_score, mode="Optimization", range_key_prefix=''):
    """
    Feature Importance 기반 진단 + LLM 가이드라인을 하나의 보고서로 통합 생성.
    1단계: generate_diagnosis_guide()로 FI 분석 텍스트 생성
    2단계: FI 분석 결과를 LLM 프롬프트에 포함하여 더 풍부한 가이드라인 생성
    3단계: 두 결과를 하나의 보고서로 합쳐서 반환
    """
    is_en = st.session_state.get('lang', 'KO') == 'EN'

    # ── 1단계: FI 기반 진단 ────────────────────────────────────────
    fi_guide = generate_diagnosis_guide(
        feasibility_info=feasibility_info,
        predicted_kpis=predicted_kpis,
        opt_result_x=None,
        confidence_score=confidence_score,
        range_key_prefix=range_key_prefix
    )

    # ── 2단계: FI 진단 요약을 LLM 프롬프트에 추가 ──────────────────
    api_key = None
    try:
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        # LLM 없이 FI 진단만 반환
        if is_en:
            return fi_guide + "\n\n---\n⚠️ LLM API key not configured — only FI-based diagnosis is provided."
        return fi_guide + "\n\n---\n⚠️ LLM API 키 미설정 — FI 기반 진단만 제공됩니다."

    client = Groq(api_key=api_key)

    # 개선③: 모든 수치를 소수점 3자리로 통일
    specs_str = "\n".join([
        f"  - {k} [{VAR_GLOSSARY.get(k, 'No definition' if is_en else '정의 없음')}]: {v:.3f}"
        for k, v in process_specs.items()
    ])

    # 개선②: 데이터 범위(외삽) 초과 변수 사전 계산 — AI가 임의 판단하지 않고 이 목록을 그대로 인용
    extrapolation_items = []
    for k, v in process_specs.items():
        try:
            _rng = db.get(k)
        except Exception:
            _rng = None
        if _rng is not None:
            _lo_x, _hi_x = _rng
            if v < _lo_x or v > _hi_x:
                extrapolation_items.append((k, v, _lo_x, _hi_x))
    if extrapolation_items:
        if is_en:
            extrapolation_str = "\n".join([
                f"  - {k}: current value {v:.3f} is outside the training data range ({lo:.3f} ~ {hi:.3f}) — extrapolation risk"
                for k, v, lo, hi in extrapolation_items
            ])
        else:
            extrapolation_str = "\n".join([
                f"  - {k}: 현재값 {v:.3f}이(가) 학습 데이터 범위({lo:.3f} ~ {hi:.3f})를 벗어남 — 외삽 위험"
                for k, v, lo, hi in extrapolation_items
            ])
    else:
        extrapolation_str = ("None — all variables are within the training data range." if is_en
                              else "없음 — 모든 변수가 학습 데이터 범위 내에 있습니다.")

    kpis_lines = []
    spec_ok_list, spec_warn_list, spec_na_list = [], [], []
    for k, v in predicted_kpis.items():
        if v is None:
            continue
        # 화면에 표시되는 값과 일치시키기 위해, 사용자가 현재 설정한 Range를 우선 사용
        # (없으면 고정 기본 스펙 SPEC_GUIDE로 대체)
        _cur_range_k = st.session_state.get(f'{range_key_prefix}{k.lower()}_s_val')
        if _cur_range_k is not None:
            spec_range = f"{float(_cur_range_k[0]):.3f}~{float(_cur_range_k[1]):.3f}"
        else:
            spec_range = SPEC_GUIDE.get(k, "N/A")
        glossary   = TARGET_GLOSSARY.get(k, k)
        na_tgts    = st.session_state.get('na_spec_targets', [])
        if k in na_tgts or spec_range == "N/A" or "~" not in spec_range:
            spec_na_list.append(k)
            if is_en:
                kpis_lines.append(f"  - {k} [{glossary}]: {v:.3f}  (Spec: N/A — excluded from optimization)")
            else:
                kpis_lines.append(f"  - {k} [{glossary}]: {v:.3f}  (스펙: N/A — 최적화 제외)")
        else:
            lo, hi = map(float, spec_range.split("~"))
            in_spec = lo <= v <= hi
            margin_lo, margin_hi = v - lo, hi - v
            (spec_ok_list if in_spec else spec_warn_list).append(k)
            if is_en:
                status = "Pass" if in_spec else "Fail"
                kpis_lines.append(
                    f"  - {k} [{glossary}]: {v:.3f}  "
                    f"(Spec: {spec_range} / Judgement: {status} / "
                    f"Lower margin: {margin_lo:+.3f}, Upper margin: {margin_hi:+.3f})"
                )
            else:
                status = "적합" if in_spec else "이탈"
                kpis_lines.append(
                    f"  - {k} [{glossary}]: {v:.3f}  "
                    f"(스펙: {spec_range} / 판정: {status} / "
                    f"하한여유: {margin_lo:+.3f}, 상한여유: {margin_hi:+.3f})"
                )
    kpis_str = "\n".join(kpis_lines)

    # FI 진단 핵심 요약 (LLM에 전달)
    fi_summary_lines = []
    for line in fi_guide.split("\n"):
        if any(k in line for k in ['####', '| ', 'GAP', '우선 조정', 'Priority', 'Feature Importance', '권장', '경로', 'Path', '예측값', 'Predicted', '스펙 이탈', 'Out-of-Spec', '달성 불가', 'Infeasible']):
            fi_summary_lines.append(line)
    fi_summary = "\n".join(fi_summary_lines[:40])  # 최대 40줄만 전달

    # 개선①: Tab1(역방향)/Tab2(순방향) 방향성을 명확히 구분
    if is_en:
        mode_desc = {
            "Optimization": "recommended process specs derived via inverse optimization (backward: quality target → process variables)",
            "Simulation":   "process variables derived via forward search within a user-defined design/process variable range (forward: process variables → quality check)"
        }.get(mode, "process optimization result")
        direction_term = "inverse optimization (backward)" if mode == "Optimization" else "forward search"
    else:
        mode_desc = {
            "Optimization": "역최적화 알고리즘으로 도출한 추천 공정 스펙 결과 (역방향: 품질 타겟 → 설계/공정 변수)",
            "Simulation":   "설계/공정 변수 목표 범위 내에서 순방향 탐색으로 도출한 결과 (순방향: 설계/공정 변수 → 품질 확인)"
        }.get(mode, "공정 최적화 결과")
        direction_term = "역최적화(역방향)" if mode == "Optimization" else "순방향 탐색"

    doc_no = f"JOINT-OPS-CABJ-{datetime.now().strftime('%Y%m%d')}-001"
    today  = datetime.now().strftime("%B %d, %Y") if is_en else datetime.now().strftime("%Y년 %m월 %d일")

    if is_en:
        system_instruction = (
            "You are an AI assistant specialized in process engineering, embedded in the JOINT AI - Process "
            "Optimization Suite. You are trained on VOLVO SPA1/2 CABJ (ball stud joint) swaging assembly "
            "process data, and support two directions of analysis: "
            "Tab1 (backward: derive a design/process variable combination via inverse optimization that "
            "satisfies a quality target) and Tab2 (forward: search for a combination that satisfies the "
            f"quality spec within a user-defined design/process variable target range). "
            f"The current analysis mode is **{mode}** ({direction_term}) — use only terminology consistent "
            "with this direction; do not mix the two.\n\n"
            "Report writing rules:\n"
            "1. Do not extend beyond the provided data with general knowledge or other parts.\n"
            "2. Write in English, in a professional working-report format. Make active use of headers, "
            "subheadings, and tables.\n"
            "3. Do not use strikethrough (~~text~~); use bold (**text**) only for emphasis.\n"
            "4. Always include numeric evidence, and write each section in sufficient detail.\n"
            "5. When Feature Importance analysis results are provided, actively use them to explain causal "
            "relationships.\n"
            "6. Round all numeric values to exactly 3 decimal places (e.g., 5.288, 0.062) consistently "
            "throughout the report.\n"
            "7. Use markdown tables (with '|') only for tabular data — do not mix bullet lists and tables "
            "for the same content.\n"
            "8. If any variable is flagged as being outside the training data range (extrapolation), you "
            "must explicitly warn about it — do not judge extrapolation risk yourself; cite the "
            "'Out-of-range (extrapolation) variables' list provided below exactly as given."
        )
    else:
        system_instruction = (
            "당신은 JOINT AI - Process Optimization Suite에 내장된 공정 엔지니어링 전문 AI 어시스턴트입니다. "
            "VOLVO SPA1/2 CABJ(볼스터드 조인트) 스웨이징 조립 공정 데이터를 학습하여, "
            "Tab1(역방향: 품질 타겟값을 만족하는 설계/공정 변수 조합을 역최적화로 도출)과 "
            "Tab2(순방향: 설계/공정 변수 타겟 값 범위 내에서 품질 스펙을 만족하는 조합을 탐색)를 지원하는 시스템입니다. "
            f"현재 분석 모드는 **{mode}**({direction_term})이며, 이 방향에 맞는 용어만 사용하고 두 방향을 혼용하지 마세요.\n\n"
            "보고서 작성 규칙:\n"
            "1. 제공된 데이터 외 일반 지식이나 다른 부품으로 확장하지 마세요.\n"
            "2. 한국어, 전문 실무형 보고서 형식으로 작성하세요. 헤더·소제목·표를 적극 활용하세요.\n"
            "3. 한자(漢字)는 절대 사용하지 마세요. 순수 한글과 영문 약어/숫자만 사용하세요.\n"
            "4. 취소선(~~텍스트~~)은 사용하지 말고, 강조는 굵게(**텍스트**)만 사용하세요.\n"
            "5. 수치 근거를 반드시 포함하고, 각 섹션을 충분히 상세하게 작성하세요.\n"
            "6. Feature Importance 분석 결과가 제공되면 이를 적극 활용하여 인과 관계를 설명하세요.\n"
            "7. 모든 수치는 소수점 셋째 자리까지 통일하여 표기하세요 (예: 5.288, 0.062).\n"
            "8. 표는 반드시 '|' 기준 markdown 표만 사용하고, 같은 내용에 표와 불릿을 섞지 마세요.\n"
            "9. 데이터 범위를 벗어나는(외삽) 변수가 있으면 반드시 명시적으로 경고하세요 — 임의로 판단하지 말고 "
            "아래 제공된 '데이터 범위 초과(외삽) 변수' 목록을 그대로 인용하세요."
        )

    if is_en:
        prompt = f"""## JOINT AI Integrated Process Analysis Report: CABJ Swaging Assembly Process
Document No.: {doc_no}  Date: {today}  Analysis Mode: {mode}
Prepared by: JOINT AI Process Engineering Assistant

---

[Analysis Mode] {mode} — {mode_desc}

[Recommended Process Variable Specs (34 part dimension variables)]
{specs_str}

[Out-of-range (extrapolation) variables — cite this list exactly, do not judge yourself]
{extrapolation_str}

[Predicted Quality Target Values and Spec Conformance]
{kpis_str}

Passing targets: {', '.join(spec_ok_list) if spec_ok_list else 'None'}
Out-of-spec/warning targets: {', '.join(spec_warn_list) if spec_warn_list else 'None'}
Spec N/A (excluded from optimization): {', '.join(spec_na_list) if spec_na_list else 'None'}

[Feature Importance Pre-Diagnosis Results (reference)]
{fi_summary}

---

[Writing Request — include and elaborate all sections below]

**[3-line summary]**
Before section 1, write a concise 3-line summary containing only the core conclusions (overall achievement status, most critical variable, one key recommendation).

1. **Result Summary** (3-5 sentences)
   - Overall characteristics of the derived process specs, summary of predicted KPI achievement
   - State the number of passing/failing KPIs out of the total

2. **KPI Conformance Assessment** (table format)
   Write as: KPI item | Predicted value | Normal spec range | Conformance | Notes (e.g., near lower/upper bound, extrapolation risk).
   - Mention specific figures for KPIs near the lower/upper bound.
   - Explicitly state if there is a model extrapolation risk, using the extrapolation list provided above.

3. **Process Variables Requiring Attention** (top 3-5, based on Feature Importance)
   - Explain the physical mechanism by which each variable affects quality.
   - Example: COHB/COHA → swaging depth → joint clamping force → effect on BT/RT/ABAMS

4. **Field Application Recommendations** (3 items, with specific figures)
   - Priority for swaging process precision control
   - Strengthened dimensional tolerance control for key parts (ball stud, seat, bearing)
   - Real-time KPI monitoring and feedback plan

5. **Risks and Limitations**
   - Specific risk factors to check before trusting this result: data-sparse regions, extrapolated variables, targets with lower model accuracy, etc.
   - Be concrete and reference the extrapolation list and any infeasible/out-of-spec targets from the FI diagnosis above.

6. **Next-Step Action Plan**
   - A concrete execution plan for physical validation: which variables to test first, in what order, and why.
   - Tie this back to the "priority variables to review" identified in the Feature Importance diagnosis above."""
    else:
        prompt = f"""## JOINT AI 통합 공정 분석 보고서: CABJ 스웨이징 조립 공정
문서 번호: {doc_no}  작성일: {today}  분석 모드: {mode}
작성자: JOINT AI Process Engineering Assistant

---

[분석 모드] {mode} — {mode_desc}

[추천 공정 변수 스펙 (34개 단품 치수 변수)]
{specs_str}

[데이터 범위 초과(외삽) 변수 — 아래 목록을 그대로 인용하고 임의 판단하지 마세요]
{extrapolation_str}

[예측 품질 타겟 값 및 스펙 적합성]
{kpis_str}

적합 타겟: {', '.join(spec_ok_list) if spec_ok_list else '없음'}
이탈/주의 타겟: {', '.join(spec_warn_list) if spec_warn_list else '없음'}
스펙 N/A (최적화 제외): {', '.join(spec_na_list) if spec_na_list else '없음'}

[Feature Importance 기반 사전 진단 결과 (참고)]
{fi_summary}

---

[작성 요청 — 아래 섹션을 모두 포함하여 상세히 작성하세요]

**[3줄 요약]**
1번 섹션 앞에, 핵심 결론만 담은 3줄 요약(전체 달성 현황, 가장 중요한 변수, 핵심 권장사항 1가지)을 작성하세요.

1. **결과 요약** (3~5문장)
   - 도출된 공정 스펙의 전반적 특성, 예측 KPI 달성 현황 종합 요약
   - 전체 KPI 중 적합/이탈 개수 명시

2. **KPI 적합성 평가** (표 형식)
   KPI항목 | 예측값 | 정상스펙범위 | 적합성 | 비고(하한치근접·외삽위험 등) 형태로 작성하세요.
   - 하한/상한치에 근접한 KPI는 구체적 수치를 언급하세요.
   - 모델 외삽(Extrapolation) 위험이 있는 경우, 위에 제공된 외삽 변수 목록을 활용해 명시하세요.

3. **주의가 필요한 공정 변수** (상위 3~5개, Feature Importance 기반)
   - 각 변수가 품질에 미치는 물리적 영향 메커니즘을 설명하세요.
   - 예: COHB/COHA → 스웨이징 깊이 → 조인트 체결력 → BT/RT/ABAMS 영향

4. **현장 적용 권장사항** (3가지, 구체적 수치 포함)
   - 스웨이징 공정 정밀도 관리 우선순위
   - 핵심 단품(볼스터드·시트·베어링) 치수 공차 관리 강화
   - KPI 실시간 모니터링 및 피드백 방안

5. **리스크 및 한계**
   - 이 결과를 신뢰하기 전에 확인해야 할 구체적 위험 요소: 데이터 부족 구간, 외삽 변수, 모델 정확도가 낮은 타겟 등
   - 위의 외삽 변수 목록과 FI 진단의 달성 불가/스펙 이탈 타겟을 구체적으로 인용하세요.

6. **다음 단계 액션 플랜**
   - 실물 검증 시 어떤 변수를 어떤 순서로, 왜 우선 테스트할지 구체적 실행 계획을 제시하세요.
   - 위 Feature Importance 진단의 '우선 조정 검토 변수'와 연결하여 작성하세요."""

    try:
        priority = ['llama-3.3-70b-versatile', 'llama-3.1-70b-versatile', 'llama-3.1-8b-instant']
        target_model = priority[0]
        try:
            available_models = [m.id for m in client.models.list().data]
            target_model = next((m for m in priority if m in available_models),
                               available_models[0] if available_models else priority[0])
        except Exception:
            pass

        response = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.25,
            max_tokens=4096
        )
        llm_text = response.choices[0].message.content
        llm_text = re.sub(r'~~(.*?)~~', r'\1', llm_text)
        llm_text = re.sub(r'[\u4e00-\u9fff]+', '', llm_text)
        llm_text = re.sub(r'[ \t]{2,}', ' ', llm_text)

    except Exception as e:
        err = str(e)
        if "429" in err:
            llm_text = ("⏳ API usage limit reached. Please try again shortly." if is_en
                        else "⏳ API 사용량이 많습니다. 잠시 후 다시 시도해 주세요.")
        else:
            llm_text = f"❌ LLM Error: {err}" if is_en else f"❌ LLM 오류: {err}"

    # ── 3단계: 두 결과 통합 ────────────────────────────────────────
    fi_section_header = "## □ Feature Importance-based Process Diagnosis (Detailed)" if is_en else "## □ Feature Importance 기반 공정 진단 상세 분석"
    combined = (
        f"{llm_text}\n\n"
        f"---\n\n"
        f"{fi_section_header}\n\n"
        f"{fi_guide}"
    )
    return combined


spec_limits = {
    'BT': (0.0, 8.0), 'RT': (0.0, 4.0), 'AGB': (0.0, 0.3),
    'RGB': (0.0, 0.4), 'AGA': (0.0, 1.0), 'RGA': (0.0, 1.0),
    'AGI': (0.0, 1.0), 'RGI': (0.0, 1.0),
    'ABAMS': (25.0, 100.0), 'RBAMS': (25.0, 100.0)
}
# [추가] 파일에서 읽은 스펙이 있으면 spec_limits 동적 업데이트
_spec_file = st.session_state.get('spec_from_file', {})
if _spec_file:
    spec_limits.update(_spec_file)
SPEC_GUIDE = {k: f"{v[0]}~{v[1]}" for k, v in spec_limits.items()}

class _LangGlossary:
    """언어 설정(st.session_state['lang'])에 따라 KO/EN 설명을 자동으로 반환하는 딕셔너리 래퍼.
    기존 .get(key, default) 호출부는 전혀 수정할 필요 없이 자동으로 언어가 바뀝니다."""
    def __init__(self, ko_dict, en_dict):
        self._ko = ko_dict
        self._en = en_dict

    def _active(self):
        return self._en if st.session_state.get('lang', 'KO') == 'EN' else self._ko

    def get(self, key, default=None):
        return self._active().get(key, default)

    def __getitem__(self, key):
        return self._active()[key]

    def __contains__(self, key):
        return key in self._ko

    def keys(self):
        return self._active().keys()

    def values(self):
        return self._active().values()

    def items(self):
        return self._active().items()

    def __iter__(self):
        return iter(self._active())

    def __len__(self):
        return len(self._ko)

VAR_GLOSSARY_KO = {
    'BD': 'ballstud_diameter_mm (볼스터드 직경)',
    'CID': 'case_inner_diameter_mm (케이스 내경)',
    'BH': 'bearing_Height_mm (베어링 높이)',
    'CIH': 'case_inner_height_mm (케이스 내측 높이)',
    'CITH': 'case_inner_taper_height_mm (케이스 내측 테이퍼 높이)',
    'COHB': 'case_outer_height_before_mm (케이스 외측 높이, 스웨이징 전)',
    'COHA': 'case_outer_height_after_mm (케이스 외측 높이, 스웨이징 후)',
    'CD1': 'case_d1_mm (케이스 치수 D1)',
    'CD2': 'case_d2_mm (케이스 치수 D2)',
    'CD3': 'case_d3_mm (케이스 치수 D3)',
    'CD4': 'case_d4_mm (케이스 치수 D4)',
    'CD5': 'case_d5_mm (케이스 치수 D5)',
    'CD6': 'case_d6_mm (케이스 치수 D6)',
    'CH1': 'case_h1_mm (케이스 높이 H1)',
    'CH2': 'case_h2_mm (케이스 높이 H2)',
    'CH3': 'case_h3_mm (케이스 높이 H3)',
    'CH4': 'case_h4_mm (케이스 높이 H4)',
    'CGW': 'case_groove_width_mm (케이스 그루브 폭)',
    'CGD': 'case_groove_depth_mm (케이스 그루브 깊이)',
    'CR': 'case_roundness_mm (케이스 진원도)',
    'CF': 'case_flatness_mm (케이스 평면도)',
    'SH1': 'seat_h1_mm (시트 높이 H1)',
    'SR': 'seat_R_mm (시트 R 치수)',
    'SH2': 'seat_h2_mm (시트 높이 H2)',
    'SH3': 'seat_h3_mm (시트 높이 H3)',
    'SH4': 'seat_h4_mm (시트 높이 H4)',
    'SH5': 'seat_h5_mm (시트 높이 H5)',
    'SH6': 'seat_h6_mm (시트 높이 H6)',
    'SID': 'seat_inner_d_mm (시트 내경)',
    'SOD': 'seat_outer_d_mm (시트 외경)',
    'SH7': 'seat_h7_mm (시트 높이 H7)',
    'SH8': 'seat_h8_mm (시트 높이 H8)',
    'SH9': 'seat_h9_mm (시트 높이 H9)',
    'SR2': 'seat_R2_mm (시트 R2 치수)',
}

VAR_GLOSSARY_EN = {
    'BD': 'ballstud_diameter_mm (Ball stud diameter)',
    'CID': 'case_inner_diameter_mm (Case inner diameter)',
    'BH': 'bearing_Height_mm (Bearing height)',
    'CIH': 'case_inner_height_mm (Case inner height)',
    'CITH': 'case_inner_taper_height_mm (Case inner taper height)',
    'COHB': 'case_outer_height_before_mm (Case outer height, before swaging)',
    'COHA': 'case_outer_height_after_mm (Case outer height, after swaging)',
    'CD1': 'case_d1_mm (Case dimension D1)',
    'CD2': 'case_d2_mm (Case dimension D2)',
    'CD3': 'case_d3_mm (Case dimension D3)',
    'CD4': 'case_d4_mm (Case dimension D4)',
    'CD5': 'case_d5_mm (Case dimension D5)',
    'CD6': 'case_d6_mm (Case dimension D6)',
    'CH1': 'case_h1_mm (Case height H1)',
    'CH2': 'case_h2_mm (Case height H2)',
    'CH3': 'case_h3_mm (Case height H3)',
    'CH4': 'case_h4_mm (Case height H4)',
    'CGW': 'case_groove_width_mm (Case groove width)',
    'CGD': 'case_groove_depth_mm (Case groove depth)',
    'CR': 'case_roundness_mm (Case roundness)',
    'CF': 'case_flatness_mm (Case flatness)',
    'SH1': 'seat_h1_mm (Seat height H1)',
    'SR': 'seat_R_mm (Seat R dimension)',
    'SH2': 'seat_h2_mm (Seat height H2)',
    'SH3': 'seat_h3_mm (Seat height H3)',
    'SH4': 'seat_h4_mm (Seat height H4)',
    'SH5': 'seat_h5_mm (Seat height H5)',
    'SH6': 'seat_h6_mm (Seat height H6)',
    'SID': 'seat_inner_d_mm (Seat inner diameter)',
    'SOD': 'seat_outer_d_mm (Seat outer diameter)',
    'SH7': 'seat_h7_mm (Seat height H7)',
    'SH8': 'seat_h8_mm (Seat height H8)',
    'SH9': 'seat_h9_mm (Seat height H9)',
    'SR2': 'seat_R2_mm (Seat R2 dimension)',
}

VAR_GLOSSARY = _LangGlossary(VAR_GLOSSARY_KO, VAR_GLOSSARY_EN)

TARGET_GLOSSARY_KO = {
    'BT': 'breakaway_torque_Nm (분리 토크 / 초기 회전 토크)',
    'RT': 'running_torque_Nm (회전 토크)',
    'AGB': 'axial_gap_before_mm (축방향 유격, 내구 시험 전)',
    'RGB': 'radial_gap_before_mm (반경방향 유격, 내구 시험 전)',
    'AGA': 'axial_gap_after_mm (축방향 유격, 내구 시험 후)',
    'RGA': 'radial_gap_after_mm (반경방향 유격, 내구 시험 후)',
    'AGI': 'axial_gap_increase_mm (축방향 유격 증가량, 시험 전후 차)',
    'RGI': 'radial_gap_increase_mm (반경방향 유격 증가량, 시험 전후 차)',
    'ABAMS': 'axial_bams_percent (축방향 BAMS율)',
    'RBAMS': 'radial_bams_percent (반경방향 BAMS율)',
}

TARGET_GLOSSARY_EN = {
    'BT': 'breakaway_torque_Nm (Breakaway torque / initial rotation torque)',
    'RT': 'running_torque_Nm (Running torque)',
    'AGB': 'axial_gap_before_mm (Axial clearance, before durability test)',
    'RGB': 'radial_gap_before_mm (Radial clearance, before durability test)',
    'AGA': 'axial_gap_after_mm (Axial clearance, after durability test)',
    'RGA': 'radial_gap_after_mm (Radial clearance, after durability test)',
    'AGI': 'axial_gap_increase_mm (Axial clearance increase, before/after difference)',
    'RGI': 'radial_gap_increase_mm (Radial clearance increase, before/after difference)',
    'ABAMS': 'axial_bams_percent (Axial BAMS rate)',
    'RBAMS': 'radial_bams_percent (Radial BAMS rate)',
}

TARGET_GLOSSARY = _LangGlossary(TARGET_GLOSSARY_KO, TARGET_GLOSSARY_EN)

def gray_out_slider(aria_label):
    """특정 슬라이더를 회색으로 만드는 JS 주입.
    Streamlit 컨테이너/CSS 클래스 방식은 버전에 따라 DOM 구조가 달라 신뢰할 수 없어서,
    슬라이더의 aria-label(각 슬라이더에 부여한 고유 라벨 문자열)로 실제 렌더링된 DOM
    요소를 직접 찾아 인라인으로 회색 필터를 적용하는 방식으로 완전히 대체.
    label_visibility='collapsed'여도 Streamlit은 접근성을 위해 aria-label을 그대로 남겨둠.
    주의: st.markdown(unsafe_allow_html=True)은 보안상 <script> 태그를 실행하지 않으므로
    (렌더링은 되지만 스크립트가 동작 안 함), 실제로 스크립트가 실행되는
    st.components.v1.html(iframe 기반)을 사용해야 함."""
    _safe_label = aria_label.replace("\\", "\\\\").replace('"', '\\"')
    components.html(
        f"""<script>
        (function() {{
            function grayIt() {{
                var el = window.parent.document.querySelector('[aria-label="{_safe_label}"]');
                if (!el) return false;
                var wrap = el.closest('[data-testid="stSlider"]') || el.closest('[data-testid="stNumberInput"]') || el;
                wrap.style.filter = 'grayscale(100%)';
                wrap.style.opacity = '0.85';
                return true;
            }}
            if (!grayIt()) {{
                var tries = 0;
                var t = window.parent.setInterval(function() {{
                    tries++;
                    if (grayIt() || tries > 20) window.parent.clearInterval(t);
                }}, 150);
            }}
        }})();
        </script>""",
        height=1
    )

def on_slider_change(prefix):
    val_tuple = st.session_state[f'{prefix}_s_val']
    if not isinstance(val_tuple, (list, tuple)):
        val_tuple = (float(val_tuple), float(val_tuple))
        st.session_state[f'{prefix}_s_val'] = val_tuple
    st.session_state[f'{prefix}_n_min'] = val_tuple[0]
    st.session_state[f'{prefix}_n_max'] = val_tuple[1]

def on_min_change(prefix):
    current_slider = st.session_state[f'{prefix}_s_val']
    if not isinstance(current_slider, (list, tuple)):
        current_slider = (float(current_slider), float(current_slider))
    new_min = st.session_state[f'{prefix}_n_min']
    if new_min > current_slider[1]:
        new_min = current_slider[1]
        st.session_state[f'{prefix}_n_min'] = new_min
    st.session_state[f'{prefix}_s_val'] = (new_min, current_slider[1])

def on_max_change(prefix):
    current_slider = st.session_state[f'{prefix}_s_val']
    if not isinstance(current_slider, (list, tuple)):
        current_slider = (float(current_slider), float(current_slider))
    new_max = st.session_state[f'{prefix}_n_max']
    if new_max < current_slider[0]:
        new_max = current_slider[0]
        st.session_state[f'{prefix}_n_max'] = new_max
    st.session_state[f'{prefix}_s_val'] = (current_slider[0], new_max)

def on_sim_slider_change(prefix):
    val_tuple = st.session_state[f'sim_tgt_{prefix}_s_val']
    if not isinstance(val_tuple, (list, tuple)):
        val_tuple = (float(val_tuple), float(val_tuple))
        st.session_state[f'sim_tgt_{prefix}_s_val'] = val_tuple
    st.session_state[f'sim_tgt_{prefix}_n_min'] = val_tuple[0]
    st.session_state[f'sim_tgt_{prefix}_n_max'] = val_tuple[1]

def on_sim_min_change(prefix):
    # current_slider가 튜플 형태인지 확인하고, 아니면 기본값을 부여
    current_slider = st.session_state.get(f'sim_tgt_{prefix}_s_val', (0.0, 1.0))
    if not isinstance(current_slider, (list, tuple)):
        current_slider = (0.0, 1.0)
        
    new_min = st.session_state[f'sim_tgt_{prefix}_n_min']
    
    # new_min이 slider[1]보다 크면 강제로 고정
    if new_min > current_slider[1]:
        new_min = current_slider[1]
        st.session_state[f'sim_tgt_{prefix}_n_min'] = new_min
        
    st.session_state[f'sim_tgt_{prefix}_s_val'] = (float(new_min), float(current_slider[1]))

def on_sim_max_change(prefix):
    current_slider = st.session_state.get(f'sim_tgt_{prefix}_s_val', (0.0, 1.0))
    new_max = st.session_state.get(f'sim_tgt_{prefix}_n_max', 1.0)
    
    if isinstance(current_slider, (list, tuple)):
        min_val = current_slider[0]
    else:
        min_val = 0.0
        
    if new_max < min_val:
        new_max = min_val
        st.session_state[f'sim_tgt_{prefix}_n_max'] = new_max
        
    st.session_state[f'sim_tgt_{prefix}_s_val'] = (min_val, float(new_max))

# [추가] Tab4 목표 Range 슬라이더 ↔ Min/Max 숫자 입력 양방향 동기화 콜백
def on_t4_slider_change(prefix):
    val_tuple = st.session_state[f't4_range_{prefix}_s_val']
    if not isinstance(val_tuple, (list, tuple)):
        val_tuple = (float(val_tuple), float(val_tuple))
        st.session_state[f't4_range_{prefix}_s_val'] = val_tuple
    st.session_state[f't4_range_{prefix}_n_min'] = val_tuple[0]
    st.session_state[f't4_range_{prefix}_n_max'] = val_tuple[1]

def on_t4_min_change(prefix):
    current_slider = st.session_state.get(f't4_range_{prefix}_s_val', (0.0, 1.0))
    if not isinstance(current_slider, (list, tuple)):
        current_slider = (0.0, 1.0)
    new_min = st.session_state[f't4_range_{prefix}_n_min']
    if new_min > current_slider[1]:
        new_min = current_slider[1]
        st.session_state[f't4_range_{prefix}_n_min'] = new_min
    st.session_state[f't4_range_{prefix}_s_val'] = (float(new_min), float(current_slider[1]))

def on_t4_max_change(prefix):
    current_slider = st.session_state.get(f't4_range_{prefix}_s_val', (0.0, 1.0))
    new_max = st.session_state.get(f't4_range_{prefix}_n_max', 1.0)
    if isinstance(current_slider, (list, tuple)):
        min_val = current_slider[0]
    else:
        min_val = 0.0
    if new_max < min_val:
        new_max = min_val
        st.session_state[f't4_range_{prefix}_n_max'] = new_max
    st.session_state[f't4_range_{prefix}_s_val'] = (min_val, float(new_max))

# [추가] Manual Expert Tuning 변수 슬라이더 ↔ Min/Max 숫자 입력 양방향 동기화 콜백
def on_manual_slider_change(v_clean):
    """슬라이더 조작 → Min/Max 숫자 입력 필드 값 갱신"""
    val = st.session_state.get(f'manual_slider_{v_clean}', (0.0, 100.0))
    if isinstance(val, (list, tuple)) and len(val) == 2:
        st.session_state[f'manual_min_{v_clean}'] = float(val[0])
        st.session_state[f'manual_max_{v_clean}'] = float(val[1])
        st.session_state[f'm_{v_clean}_min'] = float(val[0])
        st.session_state[f'm_{v_clean}_max'] = float(val[1])

def on_manual_min_change(v_clean):
    """Min 숫자 키인 → 슬라이더 위치 갱신"""
    new_min = float(st.session_state.get(f'manual_min_{v_clean}', 0.0))
    cur = st.session_state.get(f'manual_slider_{v_clean}', (0.0, 100.0))
    cur_max = float(cur[1]) if isinstance(cur, (list, tuple)) else 100.0
    new_min = min(new_min, cur_max)
    st.session_state[f'manual_slider_{v_clean}'] = (new_min, cur_max)
    st.session_state[f'm_{v_clean}_min'] = new_min

def on_manual_max_change(v_clean):
    """Max 숫자 키인 → 슬라이더 위치 갱신"""
    new_max = float(st.session_state.get(f'manual_max_{v_clean}', 100.0))
    cur = st.session_state.get(f'manual_slider_{v_clean}', (0.0, 100.0))
    cur_min = float(cur[0]) if isinstance(cur, (list, tuple)) else 0.0
    new_max = max(new_max, cur_min)
    st.session_state[f'manual_slider_{v_clean}'] = (cur_min, new_max)
    st.session_state[f'm_{v_clean}_max'] = new_max

def generate_ai_guidance(process_specs, predicted_kpis, mode="Optimization"):
    api_key = None
    try:
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "⚠️ API 키가 설정되지 않았습니다. .streamlit/secrets.toml 파일에 GROQ_API_KEY를 저장해 주세요."

    client = Groq(api_key=api_key)

    # 공정 변수 스펙 (전체)
    specs_str = "\n".join([
        f"  - {k} [{VAR_GLOSSARY.get(k, '정의되지 않은 변수')}]: {v:.3f}"
        for k, v in process_specs.items()
    ])

    # 예측 KPI (스펙 범위 포함, 적합성 판단)
    kpis_lines = []
    spec_ok_list = []
    spec_warn_list = []
    spec_na_list = []
    for k, v in predicted_kpis.items():
        spec_range = SPEC_GUIDE.get(k, "N/A")
        glossary = TARGET_GLOSSARY.get(k, '정의되지 않은 타겟 값')
        if spec_range == "N/A" or "~" not in spec_range:
            spec_na_list.append(k)
            kpis_lines.append(f"  - {k} [{glossary}]: {v:.3f}  (스펙: N/A — 최적화 제외)")
        else:
            lo, hi = map(float, spec_range.split("~"))
            in_spec = lo <= v <= hi
            margin_lo = v - lo
            margin_hi = hi - v
            status = "적합" if in_spec else "이탈"
            if in_spec:
                spec_ok_list.append(k)
            else:
                spec_warn_list.append(k)
            kpis_lines.append(
                f"  - {k} [{glossary}]: {v:.3f}  "
                f"(정상 스펙: {spec_range} / 판정: {status} / "
                f"하한 여유: {margin_lo:+.3f}, 상한 여유: {margin_hi:+.3f})"
            )
    kpis_str = "\n".join(kpis_lines)

    # 주요 공정 변수 분석 (Feature Importance 기반 — 저장된 경우 활용)
    fi_context = ""
    fi_all = st.session_state.get('feature_importance', {})
    if fi_all:
        top_vars_per_tgt = []
        for tgt, fi_dict in fi_all.items():
            if tgt in spec_na_list:
                continue
            top3 = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)[:3]
            top3_str = ", ".join([f"{v}({s:.3f})" for v, s in top3])
            top_vars_per_tgt.append(f"  - {tgt}: 주요 영향 변수 → {top3_str}")
        if top_vars_per_tgt:
            fi_context = "\n[Feature Importance — 타겟별 상위 영향 변수]\n" + "\n".join(top_vars_per_tgt)

    mode_desc = {
        "Optimization": "사용자가 설정한 목표 품질(타겟 값) 범위를 만족시키기 위해 역최적화 알고리즘으로 도출한 '추천 공정 스펙' 결과",
        "Simulation": "사용자가 What-If 시뮬레이터에서 가상으로 설정한 목표 품질 범위에 대해 역최적화로 도출한 '가상 시뮬레이션' 결과"
    }.get(mode, "공정 최적화 결과")

    doc_no = f"JOINT-OPS-CABJ-{datetime.now().strftime('%Y%m%d')}-001"
    today  = datetime.now().strftime("%Y년 %m월 %d일")

    system_instruction = (
        "당신은 'JOINT AI - Process Optimization Suite'에 내장된 공정 엔지니어링 전문 AI 어시스턴트입니다. "
        "이 시스템은 VOLVO SPA1/2 CABJ(볼스터드 조인트, Ball Stud Joint) 부품의 스웨이징 조립 공정 검사 데이터를 학습하여, "
        "목표 품질 타겟 값(분리 토크, 회전 토크, 축/반경방향 유격, ABAMS/RBAMS 등)를 만족하는 "
        "단품 치수 변수 조합을 역최적화로 도출합니다.\n\n"
        "보고서 작성 규칙:\n"
        "1. 아래 제공된 데이터와 무관한 일반 지식, 다른 부품으로 범위를 확장하지 마세요.\n"
        "2. 답변은 한국어로, 전문 실무형 보고서 형식으로 작성하세요. 헤더, 소제목, 표 형식을 적극 활용하세요.\n"
        "3. 한자(漢字, 중국어 한자)는 단 한 글자도 사용하지 마세요. 모든 단어는 순수 한글(및 영문 약어/숫자)로만 표기하세요.\n"
        "4. 마크다운 취소선(~~텍스트~~)은 사용하지 마세요. 강조는 굵게(**텍스트**)만 사용하세요.\n"
        "5. 각 섹션을 충분히 상세하게 작성하고, 수치 근거를 반드시 포함하세요.\n"
        "6. 분석 문서 헤더(번호, 날짜, 작성자 등)를 포함하여 정식 보고서 형태로 작성하세요."
    )

    prompt = f"""## JOINT AI - Process Optimization Suite: CABJ 스웨이징 조립 공정 가이드라인 ({mode} 결과 기반)
문서 번호: {doc_no}  작성일: {today}  작성자: JOINT AI Process Engineering Assistant

아래 데이터를 바탕으로 **상세한 공정 분석 보고서**를 작성해 주세요.

[분석 모드]
{mode} — {mode_desc}

[도출된 추천 공정 변수 스펙 (34개 단품 치수 변수)]
{specs_str}

[예측 품질 타겟 값 (10개 KPI, 스펙 적합성 포함)]
{kpis_str}
{fi_context}

[적합 타겟]: {', '.join(spec_ok_list) if spec_ok_list else '없음'}
[이탈/주의 타겟]: {', '.join(spec_warn_list) if spec_warn_list else '없음'}
[스펙 N/A (최적화 제외)]: {', '.join(spec_na_list) if spec_na_list else '없음'}

[작성 요청 — 아래 섹션을 모두 포함하여 상세히 작성하세요]

1. **결과 요약** (3~5문장)
   - 도출된 공정 스펙의 전반적 특성과 예측 KPI 달성 현황을 종합 요약하세요.
   - 특히 전체 KPI 중 몇 개가 적합/이탈인지 명시하세요.

2. **KPI 적합성 평가** (표 형식 권장)
   각 KPI에 대해: 예측값, 정상 스펙 범위, 적합성 판정, 비고(하한치 근접 여부, 외삽 위험 등)를 포함하세요.
   - 스펙 하한/상한치에 근접한 KPI는 특별히 언급하세요.
   - 모델 외삽(Extrapolation) 위험이 있는 경우 명시하세요.

3. **주의가 필요한 공정 변수** (핵심 3~5개)
   Feature Importance 또는 공정 지식에 근거하여 가장 중요한 변수를 선별하고,
   각 변수가 품질에 미치는 물리적 영향 메커니즘을 설명하세요.
   (예: COHB/COHA → 스웨이징 깊이 → 조인트 체결력 → BT/RT/ABAMS에 영향)

4. **현장 적용 권장사항** (3가지 이내, 구체적 수치 포함)
   도출된 최적화 결과를 실제 생산 라인에 적용할 때의 단계별 조치사항을 작성하세요.
   - 스웨이징 공정 정밀도 관리 우선순위
   - 핵심 단품(볼스터드, 시트, 베어링) 치수 공차 관리 강화 사항
   - KPI 실시간 모니터링 및 피드백 방안"""

    try:
        priority = ['llama-3.3-70b-versatile', 'llama-3.1-70b-versatile', 'llama-3.1-8b-instant', 'gemma2-9b-it']
        target_model = priority[0]
        try:
            available_models = [m.id for m in client.models.list().data]
            target_model = next((m for m in priority if m in available_models), (available_models[0] if available_models else priority[0]))
        except Exception:
            pass

        response = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.25,
            max_tokens=4096
        )
        result_text = response.choices[0].message.content
        result_text = re.sub(r'~~(.*?)~~', r'\1', result_text)
        result_text = re.sub(r'[\u4e00-\u9fff]+', '', result_text)
        result_text = re.sub(r'[ \t]{2,}', ' ', result_text)
        return result_text

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg: return "⏳ API 사용량이 많습니다. 잠시 후 다시 시도해 주세요."
        return f"❌ AI 생성 오류: {error_msg}"

# ────────────────────────────────────────────────────────────────────────────────
# [추가] 공정 진단 가이드 생성 함수 (유효 타겟만 적용 및 정상 시에도 리포트 반환)
# ────────────────────────────────────────────────────────────────────────────────
def generate_diagnosis_guide(feasibility_info, predicted_kpis, opt_result_x, confidence_score, range_key_prefix=''):
    is_en = st.session_state.get('lang', 'KO') == 'EN'
    infeasible_targets = []
    partial_targets   = []
    ok_targets        = []
    
    # 데이터가 존재하는 타겟만 추출
    valid_tgts = st.session_state.get('valid_target_vars', target_vars)
    if not valid_tgts:
        valid_tgts = target_vars

    for tgt in valid_tgts:
        pred_val = predicted_kpis.get(tgt, None)
        if pred_val is None:
            continue
        t_range = st.session_state.get(f'{range_key_prefix}{tgt.lower()}_s_val', (0, 1))
        feas    = feasibility_info.get(tgt, {})
        overlap = feas.get('overlap_ratio', 1.0)
        pred_min = feas.get('pred_min', 0)
        pred_max = feas.get('pred_max', 0)

        if overlap < 0.1:
            infeasible_targets.append({
                'tgt': tgt, 'pred_val': pred_val,
                'spec': t_range, 'pred_min': pred_min, 'pred_max': pred_max
            })
        elif pred_val < t_range[0] or pred_val > t_range[1]:
            partial_targets.append({
                'tgt': tgt, 'pred_val': pred_val, 'spec': t_range
            })
        else:
            ok_targets.append(tgt)

    lines = []
    if is_en:
        lines.append("## □ Optimization Result Diagnosis Report\n")
        lines.append(f"> Prediction Confidence: **{confidence_score}%** | Infeasible: **{len(infeasible_targets)}** | Out-of-Spec: **{len(partial_targets)}** | Achieved: **{len(ok_targets)}**\n")
    else:
        lines.append("## □ 최적화 결과 진단 리포트\n")
        lines.append(f"> 예측 신뢰도: **{confidence_score}%** | 달성 불가: **{len(infeasible_targets)}개** | 스펙 이탈: **{len(partial_targets)}개** | 정상 달성: **{len(ok_targets)}개**\n")

    # 모든 타겟이 정상일 경우에도 보고서를 반환하도록 수정
    if not infeasible_targets and not partial_targets:
        if is_en:
            lines.append("\n---\n### ✅ All Targets Achieved\n")
            lines.append("Based on the current input data, all valid quality targets are predicted to fully reach the specified goal range. The current recommended process specification can be applied on the shop floor as-is.\n")
        else:
            lines.append("\n---\n### ✅ 전체 타겟 정상 달성\n")
            lines.append("현재 입력된 데이터를 바탕으로 분석한 결과, 모든 유효 품질 타겟(타겟 값)이 설정하신 목표 스펙 범위 내에 완벽하게 도달할 수 있는 것으로 예측되었습니다. 현재 도출된 공정 조건(추천 공정 스펙)을 현장에 바로 적용하셔도 좋습니다.\n")
        return "\n".join(lines)

    fi_by_target = {}
    for tgt in (infeasible_targets + partial_targets):
        t = tgt['tgt']
        mk = f'model_{t.lower()}'
        mdl = st.session_state.get(mk)
        if mdl is not None and hasattr(mdl, 'feature_importances_'):
            fi = pd.Series(mdl.feature_importances_, index=X_list).sort_values(ascending=False)
            fi_by_target[t] = fi.head(6) 

    if infeasible_targets:
        if is_en:
            lines.append("\n---\n### 🚫 Infeasible Target Analysis\n")
            lines.append(
                "The targets below have **no training samples within the spec range at all**, so the AI cannot predict this region.  \n"
                "This is not a code error — it is a **physical fact that the process itself has never achieved this spec**.\n"
            )
        else:
            lines.append("\n---\n### 🚫 달성 불가 타겟 분석\n")
            lines.append(
                "아래 타겟들은 **학습 데이터에 스펙 범위 내 샘플이 단 한 개도 없어** AI가 예측할 수 없는 영역입니다.  \n"
                "이는 AI 코드 오류가 아니라 **공정 자체가 해당 스펙을 달성한 적이 없다는 물리적 사실**입니다.\n"
            )
        for item in infeasible_targets:
            tgt      = item['tgt']
            glossary = TARGET_GLOSSARY.get(tgt, tgt)
            gap      = item['pred_min'] - item['spec'][1] 

            if is_en:
                lines.append(f"\n#### [{tgt}] {glossary}")
                lines.append(f"- Predicted range from training data: **{item['pred_min']:.2f} ~ {item['pred_max']:.2f}**")
                lines.append(f"- Target spec: {item['spec'][0]} ~ {item['spec'][1]}")
                lines.append(f"- GAP between current process and spec: **{gap:.2f}** (experimental data is needed to close this gap)")
            else:
                lines.append(f"\n#### [{tgt}] {glossary}")
                lines.append(f"- 학습 데이터 예측 범위: **{item['pred_min']:.2f} ~ {item['pred_max']:.2f}**")
                lines.append(f"- 설정 스펙: {item['spec'][0]} ~ {item['spec'][1]}")
                lines.append(f"- 현재 공정과 스펙의 GAP: **{gap:.2f}** (이 gap을 메울 실험 데이터가 필요)")

            if tgt in fi_by_target:
                if is_en:
                    lines.append(f"\n** □ Top Influential Variables for {tgt} Prediction (Feature Importance based)**\n")
                    lines.append("| Rank | Variable | Meaning | Importance | Recommended Direction |")
                    lines.append("|------|------|------|--------|----------------|")
                else:
                    lines.append(f"\n** □ {tgt} 예측에 영향력 상위 변수 (Feature Importance 기반)**\n")
                    lines.append("| 순위 | 변수 | 의미 | 중요도 | 조정 방향 권장 |")
                    lines.append("|------|------|------|--------|----------------|")
                for rank, (var, imp) in enumerate(fi_by_target[tgt].items(), 1):
                    meaning = VAR_GLOSSARY.get(var, var)
                    if is_en:
                        if tgt in ['RBAMS', 'RGB', 'RGA', 'RGI']:
                            if var in ['CID']:     direction = "↓ Decrease (reduce inner diameter → reduce clearance)"
                            elif var in ['BD']:    direction = "↑ Increase (enlarge ball diameter → reduce clearance)"
                            elif var in ['COHB', 'COHA']: direction = "↓ Decrease (increase swaging press-in)"
                            elif var in ['SID']:   direction = "↓ Decrease (reduce seat inner diameter)"
                            elif var in ['SOD']:   direction = "↑ Increase (enlarge seat outer diameter)"
                            elif var in ['CR']:    direction = "↓ Decrease (improve roundness)"
                            else:                  direction = "Direction to be confirmed by experiment"
                        elif tgt in ['ABAMS', 'AGB', 'AGA', 'AGI']:
                            if var in ['CIH']:     direction = "↓ Decrease (reduce inner height → reduce axial clearance)"
                            elif var in ['BH']:    direction = "↑ Increase (increase bearing height)"
                            elif var in ['CITH']:  direction = "Adjust (taper height → axial load distribution)"
                            elif var in ['CH1','CH2','CH3','CH4']: direction = "Adjust (review case height dimensions)"
                            else:                  direction = "Direction to be confirmed by experiment"
                        elif tgt == 'BT':
                            direction = "Prioritize increasing swaging press-in amount (COHB→COHA)"
                        elif tgt == 'RT':
                            direction = "Review seat R dimensions (SR, SR2) and seat height adjustment"
                        else:
                            direction = "Direction to be confirmed by experiment"
                    else:
                        if tgt in ['RBAMS', 'RGB', 'RGA', 'RGI']:
                            if var in ['CID']:     direction = "↓ 감소 (내경 축소 → 유격 감소)"
                            elif var in ['BD']:    direction = "↑ 증가 (볼 직경 확대 → 유격 감소)"
                            elif var in ['COHB', 'COHA']: direction = "↓ 감소 (스웨이징 압입 증가)"
                            elif var in ['SID']:   direction = "↓ 감소 (시트 내경 축소)"
                            elif var in ['SOD']:   direction = "↑ 증가 (시트 외경 확대)"
                            elif var in ['CR']:    direction = "↓ 감소 (진원도 향상)"
                            else:                  direction = "실험으로 방향 확인 필요"
                        elif tgt in ['ABAMS', 'AGB', 'AGA', 'AGI']:
                            if var in ['CIH']:     direction = "↓ 감소 (내측 높이 축소 → 축방향 유격 감소)"
                            elif var in ['BH']:    direction = "↑ 증가 (베어링 높이 증가)"
                            elif var in ['CITH']:  direction = "조정 (테이퍼 높이 → 축방향 하중 분포)"
                            elif var in ['CH1','CH2','CH3','CH4']: direction = "조정 (케이스 높이 치수 검토)"
                            else:                  direction = "실험으로 방향 확인 필요"
                        elif tgt == 'BT':
                            direction = "스웨이징 압입량(COHB→COHA) 증가 방향 우선 검토"
                        elif tgt == 'RT':
                            direction = "시트 R 치수(SR, SR2) 및 시트 높이 조정 검토"
                        else:
                            direction = "실험으로 방향 확인 필요"
                    lines.append(f"| {rank} | **{var}** | {meaning} | {imp:.4f} | {direction} |")

    if partial_targets:
        if is_en:
            lines.append("\n---\n### ⚠️ Out-of-Spec Target Analysis\n")
            lines.append("The targets below fall within the training data range but did not reach the target spec.  \nThere is potential for improvement via the recommended variable adjustments below, even without additional data.\n")
        else:
            lines.append("\n---\n### ⚠️ 스펙 이탈 타겟 분석\n")
            lines.append("아래 타겟들은 학습 데이터 범위 안에 있으나 목표 스펙에 미달했습니다.  \n추가 데이터 없이도 아래 권장 변수 조정으로 개선 가능성이 있습니다.\n")
        for item in partial_targets:
            tgt = item['tgt']
            if is_en:
                lines.append(f"\n#### [{tgt}] {TARGET_GLOSSARY.get(tgt, tgt)}")
                lines.append(f"- Predicted value: **{item['pred_val']:.3f}** / Spec: {item['spec'][0]} ~ {item['spec'][1]}")
            else:
                lines.append(f"\n#### [{tgt}] {TARGET_GLOSSARY.get(tgt, tgt)}")
                lines.append(f"- 예측값: **{item['pred_val']:.3f}** / 스펙: {item['spec'][0]} ~ {item['spec'][1]}")
            if tgt in fi_by_target:
                top3 = list(fi_by_target[tgt].index[:3])
                if is_en:
                    lines.append(f"- Priority variables to review: **{', '.join(top3)}**")
                else:
                    lines.append(f"- 우선 조정 검토 변수: **{', '.join(top3)}**")

    if is_en:
        lines.append("\n---\n### 🗺️ Recommended Resolution Path\n")
    else:
        lines.append("\n---\n### 🗺️ 권장 해결 경로\n")

    if infeasible_targets:
        if is_en:
            lines.append("#### Path A — Acquire New Experimental Data (Fundamental Solution)\n")
            lines.append(
                "Since the current process conditions have no history of achieving this spec, you should **deliberately design trials that push the process variables toward the spec direction** to acquire data.\n"
            )
        else:
            lines.append("#### 경로 A — 신규 실험 데이터 확보 (근본 해결)\n")
            lines.append(
                "현재 공정 조건으로는 스펙 달성 이력이 없으므로, **의도적으로 공정 변수를 스펙 방향으로 밀어붙인 시험**을 설계해 데이터를 확보해야 합니다.\n"
            )
        all_priority_vars = []
        for tgt_dict in infeasible_targets:
            t = tgt_dict['tgt']
            if t in fi_by_target:
                all_priority_vars += list(fi_by_target[t].index[:3])
        priority_vars_unique = list(dict.fromkeys(all_priority_vars)) 

        if priority_vars_unique:
            if is_en:
                lines.append(f"\n**Priority experiment variables based on Feature Importance (combined):** {', '.join(priority_vars_unique[:5])}\n")
            else:
                lines.append(f"\n**Feature Importance 기반 실험 우선 변수 (통합):** {', '.join(priority_vars_unique[:5])}\n")

        if is_en:
            lines.append("**Recommended Design of Experiments (DOE) Sequence:**\n")
            lines.append("1. **Screening trial** (variable reduction): Test the priority variables above with a Plackett-Burman design over 12-16 samples → narrow down to variables that actually affect RBAMS/ABAMS")
            lines.append("2. **Focused trial on key variables** (range exploration): Deliberately extend 2-3 selected variables to ±3σ beyond the current range to explore the spec-achieving region (CCD or Box-Behnken design)")
            lines.append("3. **Confirmation trial**: Repeat the spec-achieving condition at least 5 times to confirm reproducibility")
            lines.append(f"\n> Minimum required samples: 12-16 for screening + 20-30 for focused exploration = **approximately 40-50 total**\n")

            lines.append("#### Path B — Re-examine Spec Feasibility (Short-Term Alternative)\n")
            lines.append(
                "If the current production parts are used in the field without performance issues even at RBAMS 60-75, **the spec of 0-25 may be a design spec that does not match reality**.  \n"
                "- Check the correlation between actual durability/noise/clearance test results and the RBAMS value.  \n"
                "- If performance criteria are met, it is more realistic to revise the spec to the actual achievable production range (e.g., 50-80).  \n"
                "- Once the spec is revised, this AI system will work normally immediately.\n"
            )
        else:
            lines.append("**권장 실험 설계(DOE) 순서:**\n")
            lines.append("1. **스크리닝 실험** (변수 압축): 위 우선 변수를 Plackett-Burman 설계로 12~16샘플 시험 → RBAMS/ABAMS에 실제로 영향 있는 변수 압축")
            lines.append("2. **핵심 변수 집중 실험** (범위 탐색): 선별된 2~3개 변수로 현재 범위 ±3σ 밖까지 의도적으로 확장해 스펙 달성 구간 탐색 (CCD 또는 Box-Behnken 설계)")
            lines.append("3. **확인 실험**: 스펙 달성 조건을 최소 5회 이상 반복해 재현성 확인")
            lines.append(f"\n> 최소 필요 샘플 수: 스크리닝 12~16개 + 핵심 탐색 20~30개 = **총 약 40~50개**\n")

            lines.append("#### 경로 B — 스펙 현실성 재검토 (단기 대안)\n")
            lines.append(
                "현재 생산 제품이 RBAMS 60~75임에도 현장에서 성능 문제 없이 사용 중이라면, **스펙 0~25가 현실과 맞지 않는 설계 스펙**일 수 있습니다.  \n"
                "- 실제 내구/소음/유격 성능 시험 결과와 RBAMS 값의 상관관계를 확인하세요.  \n"
                "- 성능 기준을 만족한다면 스펙을 실제 생산 가능 범위(예: 50~80)로 개정하는 것이 현실적입니다.  \n"
                "- 스펙 개정 시 이 AI 시스템은 즉시 정상 동작합니다.\n"
            )

    if is_en:
        lines.append("#### Path C — Model Retraining (After Acquiring Data)\n")
        lines.append(
            "After acquiring experimental data, retrain in the following order.\n"
            "1. **Merge** new experimental data with the existing training data (CSV or DB upload)\n"
            "2. Retrain the model using the sidebar **'Engine Init'** button\n"
            "3. Re-run the inverse optimization with the retrained model → verify whether the spec is achievable\n"
        )

        lines.append("\n---\n### 📋 Summary\n")
        lines.append("| Step | Content | Expected Duration |")
        lines.append("|------|------|-----------|")
        lines.append("| 1 | Review Feature Importance → select candidate experiment variables | Immediate (see this report) |")
        lines.append("| 2 | Review spec feasibility (discuss with design team) | 1-2 days |")
        lines.append("| 3 | DOE screening trial (12-16 samples) | 1-2 weeks |")
        lines.append("| 4 | Focused trial on key variables (20-30 samples) | 2-4 weeks |")
        lines.append("| 5 | Merge data, retrain and validate model | 1 day |")
    else:
        lines.append("#### 경로 C — 모델 재학습 (데이터 확보 후)\n")
        lines.append(
            "실험 데이터 확보 후 아래 순서로 재학습하세요.\n"
            "1. 신규 실험 데이터를 기존 학습 데이터에 **병합** (CSV 또는 DB 업로드)\n"
            "2. 사이드바 **'엔진 초기화'** 버튼으로 모델 재학습\n"
            "3. 재학습된 모델로 역최적화 재실행 → 스펙 달성 가능 여부 확인\n"
        )

        lines.append("\n---\n### 📋 요약\n")
        lines.append("| 단계 | 내용 | 예상 기간 |")
        lines.append("|------|------|-----------|")
        lines.append("| 1 | Feature Importance 확인 → 실험 변수 후보 선정 | 즉시 (이 리포트 참조) |")
        lines.append("| 2 | 스펙 현실성 검토 (설계팀 협의) | 1~2일 |")
        lines.append("| 3 | DOE 스크리닝 실험 (12~16샘플) | 1~2주 |")
        lines.append("| 4 | 핵심 변수 집중 실험 (20~30샘플) | 2~4주 |")
        lines.append("| 5 | 데이터 병합 후 모델 재학습 및 검증 | 1일 |")

    return "\n".join(lines)

    init_dict = {
        'scaler': None, 'df_caulking': pd.DataFrame(),
        'process_vars': X_list, 'target_vars': target_vars, 'active_x_list': X_list,
        'optimizer_status': "STANDBY", 'opt_result_x': None, 'confidence_score': None, 'sim_confidence': None,
        'best_algorithm_used': "SLSQP", 'sim_result_x': None,
        'prepared_db_file': None, 'data_changed_since_save': False,
        'ai_analysis_result': None,
        'valid_target_vars': target_vars,
        'diagnosis_guide_text': None,
        'feasibility': {} 
    }
    for tgt in target_vars:
        init_dict[f'model_{tgt.lower()}'] = None
        init_dict[f'opt_pred_{tgt.lower()}'] = None
        init_dict[f'sim_pred_{tgt.lower()}'] = None
    for var in X_list:
        init_dict[f'm_{var.lower()}_min'] = 0.0
        init_dict[f'm_{var.lower()}_max'] = 100.0
        init_dict[f'sim_{var.lower()}'] = 0.0
        
    for tgt in target_vars:
        t_low = tgt.lower()
        if tgt == 'BT': range_val = (0.0, 8.0)
        elif tgt == 'RT': range_val = (0.0, 4.0)
        elif tgt == 'AGB': range_val = (0.0, 0.3)
        elif tgt == 'RGB': range_val = (0.0, 0.4)    
        elif tgt in ['ABAMS', 'RBAMS']: range_val = (25.0, 100.0) 
        else: range_val = (0.0, 1.0)
        
        init_dict[f'{t_low}_s_val'] = range_val
        init_dict[f'{t_low}_n_min'] = range_val[0]
        init_dict[f'{t_low}_n_max'] = range_val[1]
        
        init_dict[f'sim_tgt_{t_low}_s_val'] = range_val
        init_dict[f'sim_tgt_{t_low}_n_min'] = range_val[0]
        init_dict[f'sim_tgt_{t_low}_n_max'] = range_val[1]
        
    st.session_state.update(init_dict)

# 6. 사이드바 제어반
with st.sidebar:
    st.markdown(f"<h2 style='color: #ffffff; font-size:1.15rem; margin-bottom: 20px;'>{L_G['console']}</h2>", unsafe_allow_html=True)

    with st.expander(L_G['upload_title'], expanded=True):
        u_input = st.file_uploader(L_G['upload_help'], type=['csv','xlsx','db'], key="new_data_file")
        db_input = st.file_uploader(L_G['upload_hist_help'], type=['db'], key="history_db_file")

    if st.button(L_G['init_btn'], type="primary"):
        if u_input or db_input:
            data_frames = []
            spec_from_file = {}   # [추가] 파일에서 읽은 타겟 스펙 저장

            if u_input:
                try:
                    if u_input.name.endswith('.db'):
                        temp_db = "temp_uploaded_joint.db"
                        with open(temp_db, "wb") as t: t.write(u_input.getvalue())
                        conn = sqlite3.connect(temp_db)
                        df_temp = pd.read_sql_query("SELECT vars FROM production_log", conn)
                        conn.close()
                        if os.path.exists(temp_db): os.remove(temp_db)
                        df_new = pd.json_normalize([json.loads(x) for x in df_temp['vars']])
                    elif u_input.name.endswith('csv'):
                        # [수정] 1행=변수명, 2행=스펙값, 3행~=데이터
                        df_spec_row = pd.read_csv(u_input, header=0, nrows=1,
                                                   na_values=['N/A','n/a','NA','null','-',''])
                        # 스펙 행에서 타겟 범위 파싱 ("0.0 ~ 8.0" 형식)
                        for tgt in target_vars:
                            if tgt in df_spec_row.columns:
                                raw = str(df_spec_row[tgt].iloc[0]).strip()
                                if '~' in raw:
                                    try:
                                        parts = raw.split('~')
                                        lo = float(parts[0].strip())
                                        hi = float(parts[1].strip())
                                        spec_from_file[tgt] = (lo, hi)
                                    except: pass
                        u_input.seek(0)
                        # 3행부터 실제 데이터로 읽기 (2행 스펙 행 건너뜀)
                        df_new = pd.read_csv(u_input, header=0, skiprows=[1],
                                             na_values=['N/A','n/a','NA','N/A ','-','null'])
                    else:
                        df_spec_row = pd.read_excel(u_input, header=0, nrows=1,
                                                     na_values=['N/A','n/a','NA','null','-',''])
                        for tgt in target_vars:
                            if tgt in df_spec_row.columns:
                                raw = str(df_spec_row[tgt].iloc[0]).strip()
                                if '~' in raw:
                                    try:
                                        parts = raw.split('~')
                                        lo = float(parts[0].strip())
                                        hi = float(parts[1].strip())
                                        spec_from_file[tgt] = (lo, hi)
                                    except: pass
                        u_input.seek(0)
                        df_new = pd.read_excel(u_input, header=0, skiprows=[1],
                                               na_values=['N/A','n/a','NA','N/A ','-','null'])
                    data_frames.append(df_new)
                    pass  # 스펙 적용/미적용 분류 표시 제거
                except Exception as e:
                    st.sidebar.error(f"신규 파일 로드 오류: {e}")

            if db_input:
                try:
                    temp_db_hist = "temp_uploaded_hist.db"
                    with open(temp_db_hist, "wb") as t: t.write(db_input.getvalue())
                    conn = sqlite3.connect(temp_db_hist)
                    df_temp_hist = pd.read_sql_query("SELECT vars FROM production_log", conn)
                    conn.close()
                    if os.path.exists(temp_db_hist): os.remove(temp_db_hist)
                    df_hist = pd.json_normalize([json.loads(x) for x in df_temp_hist['vars']])
                    data_frames.append(df_hist)
                except Exception as e:
                    st.sidebar.error(f"기존 DB 파일 로드 오류: {e}")

            df_comb = None
            if data_frames:
                df_comb = pd.concat(data_frames, ignore_index=True)
                if 'vars' in df_comb.columns: df_comb = df_comb.drop(columns=['vars'], errors='ignore')

            if df_comb is not None:
                df_comb.columns = [c.strip() for c in df_comb.columns]
                
                valid_targets = []
                for tgt in target_vars:
                    if tgt in df_comb.columns:
                        converted = pd.to_numeric(df_comb[tgt], errors='coerce')
                        if not converted.isna().all(): valid_targets.append(tgt)
                
                if not valid_targets: valid_targets = ['BT']
                
                for col in X_list + target_vars:
                    if col in df_comb.columns:
                        df_comb[col] = pd.to_numeric(df_comb[col], errors='coerce')
                    else:
                        df_comb[col] = np.nan
                
                df_imputed = df_comb.copy()
                for var in X_list:
                    df_imputed[var] = df_imputed[var].fillna(df_imputed[var].median()) if not df_imputed[var].isna().all() else 0.0
                for tgt in target_vars:
                    df_imputed[tgt] = df_imputed[tgt].fillna(df_imputed[tgt].median()) if not df_imputed[tgt].isna().all() else 0.0
                
                scaler = MinMaxScaler().fit(df_imputed[X_list])
                
                models = {}
                model_metadata = {}
                fi_dict = {}
                train_prog = st.sidebar.progress(0)
                algo_status = st.sidebar.empty()
                total_targets_n = len(target_vars)

                X_scaled_all = scaler.transform(df_imputed[X_list])

                _is_train_en2 = st.session_state.get('lang','KO')=='EN'
                for t_idx, target in enumerate(target_vars):
                    pct = t_idx / total_targets_n
                    _pct_txt = (f"⚙️ ({t_idx+1}/{total_targets_n}) Selecting algorithm for {target}..." if _is_train_en2
                                else f"⚙️ ({t_idx+1}/{total_targets_n}) {target} 알고리즘 선택 중...")
                    train_prog.progress(pct, text=_pct_txt)
                    _search_txt = (f"▸ Searching optimal algorithm → <b style='color:#38bdf8;'>{target}</b>" if _is_train_en2
                                    else f"▸ 최적 알고리즘 탐색 중 → <b style='color:#38bdf8;'>{target}</b>")
                    algo_status.markdown(
                        f"<div style='background:#060e1a;border-left:3px solid #38bdf8;border-radius:5px;padding:5px 10px;font-size:0.75rem;color:#cbd5e1;'>{_search_txt}</div>",
                        unsafe_allow_html=True
                    )
                    y_t = df_imputed[target]

                    # 자동 알고리즘 선택
                    best_model, best_name, cv_r2 = auto_select_best_model(X_scaled_all, y_t.values)
                    best_model.fit(X_scaled_all, y_t)
                    r2 = _r2_score(y_t.values, best_model.predict(X_scaled_all))

                    models[f'model_{target.lower()}'] = best_model
                    model_metadata[f'algo_{target.lower()}'] = f"{best_name} (R²={r2:.3f})"

                    # Feature Importance 추출
                    if hasattr(best_model, 'feature_importances_'):
                        fi_dict[target] = dict(zip(X_list, best_model.feature_importances_.tolist()))
                    elif hasattr(best_model, 'coef_'):
                        fi_dict[target] = dict(zip(X_list, np.abs(best_model.coef_).tolist()))
                    else:
                        fi_dict[target] = {v: 0.0 for v in X_list}

                    _is_train_en = st.session_state.get('lang','KO')=='EN'
                    _train_txt = (f"✓ {target} → <b>{best_name}</b> selected (R²={r2:.3f})" if _is_train_en
                                   else f"✓ {target} → <b>{best_name}</b> 선택 완료 (R²={r2:.3f})")
                    algo_status.markdown(
                        f"<div style='background:#060e1a;border-left:3px solid #10b981;border-radius:5px;padding:5px 10px;font-size:0.75rem;color:#a3e635;'>{_train_txt}</div>",
                        unsafe_allow_html=True
                    )

                train_prog.progress(1.0, text=("✅ All targets trained!" if st.session_state.get('lang','KO')=='EN' else "✅ 모든 타겟 학습 완료!"))
                algo_status.empty()
                st.session_state['model_metadata'] = model_metadata
                st.session_state['feature_importance'] = fi_dict
                st.session_state['df_imputed_ref'] = df_imputed.copy()
                
                data_bounds = {}
                update_data = {
                    'scaler': scaler, 'df_caulking': df_comb, 'optimizer_status': "ENGINE READY",
                    'prepared_db_file': None, 'data_changed_since_save': True,
                    'valid_target_vars': valid_targets
                }
                for tgt in target_vars: update_data[f'model_{tgt.lower()}'] = models[f'model_{tgt.lower()}']
                
                for var in X_list:
                    v_min = float(df_imputed[var].min())
                    v_max = float(df_imputed[var].max())
                    if v_min == v_max: v_max += 1.0
                    data_bounds[var] = (v_min, v_max)
                    update_data[f'm_{var.lower()}_min'] = v_min
                    update_data[f'm_{var.lower()}_max'] = v_max
                    update_data[f'sim_{var.lower()}'] = float(df_imputed[var].median()) 
                    
                update_data['data_bounds'] = data_bounds
                
                for tgt in target_vars:
                    t_low = tgt.lower()
                    # [수정] 파일에서 읽은 스펙 우선 적용, 없으면 기본값
                    if tgt in spec_from_file:
                        range_val = spec_from_file[tgt]
                    elif tgt == 'BT': range_val = (0.0, 8.0)
                    elif tgt == 'RT': range_val = (0.0, 4.0)
                    elif tgt == 'AGB': range_val = (0.0, 0.3)
                    elif tgt == 'RGB': range_val = (0.0, 0.4)
                    elif tgt in ['ABAMS', 'RBAMS']: range_val = (25.0, 100.0)
                    else: range_val = (0.0, 1.0)

                    # 자동 모드 + 수동 모드 모두 동일하게 적용
                    update_data[f'{t_low}_s_val']          = range_val
                    update_data[f'{t_low}_n_min']           = range_val[0]
                    update_data[f'{t_low}_n_max']           = range_val[1]
                    update_data[f'sim_tgt_{t_low}_s_val']  = range_val
                    update_data[f'sim_tgt_{t_low}_n_min']  = range_val[0]
                    update_data[f'sim_tgt_{t_low}_n_max']  = range_val[1]

                # 파일 스펙을 세션에 저장 (나중에 spec_limits 업데이트)
                if spec_from_file:
                    update_data['spec_from_file'] = spec_from_file
                    na_tgts = [t for t in target_vars if t not in spec_from_file]
                    update_data['na_spec_targets'] = na_tgts
                else:
                    na_tgts = []
                    update_data['na_spec_targets'] = []
                # N/A 타겟은 리셋 플래그 설정 → 첫 렌더 시 데이터 범위로 초기화
                for _nt in na_tgts:
                    update_data[f'_na_reset_{_nt.lower()}'] = True

                # X_list 변수 스펙 파싱 (Auto Mode Range 참조용)
                _xs_range_dict = {}
                _na_x_list = []
                try:
                    _u = u_input
                    if _u and not _u.name.endswith('.db'):
                        _u.seek(0)
                        if _u.name.endswith('csv'):
                            _xspec_row = pd.read_csv(_u, header=0, nrows=1,
                                                     na_values=['N/A','n/a','NA','null','-',''])
                        else:
                            _xspec_row = pd.read_excel(_u, header=0, nrows=1,
                                                       na_values=['N/A','n/a','NA','null','-',''])
                        for _xv in X_list:
                            if _xv in _xspec_row.columns:
                                _raw_x = str(_xspec_row[_xv].iloc[0]).strip()
                                if '~' in _raw_x:
                                    try:
                                        _px = _raw_x.split('~')
                                        _xlo, _xhi = float(_px[0].strip()), float(_px[1].strip())
                                        _xs_range_dict[_xv] = (_xlo, _xhi)
                                    except: _na_x_list.append(_xv)
                                else:
                                    _na_x_list.append(_xv)
                except: pass
                update_data['x_spec_parsed_dict'] = _xs_range_dict
                update_data['na_x_vars'] = _na_x_list

                st.session_state.update(update_data)
                st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<h3 style='color:#00e5ff; font-size:1.1rem;'>{L_G['db_export_title']}</h3>", unsafe_allow_html=True)
    
    if not st.session_state.get('df_caulking', pd.DataFrame()).empty:
        if st.sidebar.button(L_G['db_prepare_btn'], key="btn_create_db_snapshot"):
            today_str = datetime.now().strftime("%Y%m%d")
            if st.session_state.get('data_changed_since_save', True) or st.session_state['prepared_db_file'] is None:
                idx = 1
                while True:
                    candidate = f"joint-{today_str}-{idx}.db"
                    if not os.path.exists(candidate):
                        final_filename = candidate
                        break
                    idx += 1
                st.session_state['prepared_db_file'] = final_filename
            else: final_filename = st.session_state['prepared_db_file']
            
            try:
                existing_df = pd.DataFrame()
                if os.path.exists(final_filename):
                    try:
                        conn_old = sqlite3.connect(final_filename)
                        df_old_raw = pd.read_sql_query("SELECT vars FROM production_log", conn_old)
                        conn_old.close()
                        existing_df = pd.json_normalize([json.loads(x) for x in df_old_raw['vars']])
                    except Exception: pass

                df_to_save = st.session_state['df_caulking'].copy()
                if 'vars' in df_to_save.columns: df_to_save = df_to_save.drop(columns=['vars'], errors='ignore')
                if not existing_df.empty: df_to_save = pd.concat([existing_df, df_to_save], ignore_index=True)

                conn = sqlite3.connect(final_filename)
                df_to_save['vars'] = df_to_save.apply(lambda row: json.dumps(row.to_dict()), axis=1)
                df_to_save[['vars']].to_sql("production_log", conn, if_exists="replace", index=False)
                conn.close()
                st.session_state['data_changed_since_save'] = False
            except Exception as e: st.sidebar.error(f"Error: {e}")
        
        if st.session_state.get('prepared_db_file'):
            target_file = st.session_state['prepared_db_file']
            if os.path.exists(target_file):
                try:
                    with open(target_file, "rb") as f: db_bytes = f.read()
                    if not st.session_state.get('data_changed_since_save', True):
                        st.sidebar.markdown(f"<span style='color:#a3e635; font-size:0.85rem;'>{L_G['db_current_latest']}</span>", unsafe_allow_html=True)
                    st.sidebar.markdown(f"✅ {L_G['db_prepared_msg']} `{target_file}`")
                    st.sidebar.download_button(label=L_G['db_pc_download'], data=db_bytes, file_name=target_file, mime="application/x-sqlite3", key="db_final_download_action")
                except Exception as e: st.sidebar.error(f"File Load Error: {e}")
    else:
        st.sidebar.warning(L_G['db_save_empty'])

# 7. 메인 뷰포트
if st.session_state.get('scaler') is not None:
    db = st.session_state['data_bounds']
    valid_tgts = st.session_state['valid_target_vars']
    
    st.markdown(f"<h1 style='margin-bottom:20px; font-size:1.8rem;'>{L_G['title']}</h1>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs([L_G['tab1'], L_G['tab2'], L_G['tab3'], L_G['tab4']])

    with tab1:
        layout_l, layout_r = st.columns([1.3, 1.2], gap="large")
        with layout_l:
            st.markdown(f"<div class='glass-card'><div class='glass-card-title'>{L_G['bound_title']}</div>", unsafe_allow_html=True)
            st.markdown("<style>.stRadio label span { color: #e2e8f0 !important; font-weight: 600 !important; font-size: 0.92rem !important; } .stRadio [data-testid='stMarkdownContainer'] p { color: #e2e8f0 !important; } .stRadio div[role='radiogroup'] label { color: #e2e8f0 !important; }</style>", unsafe_allow_html=True)
            # Auto Mode only (Manual Expert Tuning 삭제)
            bound_mode = "Auto Mode"
            
            chosen_bounds = {}
            if "Auto Mode" in bound_mode:
                _sf = st.session_state.get('spec_from_file', {})
                _x_spec = st.session_state.get('x_spec_parsed_dict', {})
                # CSV 2행 스펙 → 없으면 데이터 실측 Range
                for v in X_list:
                    if v in _x_spec:
                        chosen_bounds[v] = _x_spec[v]
                    else:
                        chosen_bounds[v] = db[v]
                # 표시 텍스트
                applied_v  = [v for v in X_list if v in _x_spec]
                fallback_v = [v for v in X_list if v not in _x_spec]
                _is1 = st.session_state.get('lang', 'KO') == 'EN'
                bound_text = ""
                for v in X_list:
                    rng = chosen_bounds[v]
                    if _is1:
                        src_lbl = "Spec" if v in _x_spec else "Data"
                    else:
                        src_lbl = "스펙" if v in _x_spec else "데이터"
                    bound_text += f"• {v}: {rng[0]:.3f}~{rng[1]:.3f} <span style='color:#64748b;font-size:0.75rem;'>({src_lbl})</span><br>"
                if _is1:
                    _bound_hdr = "[Design/Process Variable Range — CSV Row-2 Spec Applied]"
                    _bound_sub = " Falls back to data min~max if no spec"
                else:
                    _bound_hdr = "[설계/공정 변수 Range — CSV 2행 스펙 적용]"
                    _bound_sub = " 스펙없으면 데이터 min~max"
                st.markdown(
                    f"<div style='background:#0f172a;padding:12px 15px;border-radius:6px;border:1px solid #1e293b;"
                    f"font-size:0.85rem;line-height:1.6;max-height:200px;overflow-y:auto;'>"
                    f"<span style='color:#38bdf8;font-weight:600;'>{_bound_hdr}</span>"
                    f"<span style='color:#64748b;font-size:0.75rem;'>{_bound_sub}</span><br>"
                    f"{bound_text}</div>",
                    unsafe_allow_html=True
                )
            else:
                _na_x_manual = st.session_state.get('na_x_vars', [])
                _xs_dict_manual = st.session_state.get('x_spec_parsed_dict', {})
                st.markdown("<div style='max-height:400px; overflow-y:auto; padding-right:10px;'>", unsafe_allow_html=True)
                for v in X_list:
                    v_clean = v.lower()
                    meaning = VAR_GLOSSARY.get(v, v)
                    db_min  = float(db[v][0] * 0.5)
                    db_max  = float(db[v][1] * 1.5)
                    step_v  = round(float((db[v][1] - db[v][0]) * 0.005), 5)
                    step_v  = max(step_v, 0.001)
                    _is_na_x = v in _na_x_manual

                    # 슬라이더 키 초기값 세팅 (첫 렌더링 시에만)
                    if f'manual_slider_{v_clean}' not in st.session_state:
                        st.session_state[f'manual_slider_{v_clean}'] = (
                            float(st.session_state[f'm_{v_clean}_min']),
                            float(st.session_state[f'm_{v_clean}_max'])
                        )
                    if f'manual_min_{v_clean}' not in st.session_state:
                        st.session_state[f'manual_min_{v_clean}'] = float(st.session_state[f'm_{v_clean}_min'])
                    if f'manual_max_{v_clean}' not in st.session_state:
                        st.session_state[f'manual_max_{v_clean}'] = float(st.session_state[f'm_{v_clean}_max'])

                    if _is_na_x:
                        st.markdown(
                            f"<div style='background:#1e293b;border:1px solid #334155;border-radius:5px;"
                            f"padding:5px 10px;margin:4px 0 2px 0;display:flex;align-items:center;gap:8px;'>"
                            f"<span style='font-size:0.8rem;font-weight:600;color:#64748b;'>{v}</span>"
                            f"<span style='background:#334155;color:#94a3b8;font-size:0.65rem;padding:1px 6px;border-radius:3px;'>N/A</span>"
                            f"<span style='font-size:0.72rem;color:#475569;'>({meaning})</span></div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<p style='font-size:0.8rem;font-weight:600;color:#38bdf8;margin:6px 0 2px 0;'>"
                            f"{v} <span style='color:#64748b;font-weight:400;'>({meaning})</span></p>",
                            unsafe_allow_html=True
                        )

                    sl_col, min_col, max_col = st.columns([2.0, 1.0, 1.0])

                    with sl_col:
                        st.slider(
                            f"{v} Range", db_min, db_max,
                            step=step_v,
                            label_visibility="collapsed",
                            key=f"manual_slider_{v_clean}",
                            on_change=on_manual_slider_change,
                            args=(v_clean,)
                        )

                    with min_col:
                        st.number_input(
                            "Min",
                            min_value=db_min, max_value=db_max,
                            step=step_v, format="%.3f",
                            key=f"manual_min_{v_clean}",
                            on_change=on_manual_min_change,
                            args=(v_clean,),
                            label_visibility="visible"
                        )

                    with max_col:
                        st.number_input(
                            "Max",
                            min_value=db_min, max_value=db_max,
                            step=step_v, format="%.3f",
                            key=f"manual_max_{v_clean}",
                            on_change=on_manual_max_change,
                            args=(v_clean,),
                            label_visibility="visible"
                        )

                    if _is_na_x:
                        gray_out_slider(f"{v} Range")

                    chosen_bounds[v] = st.session_state[f'manual_slider_{v_clean}']
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown(f"</div><div class='glass-card'><div class='glass-card-title'>{L_G['kpi_title']}</div>", unsafe_allow_html=True)
            
            with st.expander((f"▸ Expand to Adjust Target Range  |  {len(valid_tgts)} targets total" if st.session_state.get('lang','KO')=='EN' else f"▸ 목표 범위 펼쳐서 조정  |  전체 {len(valid_tgts)}개 타겟"), expanded=False):
                st.markdown("<div style='max-height:430px; overflow-y:auto; padding-right:10px;'>", unsafe_allow_html=True)
                for idx, tgt in enumerate(valid_tgts):
                    t_low = tgt.lower()
                    _na_tgts_ui = st.session_state.get('na_spec_targets', [])
                    _df_ui = st.session_state.get('df_caulking', pd.DataFrame())

                    if tgt in _na_tgts_ui:
                        # 데이터 min/max 계산
                        _d_min, _d_max = 0.0, 1.0
                        if not _df_ui.empty and tgt in _df_ui.columns:
                            _td_ui = pd.to_numeric(_df_ui[tgt], errors='coerce').dropna()
                            if not _td_ui.empty:
                                _d_min = float(_td_ui.min())
                                _d_max = float(_td_ui.max())
                                if _d_min == _d_max: _d_max = _d_min + 1.0
                        _step_na = max(round((_d_max - _d_min) / 100, 4), 0.001)

                        # N/A 헤더 박스 - 절반 너비
                        _hdr_col, _ = st.columns(2)
                        with _hdr_col:
                            st.markdown(
                                f"<div style='background:#1e293b;border:1px solid #334155;border-radius:6px;"
                                f"padding:8px 12px;margin-bottom:4px;'>"
                                f"<div style='display:flex;align-items:center;gap:8px;flex-wrap:nowrap;white-space:nowrap;overflow:hidden;'>"
                                f"<span style='font-size:0.85rem;font-weight:600;color:#94a3b8;flex-shrink:0;'>{idx+1}. Target {tgt}</span>"
                                f"<span style='background:#334155;color:#94a3b8;font-size:0.7rem;font-weight:700;"
                                f"padding:2px 8px;border-radius:4px;flex-shrink:0;'>N/A</span>"
                                f"<span style='font-size:0.72rem;color:#64748b;overflow:hidden;text-overflow:ellipsis;'>스펙 미설정 · {_d_min:.3f}~{_d_max:.3f}</span>"
                                f"</div></div>",
                                unsafe_allow_html=True
                            )
                        # 리셋 플래그를 컬럼 진입 전에 처리 (pop은 1번만)
                        _do_reset = st.session_state.pop(f"_na_reset_{t_low}", False)
                        if _do_reset or f"{t_low}_n_min" not in st.session_state:
                            st.session_state[f"{t_low}_n_min"] = _d_min
                        if _do_reset or f"{t_low}_n_max" not in st.session_state:
                            st.session_state[f"{t_low}_n_max"] = _d_max

                        # 슬라이더와 Min/Max 단일 행 수평 배치
                        col_c1, col_c2, col_c3 = st.columns([1.8, 0.6, 0.6])
                        with col_c1:
                            if f"{t_low}_s_val" not in st.session_state:
                                st.session_state[f"{t_low}_s_val"] = (_d_min, _d_max)
                            st.slider(f"{tgt} Slider UI", _d_min, _d_max, step=_step_na,
                                      format="%.3f",
                                      label_visibility="collapsed", key=f"{t_low}_s_val",
                                      on_change=on_slider_change, args=(t_low,))
                        with col_c2:
                            st.number_input("Min", step=_step_na, format="%.3f",
                                            key=f"{t_low}_n_min",
                                            on_change=on_min_change, args=(t_low,))
                        with col_c3:
                            st.number_input("Max", step=_step_na, format="%.3f",
                                            key=f"{t_low}_n_max",
                                            on_change=on_max_change, args=(t_low,))
                        gray_out_slider(f"{tgt} Slider UI")
                        st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)
                        continue

                    # spec_from_file 우선, 없으면 SPEC_GUIDE 폴백
                    _sf2 = st.session_state.get('spec_from_file', {})
                    if tgt in _sf2:
                        spec_min, spec_max = _sf2[tgt]
                    else:
                        spec_txt = SPEC_GUIDE.get(tgt, "0.0 ~ 1.0")
                        try:
                            spec_min, spec_max = map(float, spec_txt.split(" ~ "))
                        except:
                            spec_min, spec_max = 0.0, 1.0
                    # 슬라이더 범위: spec 범위의 1.5배 (통일성)
                    _span = max(spec_max - spec_min, 0.01)
                    max_slider_val = round(spec_max + _span * 0.5, 4)
                    # step: 범위에 따라 자동 (0.001~0.1)
                    step_size = max(round(_span / 200, 4), 0.001)
                    if _span >= 50:    step_size = 1.0
                    elif _span >= 5:   step_size = 0.1
                    elif _span >= 0.5: step_size = 0.01
                    elif _span >= 0.1: step_size = 0.005
                    else:              step_size = 0.001
                    
                    st.markdown(f"<p style='font-size:0.85rem; font-weight:600; color:#38bdf8; margin-bottom:5px;'>{idx+1}. Target {tgt} Range (Spec: {spec_min:.1f} ~ {spec_max:.1f})</p>", unsafe_allow_html=True)
                    
                    col_c1, col_c2, col_c3 = st.columns([1.8, 0.6, 0.6])
                    with col_c1:
                        st.session_state.setdefault(f"{t_low}_s_val", (0.0, float(max_slider_val)))
                        st.slider(f"{tgt} Slider UI", 0.0, float(max_slider_val), step=step_size, label_visibility="collapsed", key=f"{t_low}_s_val", on_change=on_slider_change, args=(t_low,))
                    with col_c2:
                        st.number_input("Min", step=step_size, key=f"{t_low}_n_min", on_change=on_min_change, args=(t_low,))
                    with col_c3:
                        st.number_input("Max", step=step_size, key=f"{t_low}_n_max", on_change=on_max_change, args=(t_low,))
                    st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            undefined_tgts = [t for t in target_vars if t not in valid_tgts]
            if undefined_tgts:
                with st.expander(f"Undefined Quality Targets (미정 항목 {len(undefined_tgts)}종 제어단)"):
                    for tgt in undefined_tgts:
                        t_low = tgt.lower()
                        st.markdown(f"<p style='font-size:0.8rem; margin:2px 0; color:#94a3b8;'>• {tgt} Range</p>", unsafe_allow_html=True)
                        cx1, cx2, cx3 = st.columns([1.8, 0.6, 0.6])
                        st.session_state.setdefault(f"{t_low}_s_val", (-0.5, 100.0))
                        with cx1: st.slider(f"{tgt} S", -0.5, 100.0, step=0.01, label_visibility="collapsed", key=f"{t_low}_s_val", on_change=on_slider_change, args=(t_low,))
                        with cx2:
                            st.number_input("Min", step=0.01, key=f"{t_low}_n_min", on_change=on_min_change, args=(t_low,))
                        with cx3:
                            st.number_input("Max", step=0.01, key=f"{t_low}_n_max", on_change=on_max_change, args=(t_low,))
            st.markdown("</div>", unsafe_allow_html=True)

            # N/A 공정 변수 처리 방식 옵션
            _is1b = st.session_state.get('lang', 'KO') == 'EN'
            _na_x_opt_list = st.session_state.get('na_x_vars', [])
            if _na_x_opt_list:
                _na_hdr_txt = "N/A Process Variable Handling" if _is1b else "N/A 공정 변수 처리 방식"
                st.markdown(
                    f"<div style='background:#0a1628;border:1px solid #1e3a5f;border-radius:6px;"
                    f"padding:8px 14px;margin-bottom:8px;'>"
                    f"<span style='font-size:0.78rem;color:#38bdf8;font-weight:600;'>{_na_hdr_txt}</span>"
                    f"<span style='font-size:0.70rem;color:#64748b;margin-left:8px;'>"
                    f"({', '.join(_na_x_opt_list[:5])}{'...' if len(_na_x_opt_list)>5 else ''})</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                _na_x_mode = st.radio(
                    "N/A Variable Handling" if _is1b else "N/A 변수 처리",
                    options=(["Range Search (current)", "Fix at Mean"] if _is1b else ["범위 탐색 (현재)", "평균값 고정"]),
                    index=0,
                    horizontal=True,
                    key="na_x_opt_mode",
                    label_visibility="collapsed",
                    help=("Range Search: auto-select the best value within data min~max | Fix at Mean: fix to the data average and exclude from search"
                          if _is1b else "범위 탐색: 데이터 min~max 안에서 최적값 자동 선택 | 평균값 고정: 데이터 평균값으로 고정 후 제외")
                )
            else:
                _na_x_mode = "Range Search (current)" if _is1b else "범위 탐색 (현재)"

            if st.button(L_G['run_opt'], type="primary", key="main_run_opt"):
                base_x = [st.session_state[f'sim_{v.lower()}'] for v in X_list]
                q_base = st.session_state['scaler'].transform(pd.DataFrame([base_x], columns=X_list))[0]

                # N/A X변수 처리 방식 적용
                _na_x_vars_opt = st.session_state.get('na_x_vars', [])
                _na_x_means = {}
                if any(k in st.session_state.get('na_x_opt_mode', '') for k in ("평균값 고정", "Fix at Mean")) and _na_x_vars_opt:
                    _df_ref = st.session_state.get('df_imputed_ref')
                    if _df_ref is not None:
                        for _nxv in _na_x_vars_opt:
                            if _nxv in _df_ref.columns:
                                _na_x_means[_nxv] = float(pd.to_numeric(_df_ref[_nxv], errors='coerce').mean())
                    st.sidebar.info(f"N/A 변수 {len(_na_x_means)}개 평균값 고정 적용")

                df_train = st.session_state.get('df_imputed_ref')
                _na_tgts_opt = st.session_state.get('na_spec_targets', [])
                feasibility = {}
                if df_train is not None:
                    q_train = st.session_state['scaler'].transform(df_train[X_list])
                    for tgt in target_vars:
                        if tgt in _na_tgts_opt:          # N/A 타겟 — feasibility 계산 제외
                            continue
                        mk = f'model_{tgt.lower()}'
                        if st.session_state[mk] is not None:
                            preds = st.session_state[mk].predict(q_train)
                            pred_min, pred_max = float(np.min(preds)), float(np.max(preds))
                            t_lo, t_hi = st.session_state[f'{tgt.lower()}_s_val']
                            overlap = max(0.0, min(pred_max, t_hi) - max(pred_min, t_lo))
                            spec_span = max(t_hi - t_lo, 1e-9)
                            pred_span = max(pred_max - pred_min, 1e-9)
                            overlap_ratio = overlap / min(spec_span, pred_span)
                            feasibility[tgt] = {
                                'pred_min': pred_min, 'pred_max': pred_max,
                                'overlap_ratio': overlap_ratio,
                                'weight': max(0.05, min(1.0, overlap_ratio))
                            }
                        else:
                            feasibility[tgt] = {'pred_min': 0, 'pred_max': 0, 'overlap_ratio': 1.0, 'weight': 1.0}
                else:
                    for tgt in target_vars:
                        if tgt in _na_tgts_opt:
                            continue
                        feasibility[tgt] = {'pred_min': 0, 'pred_max': 0, 'overlap_ratio': 1.0, 'weight': 1.0}

                st.session_state['feasibility'] = feasibility

                infeasible_count = sum(1 for v in feasibility.values() if v['overlap_ratio'] < 0.1)
                lambda_reg = 0.5 + (infeasible_count * 0.3)

                def target_loss(x):
                    df_x = pd.DataFrame([x], columns=X_list)
                    q = st.session_state['scaler'].transform(df_x)
                    total_loss = 0.0
                    for tgt in target_vars:
                        if tgt in _na_tgts_opt:          # N/A 타겟 — 손실 계산 제외
                            continue
                        model_key = f'model_{tgt.lower()}'
                        if st.session_state[model_key] is None:
                            continue
                        pred = st.session_state[model_key].predict(q)[0]
                        t_range = st.session_state[f'{tgt.lower()}_s_val']
                        w = feasibility.get(tgt, {}).get('weight', 1.0)
                        if w >= 0.99:
                            boundary_loss = (max(0, t_range[0] - pred) + max(0, pred - t_range[1]))**2
                        else:
                            nearest_spec = t_range[0] if pred < t_range[0] else (t_range[1] if pred > t_range[1] else pred)
                            boundary_loss = w * (pred - nearest_spec)**2
                        total_loss += boundary_loss
                    dist_penalty = np.sum((q[0] - q_base)**2)
                    return total_loss + (lambda_reg * dist_penalty)

                init_x = base_x.copy()
                # 평균값 고정 모드: N/A X변수를 평균값으로 초기화
                if _na_x_means:
                    for _i, _v in enumerate(X_list):
                        if _v in _na_x_means:
                            init_x[_i] = _na_x_means[_v]

                bands = []
                for _v in X_list:
                    if _na_x_means and _v in _na_x_means:
                        _mv = _na_x_means[_v]
                        bands.append((_mv, _mv + 1e-9))   # 고정: 범위를 평균값으로 좁힘
                    else:
                        bands.append(db[_v])

                algorithms = ['L-BFGS-B', 'SLSQP', 'Powell', 'Nelder-Mead']
                best_loss = float('inf')
                best_res = None
                selected_algo = 'SLSQP'
                algo_loss_dict = {}   # 각 알고리즘별 손실값 저장

                _is1d = st.session_state.get('lang', 'KO') == 'EN'
                opt_progress_bar = st.progress(0, text=("Preparing inverse optimization search... (0%)" if _is1d else "역추론 최적화 탐색 준비 중... (0%)"))
                total_algos_n = len(algorithms)
                
                for a_idx, algo in enumerate(algorithms):
                    opt_progress_pct = int((a_idx / total_algos_n) * 100)
                    _prog_txt = (f" Searching algorithm ({a_idx+1}/{total_algos_n}): {algo} ({opt_progress_pct}%)" if _is1d
                                 else f" 알고리즘 탐색 중 ({a_idx+1}/{total_algos_n}): {algo} ({opt_progress_pct}%)")
                    opt_progress_bar.progress(a_idx / total_algos_n, text=_prog_txt)
                    try:
                        if algo in ['L-BFGS-B', 'SLSQP']: res_temp = minimize(target_loss, init_x, method=algo, bounds=bands)
                        else: res_temp = minimize(target_loss, init_x, method=algo)
                        final_x = np.clip(res_temp.x, [b[0] for b in bands], [b[1] for b in bands])
                        current_score_loss = target_loss(final_x)
                        algo_loss_dict[algo] = round(float(current_score_loss), 6)   # 저장
                        
                        if current_score_loss < best_loss:
                            best_loss = current_score_loss
                            best_res = res_temp
                            best_res.x = final_x
                            selected_algo = algo
                    except Exception as e:
                        algo_loss_dict[algo] = None
                        continue
                
                opt_progress_bar.progress(1.0, text=(f"✅ Optimization complete (100%) - Selected algorithm: {selected_algo}" if _is1d
                                                       else f"✅ 최적화 완료 (100%) - 선택된 알고리즘: {selected_algo}"))
                q_opt = st.session_state['scaler'].transform(pd.DataFrame([best_res.x], columns=X_list))
                
                update_opt_dict = {
                    'opt_result_x': best_res.x, 
                    'confidence_score': round(max(0.0, 100.0 - (best_loss * 5)), 1),
                    'best_algorithm_used': selected_algo,
                    'algo_loss_dict': algo_loss_dict,
                    'na_x_mode_used': "평균값 고정" if _na_x_means else "범위 탐색",
                    'na_x_means_used': _na_x_means,
                    'ai_analysis_result': None
                }
                for tgt in target_vars:
                    model_key = f'model_{tgt.lower()}'
                    if st.session_state[model_key] is not None: update_opt_dict[f'opt_pred_{tgt.lower()}'] = float(st.session_state[model_key].predict(q_opt)[0])
                    else: update_opt_dict[f'opt_pred_{tgt.lower()}'] = 0.0
                        
                st.session_state.update(update_opt_dict)
                st.rerun()

        with layout_r:
                    _is1d = st.session_state.get('lang', 'KO') == 'EN'
                    if st.session_state.get('opt_result_x') is not None:
                        st.markdown(f"<div class='glass-card'><div class='glass-card-title' style='color:#3b82f6;'>{L_G['pred_title']}</div>", unsafe_allow_html=True)
                        with st.expander((f"▸ Backward Optimization Algorithm Competition  |  Selected: {st.session_state['best_algorithm_used']}" if _is1d
                                           else f"▸ 역방향 최적화 알고리즘 경쟁 결과  |  채택: {st.session_state['best_algorithm_used']}"), expanded=False):
                            _algo_info = ({
                                'L-BFGS-B':    'Gradient-based, bound-constrained',
                                'SLSQP':       'Gradient-based, bound-constrained',
                                'Powell':      'Directional search, clipped',
                                'Nelder-Mead': 'Simplex-based, clipped',
                            } if _is1d else {
                                'L-BFGS-B':    '기울기 기반, 경계 준수',
                                'SLSQP':       '기울기 기반, 경계 준수',
                                'Powell':      '방향 탐색, clip 처리',
                                'Nelder-Mead': '도형 변형, clip 처리',
                            })
                            _sel = st.session_state['best_algorithm_used']
                            _loss_d = st.session_state.get('algo_loss_dict', {})
                            _rows_algo = ""
                            for _a, _m in _algo_info.items():
                                _is_sel = _a == _sel
                                _bg = "#0a2010" if _is_sel else "#0d1f3c"
                                _nc = "#10b981" if _is_sel else "#94a3b8"
                                _star = "★ " if _is_sel else ""
                                _lv = _loss_d.get(_a)
                                if _lv is None:
                                    _loss_str = f"<span style='color:#475569;'>{'Failed' if _is1d else '실행 실패'}</span>"
                                    _conf_str = "<span style='color:#475569;'>—</span>"
                                    _bar_w = 0
                                else:
                                    _conf_v = round(max(0.0, 100.0 - (_lv * 5)), 1)
                                    _lc = "#10b981" if _lv < 0.01 else "#f59e0b" if _lv < 1.0 else "#f87171"
                                    _loss_str = f"<span style='color:{_lc};font-weight:700;font-family:monospace;'>{_lv:.4f}</span>"
                                    _conf_str = f"<span style='color:{_lc};font-weight:700;'>{_conf_v}%</span>"
                                    _bar_w = min(int(_conf_v), 100)
                                _badge = (f"<span style='background:#0a2010;color:#10b981;font-size:0.68rem;padding:1px 6px;border-radius:3px;'>✅ {'Selected' if _is1d else '채택'}</span>" if _is_sel else "")
                                _bar = f"<div style='background:#1e293b;border-radius:2px;height:6px;margin-top:2px;'><div style='width:{_bar_w}%;background:{'#10b981' if _bar_w>80 else '#f59e0b' if _bar_w>50 else '#f87171'};height:6px;border-radius:2px;'></div></div>"
                                _rows_algo += (
                                    f"<tr style='background:{_bg};'>"
                                    f"<td style='padding:6px 8px;color:{_nc};font-weight:700;font-family:monospace;white-space:nowrap;'>{_star}{_a}</td>"
                                    f"<td style='padding:6px 8px;color:#64748b;font-size:0.72rem;'>{_m}</td>"
                                    f"<td style='padding:6px 8px;text-align:center;'>{_loss_str}</td>"
                                    f"<td style='padding:6px 8px;min-width:80px;'>{_conf_str}{_bar}</td>"
                                    f"<td style='padding:6px 8px;text-align:center;'>{_badge}</td>"
                                    f"</tr>"
                                )
                            _th_algo1, _th_algo2, _th_algo3, _th_algo4, _th_algo5 = (
                                ("Algorithm", "Method", "Loss ↓lower is better", "Confidence", "Selected") if _is1d
                                else ("알고리즘", "방식", "손실값 ↓낮을수록 좋음", "신뢰도", "채택")
                            )
                            _algo_footer = ("The algorithm with the lowest loss is auto-selected. &nbsp;Confidence = max(0, 100 − loss×5)" if _is1d
                                             else "손실값이 가장 낮은 알고리즘이 자동 채택됩니다. &nbsp;신뢰도 = max(0, 100 − 손실값×5)")
                            st.markdown(
                                f"<div style='background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;padding:12px 14px;'>"
                                f"<table style='width:100%;border-collapse:collapse;'>"
                                f"<thead><tr style='border-bottom:1px solid #1e3a5f;'>"
                                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:left;'>{_th_algo1}</th>"
                                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;'>{_th_algo2}</th>"
                                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:center;'>{_th_algo3}</th>"
                                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;'>{_th_algo4}</th>"
                                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:center;'>{_th_algo5}</th>"
                                f"</tr></thead>"
                                f"<tbody>{_rows_algo}</tbody>"
                                f"</table>"
                                f"<div style='margin-top:8px;font-size:0.72rem;color:#475569;'>{_algo_footer}</div>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                
                        st.markdown("<div style='margin: -10px 0 10px 0;'></div>", unsafe_allow_html=True)
                        # Auto-selected Algorithms → expander
                        _mm_disp = st.session_state.get('model_metadata', {})
                        def _ac(n):
                            if 'XGBoost' in n:          return '#f59e0b'
                            if 'RandomForest' in n:     return '#10b981'
                            if 'GradientBoosting' in n: return '#a78bfa'
                            if 'ExtraTrees' in n:       return '#38bdf8'
                            return '#94a3b8'
                        def _pm(s):
                            a = s.split('(')[0].strip() if '(' in s else s
                            r = s.split('R²=')[-1].split(',')[0].rstrip(')') if 'R²=' in s else '-'
                            return a, r
                        with st.expander(("▸ Prediction Model Selection Results (Auto-selected Algorithms)" if _is1d
                                           else "▸ 예측 모델 선택 결과 (Auto-selected Algorithms)"), expanded=False):
                            _msr_rows = ""
                            for _t in valid_tgts:
                                _mk = f'algo_{_t.lower()}'
                                if _mk not in _mm_disp: continue
                                _al, _r2v = _pm(_mm_disp[_mk])
                                _clr = _ac(_al)
                                _bdg = '★ ' if _al in ('XGBoost','RandomForest','GradientBoosting','ExtraTrees') else ''
                                try:    _r2f = float(_r2v)
                                except: _r2f = 0.0
                                _bar_r2 = min(int(_r2f*100), 100)
                                _r2_color = '#10b981' if _r2f>=0.9 else '#f59e0b' if _r2f>=0.7 else '#f87171'
                                _msr_rows += (
                                    f"<tr>"
                                    f"<td style='padding:5px 8px;color:#e2e8f0;font-weight:700;'>{_t}</td>"
                                    f"<td style='padding:5px 8px;color:{_clr};font-weight:700;font-size:0.82rem;'>{_bdg}{_al}</td>"
                                    f"<td style='padding:5px 8px;min-width:90px;'>"
                                    f"<span style='color:{_r2_color};font-weight:700;font-family:monospace;'>{_r2v}</span>"
                                    f"<div style='background:#1e293b;border-radius:2px;height:4px;margin-top:2px;'>"
                                    f"<div style='width:{_bar_r2}%;background:{_r2_color};height:4px;border-radius:2px;'></div></div>"
                                    f"</td>"
                                    f"</tr>"
                                )
                            _th_ms1, _th_ms2, _th_ms3 = (("Target", "Selected Algorithm", "R² (Training Accuracy)") if _is1d
                                                          else ("타겟", "선택 알고리즘", "R² (학습 정확도)"))
                            _ms_footer = ("★ Tree ensemble &nbsp;|&nbsp;"
                                          "<span style='color:#f59e0b;'>■</span> XGBoost &nbsp;"
                                          "<span style='color:#10b981;'>■</span> RandomForest &nbsp;"
                                          "<span style='color:#a78bfa;'>■</span> GradBoost &nbsp;"
                                          "<span style='color:#38bdf8;'>■</span> ExtraTrees &nbsp;"
                                          "<span style='color:#94a3b8;'>■</span> Linear<br>"
                                          "The closer R² is to 1.0, the higher the training accuracy. Auto-selected by CV R² among 6 algorithms."
                                          if _is1d else
                                          "★ 트리 앙상블 &nbsp;|&nbsp;"
                                          "<span style='color:#f59e0b;'>■</span> XGBoost &nbsp;"
                                          "<span style='color:#10b981;'>■</span> RandomForest &nbsp;"
                                          "<span style='color:#a78bfa;'>■</span> GradBoost &nbsp;"
                                          "<span style='color:#38bdf8;'>■</span> ExtraTrees &nbsp;"
                                          "<span style='color:#94a3b8;'>■</span> Linear<br>"
                                          "R² 1.0에 가까울수록 학습 정확도 높음. 6종 알고리즘 중 CV R² 기준 자동 선택.")
                            st.markdown(
                                f"<div style='background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;padding:12px 14px;'>"
                                f"<table style='width:100%;border-collapse:collapse;'>"
                                f"<thead><tr style='border-bottom:1px solid #1e3a5f;'>"
                                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:left;'>{_th_ms1}</th>"
                                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;'>{_th_ms2}</th>"
                                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;'>{_th_ms3}</th>"
                                f"</tr></thead>"
                                f"<tbody>{_msr_rows}</tbody>"
                                f"</table>"
                                f"<div style='margin-top:8px;font-size:0.68rem;color:#475569;'>{_ms_footer}</div>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                
                        # 타겟별 손실 현황 상세 표시
                        _feas_info = st.session_state.get('feasibility', {})
                        _na_tgts_d = st.session_state.get('na_spec_targets', [])
                        _loss_rows = ""
                        _total_loss = 0.0
                        for _tgt in valid_tgts:
                            _pv = st.session_state.get(f'opt_pred_{_tgt.lower()}')
                            if _pv is None:
                                continue
                            if _tgt in _na_tgts_d:
                                _excl_txt = "Excluded" if _is1d else "제외"
                                _loss_rows += (
                                    f"<tr>"
                                    f"<td style='padding:4px 8px;color:#64748b;font-weight:600;'>{_tgt}</td>"
                                    f"<td style='padding:4px 8px;color:#64748b;text-align:center;'>{_pv:.3f}</td>"
                                    f"<td style='padding:4px 8px;color:#475569;text-align:center;font-size:0.72rem;'>N/A</td>"
                                    f"<td style='padding:4px 8px;color:#475569;text-align:center;'>—</td>"
                                    f"<td style='padding:4px 8px;text-align:center;'><span style='background:#1e293b;color:#475569;font-size:0.7rem;padding:1px 6px;border-radius:3px;'>{_excl_txt}</span></td>"
                                    f"</tr>"
                                )
                                continue
                            _t_range = st.session_state.get(f'{_tgt.lower()}_s_val', (None, None))
                            if _t_range[0] is None:
                                continue
                            _lo, _hi = _t_range
                            _over = max(0, _lo - _pv) + max(0, _pv - _hi)
                            _loss_v = _over ** 2
                            _total_loss += _loss_v
                            _spec_str = f"{_lo}~{_hi}"
                            _in_spec = _lo <= _pv <= _hi
                            _loss_color = "#10b981" if _loss_v == 0 else "#f87171"
                            if _is1d:
                                _status_badge = (
                                    f"<span style='background:#0a2010;color:#10b981;font-size:0.7rem;padding:1px 6px;border-radius:3px;'>✅ Achieved</span>"
                                    if _in_spec else
                                    f"<span style='background:#2d0f0f;color:#f87171;font-size:0.7rem;padding:1px 6px;border-radius:3px;'>⚠️ Deviated</span>"
                                )
                            else:
                                _status_badge = (
                                    f"<span style='background:#0a2010;color:#10b981;font-size:0.7rem;padding:1px 6px;border-radius:3px;'>✅ 달성</span>"
                                    if _in_spec else
                                    f"<span style='background:#2d0f0f;color:#f87171;font-size:0.7rem;padding:1px 6px;border-radius:3px;'>⚠️ 이탈</span>"
                                )
                            _loss_rows += (
                                f"<tr>"
                                f"<td style='padding:4px 8px;color:#e2e8f0;font-weight:700;'>{_tgt}</td>"
                                f"<td style='padding:4px 8px;color:#ffffff;text-align:center;font-weight:600;'>{_pv:.3f}</td>"
                                f"<td style='padding:4px 8px;color:#94a3b8;text-align:center;font-size:0.8rem;'>{_spec_str}</td>"
                                f"<td style='padding:4px 8px;color:{_loss_color};text-align:center;font-weight:600;'>{_loss_v:.4f}</td>"
                                f"<td style='padding:4px 8px;text-align:center;'>{_status_badge}</td>"
                                f"</tr>"
                            )
                        if _loss_rows:
                            _loss_exp_title = (f"▸ Loss Status by Target  |  Total Loss: {_total_loss:.4f} {'✅' if _total_loss < 0.01 else '⚠️'}" if _is1d
                                                else f"▸ 타겟별 손실 현황  |  전체 손실: {_total_loss:.4f} {'✅' if _total_loss < 0.01 else '⚠️'}")
                            with st.expander(_loss_exp_title, expanded=False):
                                _lh1, _lh2, _lh3, _lh4, _lh5 = (("Target", "Predicted", "Spec", "Loss", "Judgement") if _is1d
                                                                  else ("타겟", "예측값", "스펙", "손실", "판정"))
                                _total_loss_lbl = "Total Loss" if _is1d else "전체 손실 합계"
                                _perfect_lbl = "← Perfect ✅" if _is1d else "← 완벽 ✅"
                                st.markdown(
                                    f"<div style='background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;padding:12px 14px;'>"
                                    f"<table style='width:100%;border-collapse:collapse;'>"
                                    f"<thead><tr style='border-bottom:1px solid #1e3a5f;'>"
                                    f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:left;'>{_lh1}</th>"
                                    f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:center;'>{_lh2}</th>"
                                    f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:center;'>{_lh3}</th>"
                                    f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:center;'>{_lh4}</th>"
                                    f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:center;'>{_lh5}</th>"
                                    f"</tr></thead>"
                                    f"<tbody>{_loss_rows}</tbody>"
                                    f"</table>"
                                    f"<div style='border-top:1px solid #1e3a5f;margin-top:6px;padding-top:6px;"
                                    f"display:flex;justify-content:space-between;align-items:center;'>"
                                    f"<span style='font-size:0.75rem;color:#94a3b8;'>{_total_loss_lbl}</span>"
                                    f"<span style='font-size:0.9rem;font-weight:700;color:{'#10b981' if _total_loss < 0.01 else '#f87171'};'>"
                                    f"{_total_loss:.4f} {_perfect_lbl if _total_loss < 0.01 else ''}</span>"
                                    f"</div></div>",
                                    unsafe_allow_html=True
                                )
                        st.markdown("</div>", unsafe_allow_html=True)

                        cols_p_card = st.columns(3)
                        _na_tgts = st.session_state.get('na_spec_targets', [])
                        for idx, tgt in enumerate(valid_tgts):
                            p_val = st.session_state[f'opt_pred_{tgt.lower()}']
                            val_display = f"{p_val:.3f}" if isinstance(p_val, float) else "0.000"
                            if tgt in _na_tgts:
                                cols_p_card[idx % 3].markdown(
                                    f"<div style='padding:8px;background:#1a1a2e;border:1px dashed #475569;"
                                    f"border-radius:4px;margin-bottom:6px;opacity:0.8;'>"
                                    f"<span style='color:#64748b;font-size:0.72rem;'>Predicted {tgt}</span>"
                                    f"<span style='float:right;background:#334155;color:#94a3b8;font-size:0.65rem;"
                                    f"padding:1px 5px;border-radius:3px;'>N/A</span><br>"
                                    f"<strong style='font-size:1.05rem;color:#94a3b8;'>{val_display}</strong></div>",
                                    unsafe_allow_html=True
                                )
                            else:
                                _sp_lo, _sp_hi = spec_limits.get(tgt, (None, None))
                                if _sp_lo is not None and isinstance(p_val, float) and (p_val < _sp_lo or p_val > _sp_hi):
                                    cols_p_card[idx % 3].markdown(
                                        f"<div style='padding:8px;background:#2d0f0f;border:1px solid #ef4444;"
                                        f"border-radius:4px;margin-bottom:6px;'>"
                                        f"<span style='color:#f87171;font-size:0.72rem;'>⚠️ {tgt} [이탈]</span><br>"
                                        f"<strong style='font-size:1.05rem;color:#ef4444;'>{val_display}</strong>"
                                        f"<span style='color:#f87171;font-size:0.68rem;'> ({_sp_lo}~{_sp_hi})</span></div>",
                                        unsafe_allow_html=True
                                    )
                                else:
                                    cols_p_card[idx % 3].markdown(
                                        f"<div style='padding:8px;background:#1e293b;border-radius:4px;margin-bottom:6px;'>"
                                        f"<span style='color:#94a3b8;font-size:0.72rem;'>Predicted {tgt}</span><br>"
                                        f"<strong style='font-size:1.05rem;color:#ffffff;'>{val_display}</strong></div>",
                                        unsafe_allow_html=True
                                    )
                
                        st.metric(L_G['opt_conf'], f"{st.session_state['confidence_score']}%")

                        # 예측 데이터 다운로드 (예측카드 아래)
                        pred_dict = {tgt: [st.session_state[f'opt_pred_{tgt.lower()}']] for tgt in valid_tgts}
                        df_pred_export = pd.DataFrame(pred_dict)
                        col_pred_sel, col_pred_trigger = st.columns([1, 1])
                        with col_pred_sel:
                            file_format_pred = st.selectbox(L_G['dl_format'], ["Excel (.xlsx)", "Database (.db)"],
                                                            key="fmt_pred", label_visibility="collapsed")
                        with col_pred_trigger:
                            if "Excel" in file_format_pred:
                                buffer_p = io.BytesIO()
                                with pd.ExcelWriter(buffer_p) as writer: df_pred_export.to_excel(writer, index=False, sheet_name='Predicted_Performance')
                                st.download_button(label=L_G['dl_btn_pred'], data=buffer_p.getvalue(), file_name="predicted_performance.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_btn_pred_trigger")
                            else:
                                conn_p = sqlite3.connect(":memory:")
                                df_pred_export.to_sql("predicted_performance", conn_p, index=False, if_exists="replace")
                                backup_conn_p = sqlite3.connect("temp_pred.db")
                                conn_p.backup(backup_conn_p); backup_conn_p.close(); conn_p.close()
                                with open("temp_pred.db", "rb") as f: db_bytes_p = f.read()
                                st.download_button(label=L_G['dl_btn_pred'], data=db_bytes_p, file_name="predicted_performance.db", mime="application/x-sqlite3", key="dl_btn_pred_db_trigger")
                        st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)

                
                        st.markdown(f"<div class='glass-card'><div class='glass-card-title' style='color:#10b981;'>{L_G['rec_title']}</div>", unsafe_allow_html=True)
                        ox = st.session_state['opt_result_x']
                        df_export = pd.DataFrame([ox], columns=X_list)
                        st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)
                        _na_x_res = st.session_state.get('na_x_vars', [])
                        _na_mode_used = st.session_state.get('na_x_mode_used', '범위 탐색')
                        _na_means_used = st.session_state.get('na_x_means_used', {})
                        _is1c = st.session_state.get('lang', 'KO') == 'EN'
                        if _na_x_res:
                            _mc = "#f59e0b" if "평균값" in _na_mode_used else "#38bdf8"
                            _na_mode_disp = ("Fix at Mean" if "평균값" in _na_mode_used else "Range Search") if _is1c else _na_mode_used
                            _na_hdr2 = "N/A Variable Handling:" if _is1c else "N/A 변수 처리 방식:"
                            st.markdown(
                                f"<div style='background:#0a1628;border:1px solid {_mc};"
                                f"border-radius:5px;padding:5px 12px;margin-bottom:10px;"
                                f"font-size:0.75rem;color:{_mc};'>"
                                f"{_na_hdr2} <b>{_na_mode_disp}</b>"
                                f"<span style='color:#64748b;margin-left:8px;'>"
                                f"({', '.join(_na_x_res[:6])}{'...' if len(_na_x_res)>6 else ''})</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        cols = st.columns(3)
                        for idx, v_name in enumerate(X_list):
                            _xv = ox[idx] if ox[idx] is not None else 0.0
                            val_display = f"{_xv:.3f}"
                            _is_fixed = v_name in _na_x_res and "평균값" in _na_mode_used
                            if v_name in _na_x_res:
                                if _is1c:
                                    _sub_lbl = "Fixed Mean" if _is_fixed else "Range Search"
                                    _badge_txt = "Fixed" if _is_fixed else "Spec N/A"
                                else:
                                    _sub_lbl = "평균 고정" if _is_fixed else "범위 탐색"
                                    _badge_txt = "고정값" if _is_fixed else "스펙N/A"
                                _sub_color = "#f59e0b" if _is_fixed else "#64748b"
                                cols[idx % 3].markdown(
                                    f"<div style='padding:8px;background:#1e293b;"
                                    f"border:1px dashed #334155;border-radius:4px;margin-bottom:6px;opacity:0.8;'>"
                                    f"<span style='color:#64748b;font-size:0.72rem;'>{v_name}</span>"
                                    f"<span style='float:right;background:#334155;color:{_sub_color};font-size:0.6rem;"
                                    f"padding:1px 4px;border-radius:3px;line-height:1.4;'>{_badge_txt}</span><br>"
                                    f"<strong style='font-size:1.05rem;color:#94a3b8;'>{val_display}</strong></div>",
                                    unsafe_allow_html=True
                                )
                            else:
                                cols[idx % 3].markdown(
                                    f"<div style='padding:8px;background:#1e293b;"
                                    f"border-radius:4px;margin-bottom:6px;'>"
                                    f"<span style='color:#94a3b8;font-size:0.72rem;'>{v_name}</span><br>"
                                    f"<strong style='font-size:1.05rem;color:#ffffff;'>{val_display}</strong></div>",
                                    unsafe_allow_html=True
                                )
                        # 추천 스펙 다운로드 (스펙 카드 아래)
                        col_dl_sel, col_dl_trigger = st.columns([1, 1])
                        with col_dl_sel:
                            file_format = st.selectbox(L_G['dl_format'], ["Excel (.xlsx)", "Database (.db)"],
                                                       key="fmt_spec", label_visibility="collapsed")
                        with col_dl_trigger:
                            if "Excel" in file_format:
                                buffer = io.BytesIO()
                                with pd.ExcelWriter(buffer) as writer: df_export.to_excel(writer, index=False, sheet_name='Optimized_Specs')
                                st.download_button(label=L_G['dl_btn_spec'], data=buffer.getvalue(), file_name="recommended_process_spec.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_btn_spec_trigger")
                            else:
                                conn = sqlite3.connect(":memory:")
                                df_export.to_sql("recommended_spec", conn, index=False, if_exists="replace")
                                backup_conn = sqlite3.connect("temp_spec.db")
                                conn.backup(backup_conn); backup_conn.close(); conn.close()
                                with open("temp_spec.db", "rb") as f: db_bytes = f.read()
                                st.download_button(label=L_G['dl_btn_spec'], data=db_bytes, file_name="recommended_process_spec.db", mime="application/x-sqlite3", key="dl_btn_spec_db_trigger")
                        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.get('opt_result_x') is not None:
            _is1e = st.session_state.get('lang', 'KO') == 'EN'
            st.markdown(f"### □ {'Result Analysis & Diagnosis' if _is1e else '결과 분석 및 진단'}")
            has_warning = False
            feasibility_info = st.session_state.get('feasibility', {})

            result_cards = []
            _na_tgts_res = st.session_state.get('na_spec_targets', [])
            for tgt in valid_tgts:
                pred_val = st.session_state.get(f'opt_pred_{tgt.lower()}')
                if pred_val is None: continue

                if tgt in _na_tgts_res:
                    if _is1e:
                        result_cards.append(('na',
                            f"⬜ **{tgt}** — Spec N/A\n\n"
                            f"Predicted: **{pred_val:.3f}**\n\n"
                            f"Excluded from optimization (no spec set)"
                        ))
                    else:
                        result_cards.append(('na',
                            f"⬜ **{tgt}** — 스펙 N/A\n\n"
                            f"예측값: **{pred_val:.3f}**\n\n"
                            f"스펙 미설정으로 최적화 계산에서 제외"
                        ))
                    continue

                t_range = st.session_state[f'{tgt.lower()}_s_val']
                feas = feasibility_info.get(tgt, {})
                overlap = feas.get('overlap_ratio', 1.0)
                pred_min = feas.get('pred_min', None)
                pred_max = feas.get('pred_max', None)

                if overlap < 0.1 and pred_min is not None:
                    if _is1e:
                        result_cards.append(('error',
                            f"🚫 **{tgt}** Infeasible\n\n"
                            f"Predicted range: **{pred_min:.2f}~{pred_max:.2f}**\n\n"
                            f"Target spec: {t_range[0]}~{t_range[1]}\n\n"
                            f"Prediction: **{pred_val:.3f}**"
                        ))
                    else:
                        result_cards.append(('error',
                            f"🚫 **{tgt}** 달성 불가\n\n"
                            f"예측 범위: **{pred_min:.2f}~{pred_max:.2f}**\n\n"
                            f"설정 스펙: {t_range[0]}~{t_range[1]}\n\n"
                            f"예측 결과: **{pred_val:.3f}**"
                        ))
                    has_warning = True
                elif t_range[0] is not None and t_range[1] is not None and (pred_val > t_range[1] or pred_val < t_range[0]):
                    if _is1e:
                        result_cards.append(('warning', f"⚠️ **{tgt}** Out of Spec\n\nPredicted **{pred_val:.3f}**\n\nTarget spec {t_range[0]}~{t_range[1]}"))
                    else:
                        result_cards.append(('warning', f"⚠️ **{tgt}** 스펙 이탈\n\n예측값 **{pred_val:.3f}**\n\n설정 스펙 {t_range[0]}~{t_range[1]}"))
                    has_warning = True
                else:
                    if _is1e:
                        result_cards.append(('success', f"✅ **{tgt}** Achieved\n\nPredicted **{pred_val:.3f}**\n\nTarget spec {t_range[0]}~{t_range[1]}"))
                    else:
                        result_cards.append(('success', f"✅ **{tgt}** 정상 도달\n\n예측값 **{pred_val:.3f}**\n\n설정 스펙 {t_range[0]}~{t_range[1]}"))

            RESULT_COLS_PER_ROW = 5
            for i in range(0, len(result_cards), RESULT_COLS_PER_ROW):
                row_cards = result_cards[i:i + RESULT_COLS_PER_ROW]
                grid_cols = st.columns(RESULT_COLS_PER_ROW)
                for gc, (rtype, msg) in zip(grid_cols, row_cards):
                    with gc:
                        if rtype == 'na':
                            # N/A 카드 — 다른 카드와 동일 크기
                            _p = msg.replace('**','').split('\n\n')
                            _t = _p[0] if _p else ''
                            gc.markdown(
                                f"<div style='background:#1e293b;border:1px dashed #475569;"
                                f"border-radius:6px;padding:8px 12px;font-size:0.82rem;margin-bottom:6px;'>"
                                f"<span style='color:#94a3b8;font-weight:700;font-size:0.80rem;'>{_t}</span><br>"
                                f"<span style='color:#64748b;font-size:0.78rem;'>&nbsp;</span>"
                                f"<span style='color:#475569;font-size:0.72rem;display:block;'>&nbsp;</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            # 정상/이탈/불가 카드 — 컴팩트 HTML (세로 2/3)
                            _parts = msg.split('\n\n')
                            _title = _parts[0].replace('**','') if _parts else ''
                            _pred  = _parts[1].replace('**','') if len(_parts) > 1 else ''
                            _spec  = _parts[2].replace('**','') if len(_parts) > 2 else ''
                            if rtype == 'success':
                                _bg, _bd, _tc, _vc = '#0a2010', '#10b981', '#6ee7b7', '#ffffff'
                            elif rtype == 'warning':
                                _bg, _bd, _tc, _vc = '#1a1200', '#f59e0b', '#fcd34d', '#ffffff'
                            else:
                                _bg, _bd, _tc, _vc = '#2d0f0f', '#ef4444', '#f87171', '#ffffff'
                            gc.markdown(
                                f"<div style='background:{_bg};border:1px solid {_bd};"
                                f"border-radius:6px;padding:8px 12px;font-size:0.82rem;margin-bottom:6px;'>"
                                f"<span style='color:{_tc};font-weight:700;font-size:0.80rem;'>{_title}</span><br>"
                                f"<span style='color:{_vc};font-size:0.78rem;'>{_pred}</span>"
                                f"<span style='color:#64748b;font-size:0.72rem;display:block;'>{_spec}</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

            if has_warning:
                if _is1e:
                    st.info(
                        "**Reliability-Guaranteed Optimization Applied**\n\n"
                        "Infeasible targets are converted to soft weights in the loss function, so the "
                        "optimizer is automatically adjusted to avoid drifting to extreme values "
                        "(Feasibility-Aware Adaptive Optimization). The current result is the most realistic "
                        "best process condition derivable within the data's limits."
                    )
                else:
                    st.info(
                        "**신뢰성 보장 최적화 적용됨**\n\n"
                        "달성 불가 타겟은 손실 함수에서 소프트 가중치(Soft Weight)로 전환되어, "
                        "옵티마이저가 극단적인 값으로 이탈하지 않도록 자동 조정되었습니다(Feasibility-Aware Adaptive Optimization). "
                        "현재 결과는 데이터 한계 내에서 가장 현실적으로 도출된 최선의 공정 조건입니다."
                    )
            else:
                if _is1e:
                    st.info("□  **All Valid Targets Achieved**\n\nBased on the current input data, all targets are predicted to fully reach the specified spec range.")
                else:
                    st.info("□  **전체 유효 타겟 정상 도달**\n\n현재 입력된 데이터를 기준으로 모든 타겟이 설정하신 스펙 범위 내에 완벽히 도달할 수 있는 것으로 분석되었습니다.")

        if st.session_state.get('opt_result_x') is not None:
            st.markdown("---")
            st.markdown(f"### □ {'Process Improvement Guide (Result Diagnosis Report)' if _is1e else '공정 개선 가이드 (결과 진단 리포트)'}")
            _fi_col, _ = st.columns([1.3, 1.2], gap="large")
            with _fi_col:
                _fi_btn1 = st.button(("□ Generate FI Diagnosis + LLM Integrated Process Guide" if _is1e else "□ FI 진단 + LLM 통합 공정 가이드 생성"), key="btn_combined_tab1", type="primary")
            if _fi_btn1:
                with st.spinner("Generating Feature Importance analysis + LLM guideline integration... (10-20 sec)" if _is1e else "Feature Importance 분석 + LLM 가이드라인 통합 생성 중... (10~20초 소요)"):
                    pred_kpis_now = {tgt: st.session_state[f'opt_pred_{tgt.lower()}'] for tgt in valid_tgts if st.session_state.get(f'opt_pred_{tgt.lower()}') is not None}
                    current_p_specs = {X_list[i]: st.session_state['opt_result_x'][i] for i in range(len(X_list))}
                    combined_text = generate_combined_report(
                        process_specs=current_p_specs,
                        predicted_kpis=pred_kpis_now,
                        feasibility_info=st.session_state.get('feasibility', {}),
                        confidence_score=st.session_state.get('confidence_score', 0),
                        mode="Optimization"
                    )
                    st.session_state['combined_report_text'] = combined_text

            if st.session_state.get('combined_report_text'):
                with st.expander(("□ Integrated Process Analysis Report (FI Diagnosis + LLM)" if _is1e else "□ 통합 공정 분석 보고서 (FI 진단 + LLM)"), expanded=True):
                    st.markdown(
                        f"<div class='scrollable-box' style='height:600px;'>"
                        f"{st.session_state['combined_report_text'].replace(chr(10), '<br>')}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    st.download_button(
                        label=("□ Download Integrated Report" if _is1e else "□ 통합 보고서 다운로드"),
                        data=st.session_state['combined_report_text'],
                        file_name=f"JOINT_AI_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain",
                        key="dl_combined_tab1"
                    )

    with tab2:
        sim_l, sim_r = st.columns([1.3, 1.2], gap="large")
        with sim_l:
            _is2a = st.session_state.get('lang', 'KO') == 'EN'
            st.markdown(f"<div class='glass-card'><div class='glass-card-title'>{L_G['bound_title']}</div>", unsafe_allow_html=True)
            st.markdown("<style>.stRadio label span { color: #e2e8f0 !important; font-weight: 600 !important; font-size: 0.92rem !important; } .stRadio [data-testid='stMarkdownContainer'] p { color: #e2e8f0 !important; } .stRadio div[role='radiogroup'] label { color: #e2e8f0 !important; }</style>", unsafe_allow_html=True)

            _sf_dx = st.session_state.get('spec_from_file', {})
            _x_spec_dx = st.session_state.get('x_spec_parsed_dict', {})
            chosen_bounds_dx = {}
            for v in X_list:
                if v in _x_spec_dx:
                    chosen_bounds_dx[v] = _x_spec_dx[v]
                else:
                    chosen_bounds_dx[v] = db[v]
            # 정보 박스 내용: 타겟(품질) 스펙 값 표시
            _spec_lbl_dx = "Spec" if _is2a else "스펙"
            bound_text_dx = ""
            for tgt in valid_tgts:
                _lo_t, _hi_t = spec_limits.get(tgt, (None, None))
                if _lo_t is None:
                    continue
                bound_text_dx += f"• {tgt}: {_lo_t:.3f}~{_hi_t:.3f} <span style='color:#64748b;font-size:0.75rem;'>({_spec_lbl_dx})</span><br>"
            _bound_hdr_dx = "[Target Value Range — Based on Quality Spec]" if _is2a else "[타겟 값 Range — 품질 스펙 기준]"
            _bound_sub_dx = " Falls back to default guide value if no CSV row-2 spec" if _is2a else " CSV 2행 스펙 없으면 기본 가이드값"
            st.markdown(
                f"<div style='background:#0f172a;padding:12px 15px;border-radius:6px;border:1px solid #1e293b;"
                f"font-size:0.85rem;line-height:1.6;max-height:200px;overflow-y:auto;'>"
                f"<span style='color:#38bdf8;font-weight:600;'>{_bound_hdr_dx}</span>"
                f"<span style='color:#64748b;font-size:0.75rem;'>{_bound_sub_dx}</span><br>"
                f"{bound_text_dx}</div>",
                unsafe_allow_html=True
            )

            _card2_title_top_dx = "Design/Process Variable Target Value Range Setting" if _is2a else "설계/공정 변수 타겟 값 범위 설정"
            st.markdown(f"</div><div class='glass-card'><div class='glass-card-title'>{_card2_title_top_dx}</div>", unsafe_allow_html=True)

            _na_x_dx = st.session_state.get('na_x_vars', [])
            with st.expander((f"▸ Expand to Adjust Target Range  |  {len(X_list)} variables total" if st.session_state.get('lang','KO')=='EN' else f"▸ 목표 범위 펼쳐서 조정  |  전체 {len(X_list)}개 변수"), expanded=False):
                st.markdown("<div style='max-height:430px; overflow-y:auto; padding-right:10px;'>", unsafe_allow_html=True)
                for idx, v in enumerate(X_list):
                    v_low = v.lower()
                    spec_min, spec_max = chosen_bounds_dx[v]
                    _span_dx = max(spec_max - spec_min, 0.01)
                    slider_min_dx = round(spec_min - _span_dx * 0.25, 4)
                    slider_max_dx = round(spec_max + _span_dx * 0.25, 4)
                    step_size_dx = max(round(_span_dx / 200, 5), 0.001)
                    _is_na_x_dx = v in _na_x_dx

                    # Min/Max 입력 박스 초기값을 Range 값으로 미리 세팅
                    st.session_state.setdefault(f"sim_tgt_{v_low}_s_val", (float(spec_min), float(spec_max)))
                    st.session_state.setdefault(f"sim_tgt_{v_low}_n_min", float(spec_min))
                    st.session_state.setdefault(f"sim_tgt_{v_low}_n_max", float(spec_max))

                    if _is_na_x_dx:
                        st.markdown(
                            f"<div style='display:flex;align-items:center;gap:6px;margin:4px 0 2px 0;'>"
                            f"<span style='font-size:0.78rem;font-weight:600;color:#94a3b8;'>{idx+1}. {v}</span>"
                            f"<span style='background:#334155;color:#94a3b8;font-size:0.65rem;font-weight:700;"
                            f"padding:1px 6px;border-radius:3px;'>N/A</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(f"<p style='font-size:0.78rem; font-weight:600; color:#38bdf8; margin:6px 0 2px 0;'>{idx+1}. {v} <span style='color:#64748b;font-weight:400;font-size:0.7rem;'>(Spec: {spec_min:.2f} ~ {spec_max:.2f})</span></p>", unsafe_allow_html=True)

                    col_c1, col_c2, col_c3 = st.columns([1.8, 0.6, 0.6])
                    with col_c1:
                        st.slider(f"{v} Slider UI", float(slider_min_dx), float(slider_max_dx), step=step_size_dx,
                                  format="%.3f", label_visibility="collapsed", key=f"sim_tgt_{v_low}_s_val",
                                  on_change=on_sim_slider_change, args=(v_low,))
                    with col_c2:
                        st.number_input("Min", step=step_size_dx, format="%.3f", key=f"sim_tgt_{v_low}_n_min",
                                         on_change=on_sim_min_change, args=(v_low,))
                    with col_c3:
                        st.number_input("Max", step=step_size_dx, format="%.3f", key=f"sim_tgt_{v_low}_n_max",
                                         on_change=on_sim_max_change, args=(v_low,))

                    if _is_na_x_dx:
                        gray_out_slider(f"{v} Slider UI")
                        st.markdown("<div style='margin-bottom:6px;'></div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # N/A 공정 변수 처리 방식 옵션 (Tab1과 동일)
            _na_x_opt_list_dx = st.session_state.get('na_x_vars', [])
            if _na_x_opt_list_dx:
                _na_hdr_dx = "N/A Process Variable Handling" if _is2a else "N/A 공정 변수 처리 방식"
                st.markdown(
                    f"<div style='background:#0a1628;border:1px solid #1e3a5f;border-radius:6px;"
                    f"padding:8px 14px;margin-bottom:8px;'>"
                    f"<span style='font-size:0.78rem;color:#38bdf8;font-weight:600;'>{_na_hdr_dx}</span>"
                    f"<span style='font-size:0.70rem;color:#64748b;margin-left:8px;'>"
                    f"({', '.join(_na_x_opt_list_dx[:5])}{'...' if len(_na_x_opt_list_dx)>5 else ''})</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                _na_x_mode_dx = st.radio(
                    "N/A Variable Handling" if _is2a else "N/A 변수 처리",
                    options=(["Range Search (current)", "Fix at Mean"] if _is2a else ["범위 탐색 (현재)", "평균값 고정"]),
                    index=0,
                    horizontal=True,
                    key="sim_dx_na_opt_mode",
                    label_visibility="collapsed",
                    help=("Range Search: auto-select the best value within the set target range | Fix at Mean: fix to the data average and exclude from search"
                          if _is2a else "범위 탐색: 설정한 목표 범위 안에서 최적값 자동 선택 | 평균값 고정: 데이터 평균값으로 고정 후 제외")
                )
            else:
                _na_x_mode_dx = "Range Search (current)" if _is2a else "범위 탐색 (현재)"

            if st.button(("Run Forward Optimization Search" if _is2a else "순방향 최적화 탐색 실행"), type="primary", key="main_run_dx2"):
                base_x_dx = []
                for v in X_list:
                    v_low = v.lower()
                    _lo_b, _hi_b = st.session_state[f'sim_tgt_{v_low}_s_val']
                    base_x_dx.append(float((_lo_b + _hi_b) / 2))
                q_base_dx = st.session_state['scaler'].transform(pd.DataFrame([base_x_dx], columns=X_list))[0]

                _na_x_vars_opt_dx = st.session_state.get('na_x_vars', [])
                _na_x_means_dx = {}
                if any(k in st.session_state.get('sim_dx_na_opt_mode', '') for k in ("평균값 고정", "Fix at Mean")) and _na_x_vars_opt_dx:
                    _df_ref_dx = st.session_state.get('df_imputed_ref')
                    if _df_ref_dx is not None:
                        for _nxv in _na_x_vars_opt_dx:
                            if _nxv in _df_ref_dx.columns:
                                _na_x_means_dx[_nxv] = float(pd.to_numeric(_df_ref_dx[_nxv], errors='coerce').mean())
                    st.sidebar.info(f"N/A 변수 {len(_na_x_means_dx)}개 평균값 고정 적용 (설계/공정 변수 타겟팅)")

                df_train_dx = st.session_state.get('df_imputed_ref')
                _na_tgts_dx_opt = st.session_state.get('na_spec_targets', [])
                dx_feasibility = {}
                if df_train_dx is not None:
                    q_train_dx = st.session_state['scaler'].transform(df_train_dx[X_list])
                    for tgt in target_vars:
                        if tgt in _na_tgts_dx_opt:
                            continue
                        mk = f'model_{tgt.lower()}'
                        if st.session_state[mk] is not None:
                            preds_dx = st.session_state[mk].predict(q_train_dx)
                            pred_min_dx, pred_max_dx = float(np.min(preds_dx)), float(np.max(preds_dx))
                            t_lo_dx, t_hi_dx = spec_limits.get(tgt, (0, 1))
                            overlap_dx = max(0.0, min(pred_max_dx, t_hi_dx) - max(pred_min_dx, t_lo_dx))
                            spec_span_dx = max(t_hi_dx - t_lo_dx, 1e-9)
                            pred_span_dx = max(pred_max_dx - pred_min_dx, 1e-9)
                            overlap_ratio_dx = overlap_dx / min(spec_span_dx, pred_span_dx)
                            dx_feasibility[tgt] = {
                                'pred_min': pred_min_dx, 'pred_max': pred_max_dx,
                                'overlap_ratio': overlap_ratio_dx,
                                'weight': max(0.05, min(1.0, overlap_ratio_dx))
                            }
                        else:
                            dx_feasibility[tgt] = {'pred_min': 0, 'pred_max': 0, 'overlap_ratio': 1.0, 'weight': 1.0}
                else:
                    for tgt in target_vars:
                        if tgt in _na_tgts_dx_opt:
                            continue
                        dx_feasibility[tgt] = {'pred_min': 0, 'pred_max': 0, 'overlap_ratio': 1.0, 'weight': 1.0}
                st.session_state['sim_dx_feasibility'] = dx_feasibility

                infeasible_count_dx = sum(1 for v in dx_feasibility.values() if v['overlap_ratio'] < 0.1)
                lambda_reg_dx = 0.5 + (infeasible_count_dx * 0.3)

                def dx_target_loss(x):
                    df_x = pd.DataFrame([x], columns=X_list)
                    q = st.session_state['scaler'].transform(df_x)
                    total_loss = 0.0
                    for tgt in target_vars:
                        if tgt in _na_tgts_dx_opt:
                            continue
                        model_key = f'model_{tgt.lower()}'
                        if st.session_state[model_key] is None:
                            continue
                        pred = st.session_state[model_key].predict(q)[0]
                        t_range = spec_limits.get(tgt, (None, None))
                        if t_range[0] is None:
                            continue
                        w = dx_feasibility.get(tgt, {}).get('weight', 1.0)
                        if w >= 0.99:
                            boundary_loss = (max(0, t_range[0] - pred) + max(0, pred - t_range[1]))**2
                        else:
                            nearest_spec = t_range[0] if pred < t_range[0] else (t_range[1] if pred > t_range[1] else pred)
                            boundary_loss = w * (pred - nearest_spec)**2
                        total_loss += boundary_loss
                    dist_penalty = np.sum((q[0] - q_base_dx)**2)
                    return total_loss + (lambda_reg_dx * dist_penalty)

                init_x_dx = list(base_x_dx)
                if _na_x_means_dx:
                    for _i, _v in enumerate(X_list):
                        if _v in _na_x_means_dx:
                            init_x_dx[_i] = _na_x_means_dx[_v]

                bands_dx = []
                for _v in X_list:
                    if _na_x_means_dx and _v in _na_x_means_dx:
                        _mv = _na_x_means_dx[_v]
                        bands_dx.append((_mv, _mv + 1e-9))
                    else:
                        _lo_b2, _hi_b2 = st.session_state[f'sim_tgt_{_v.lower()}_s_val']
                        bands_dx.append((float(_lo_b2), float(_hi_b2)))

                algorithms_dx = ['L-BFGS-B', 'SLSQP', 'Powell', 'Nelder-Mead']
                best_loss_dx = float('inf')
                best_res_dx = None
                selected_algo_dx = 'SLSQP'
                algo_loss_dict_dx = {}

                _is2 = st.session_state.get('lang', 'KO') == 'EN'
                dx_progress_bar = st.progress(0, text=("Preparing forward optimization search... (0%)" if _is2 else "순방향 최적화 탐색 준비 중... (0%)"))
                total_algos_dx = len(algorithms_dx)
                for a_idx, algo in enumerate(algorithms_dx):
                    dx_progress_pct = int((a_idx / total_algos_dx) * 100)
                    _prog_txt_dx = (f" Searching algorithm ({a_idx+1}/{total_algos_dx}): {algo} ({dx_progress_pct}%)" if _is2
                                    else f" 알고리즘 탐색 중 ({a_idx+1}/{total_algos_dx}): {algo} ({dx_progress_pct}%)")
                    dx_progress_bar.progress(a_idx / total_algos_dx, text=_prog_txt_dx)
                    try:
                        if algo in ['L-BFGS-B', 'SLSQP']: res_temp = minimize(dx_target_loss, init_x_dx, method=algo, bounds=bands_dx)
                        else: res_temp = minimize(dx_target_loss, init_x_dx, method=algo)
                        final_x = np.clip(res_temp.x, [b[0] for b in bands_dx], [b[1] for b in bands_dx])
                        current_score_loss = dx_target_loss(final_x)
                        algo_loss_dict_dx[algo] = round(float(current_score_loss), 6)
                        if current_score_loss < best_loss_dx:
                            best_loss_dx = current_score_loss
                            best_res_dx = res_temp
                            best_res_dx.x = final_x
                            selected_algo_dx = algo
                    except Exception:
                        algo_loss_dict_dx[algo] = None
                        continue

                dx_progress_bar.progress(1.0, text=(f"✅ Optimization complete (100%) - Selected algorithm: {selected_algo_dx}" if _is2
                                                      else f"✅ 최적화 완료 (100%) - 선택된 알고리즘: {selected_algo_dx}"))
                q_opt_dx = st.session_state['scaler'].transform(pd.DataFrame([best_res_dx.x], columns=X_list))

                update_dx_dict = {
                    'sim_dx_result_x': best_res_dx.x,
                    'sim_dx_confidence': round(max(0.0, 100.0 - (best_loss_dx * 5)), 1),
                    'sim_dx_best_algorithm': selected_algo_dx,
                    'sim_dx_algo_loss_dict': algo_loss_dict_dx,
                    'sim_dx_na_mode_used': "평균값 고정" if _na_x_means_dx else "범위 탐색",
                    'sim_dx_na_means_used': _na_x_means_dx,
                    'ai_analysis_result': None
                }
                for tgt in target_vars:
                    model_key = f'model_{tgt.lower()}'
                    if st.session_state[model_key] is not None: update_dx_dict[f'sim_dx_pred_{tgt.lower()}'] = float(st.session_state[model_key].predict(q_opt_dx)[0])
                    else: update_dx_dict[f'sim_dx_pred_{tgt.lower()}'] = 0.0
                st.session_state.update(update_dx_dict)
                st.rerun()

        with sim_r:
                    if st.session_state.get('sim_dx_result_x') is not None:
                        _is2b = st.session_state.get('lang', 'KO') == 'EN'
                        st.markdown(f"<div class='glass-card'><div class='glass-card-title' style='color:#3b82f6;'>{L_G['pred_title']}</div>", unsafe_allow_html=True)
                        with st.expander((f"▸ Forward Optimization Algorithm Competition  |  Selected: {st.session_state['sim_dx_best_algorithm']}" if _is2b
                                           else f"▸ 순방향 최적화 알고리즘 경쟁 결과  |  채택: {st.session_state['sim_dx_best_algorithm']}"), expanded=False):
                            _algo_info_dx = ({
                                'L-BFGS-B':    'Gradient-based, bound-constrained',
                                'SLSQP':       'Gradient-based, bound-constrained',
                                'Powell':      'Directional search, clipped',
                                'Nelder-Mead': 'Simplex-based, clipped',
                            } if _is2b else {
                                'L-BFGS-B':    '기울기 기반, 경계 준수',
                                'SLSQP':       '기울기 기반, 경계 준수',
                                'Powell':      '방향 탐색, clip 처리',
                                'Nelder-Mead': '도형 변형, clip 처리',
                            })
                            _sel_dx = st.session_state['sim_dx_best_algorithm']
                            _loss_d_dx = st.session_state.get('sim_dx_algo_loss_dict', {})
                            _rows_algo_dx = ""
                            for _a, _m in _algo_info_dx.items():
                                _is_sel_dx = _a == _sel_dx
                                _bg_dx = "#0a2010" if _is_sel_dx else "#0d1f3c"
                                _nc_dx = "#10b981" if _is_sel_dx else "#94a3b8"
                                _star_dx = "★ " if _is_sel_dx else ""
                                _lv_dx = _loss_d_dx.get(_a)
                                if _lv_dx is None:
                                    _loss_str_dx = f"<span style='color:#475569;'>{'Failed' if _is2b else '실행 실패'}</span>"
                                    _conf_str_dx = "<span style='color:#475569;'>—</span>"
                                    _bar_w_dx = 0
                                else:
                                    _conf_v_dx = round(max(0.0, 100.0 - (_lv_dx * 5)), 1)
                                    _lc_dx = "#10b981" if _lv_dx < 0.01 else "#f59e0b" if _lv_dx < 1.0 else "#f87171"
                                    _loss_str_dx = f"<span style='color:{_lc_dx};font-weight:700;font-family:monospace;'>{_lv_dx:.4f}</span>"
                                    _conf_str_dx = f"<span style='color:{_lc_dx};font-weight:700;'>{_conf_v_dx}%</span>"
                                    _bar_w_dx = min(int(_conf_v_dx), 100)
                                _badge_dx = (f"<span style='background:#0a2010;color:#10b981;font-size:0.68rem;padding:1px 6px;border-radius:3px;'>✅ {'Selected' if _is2b else '채택'}</span>" if _is_sel_dx else "")
                                _bar_dx = f"<div style='background:#1e293b;border-radius:2px;height:6px;margin-top:2px;'><div style='width:{_bar_w_dx}%;background:{'#10b981' if _bar_w_dx>80 else '#f59e0b' if _bar_w_dx>50 else '#f87171'};height:6px;border-radius:2px;'></div></div>"
                                _rows_algo_dx += (
                                    f"<tr style='background:{_bg_dx};'>"
                                    f"<td style='padding:6px 8px;color:{_nc_dx};font-weight:700;font-family:monospace;white-space:nowrap;'>{_star_dx}{_a}</td>"
                                    f"<td style='padding:6px 8px;color:#64748b;font-size:0.72rem;'>{_m}</td>"
                                    f"<td style='padding:6px 8px;text-align:center;'>{_loss_str_dx}</td>"
                                    f"<td style='padding:6px 8px;min-width:80px;'>{_conf_str_dx}{_bar_dx}</td>"
                                    f"<td style='padding:6px 8px;text-align:center;'>{_badge_dx}</td>"
                                    f"</tr>"
                                )
                            _th_a1, _th_a2, _th_a3, _th_a4, _th_a5 = (("Algorithm", "Method", "Loss ↓lower is better", "Confidence", "Selected") if _is2b
                                                                        else ("알고리즘", "방식", "손실값 ↓낮을수록 좋음", "신뢰도", "채택"))
                            _algo_footer_dx = ("The algorithm with the lowest loss is auto-selected. &nbsp;Confidence = max(0, 100 − loss×5)" if _is2b
                                                else "손실값이 가장 낮은 알고리즘이 자동 채택됩니다. &nbsp;신뢰도 = max(0, 100 − 손실값×5)")
                            st.markdown(
                                f"<div style='background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;padding:12px 14px;'>"
                                f"<table style='width:100%;border-collapse:collapse;'>"
                                f"<thead><tr style='border-bottom:1px solid #1e3a5f;'>"
                                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:left;'>{_th_a1}</th>"
                                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;'>{_th_a2}</th>"
                                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:center;'>{_th_a3}</th>"
                                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;'>{_th_a4}</th>"
                                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:center;'>{_th_a5}</th>"
                                f"</tr></thead>"
                                f"<tbody>{_rows_algo_dx}</tbody>"
                                f"</table>"
                                f"<div style='margin-top:8px;font-size:0.72rem;color:#475569;'>{_algo_footer_dx}</div>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                        st.markdown("<div style='margin: -10px 0 10px 0;'></div>", unsafe_allow_html=True)
                        with st.expander(("▸ Design/Process Variable Target Range Precision" if _is2b else "▸ 설계/공정 변수 목표 범위 정밀도"), expanded=False):
                            st.markdown(
                                "<div style='font-size:0.72rem;color:#64748b;margin-bottom:6px;'>"
                                +
                                (
                                "How narrowly (precisely) the user-set target range is specified, relative to "
                                "the total available range (spec or measured data min~max). Closer to 1.0 means "
                                "a tighter target; closer to 0 means a broader range search."
                                if _is2b else
                                "전체 가용 범위(스펙 또는 데이터 실측 min~max) 대비, 사용자가 설정한 목표 범위가 "
                                "얼마나 좁게(정밀하게) 지정됐는지 보여줍니다. 1.0에 가까울수록 타이트한 목표, "
                                "0에 가까울수록 넓은 범위 탐색을 의미합니다."
                                )
                                + "</div>", unsafe_allow_html=True
                            )
                            _msr_rows_dx = ""
                            for _v in X_list:
                                _v_low = _v.lower()
                                _cur_range_dx = st.session_state.get(f'sim_tgt_{_v_low}_s_val', chosen_bounds_dx.get(_v, (0.0, 1.0)))
                                _full_lo, _full_hi = chosen_bounds_dx.get(_v, (0.0, 1.0))
                                _full_span_dx = max(_full_hi - _full_lo, 1e-9)
                                _tgt_span_dx = max(float(_cur_range_dx[1]) - float(_cur_range_dx[0]), 0.0)
                                _r2f_dx = max(0.0, min(1.0, 1.0 - (_tgt_span_dx / _full_span_dx)))
                                _bar_r2_dx = min(int(_r2f_dx*100), 100)
                                _r2_color_dx = '#10b981' if _r2f_dx>=0.9 else '#f59e0b' if _r2f_dx>=0.7 else '#f87171'
                                _msr_rows_dx += (
                                    f"<tr>"
                                    f"<td style='padding:5px 8px;color:#e2e8f0;font-weight:700;'>{_v}</td>"
                                    f"<td style='padding:5px 8px;min-width:120px;'>"
                                    f"<span style='color:{_r2_color_dx};font-weight:700;font-family:monospace;'>{_r2f_dx:.2f}</span>"
                                    f"<div style='background:#1e293b;border-radius:2px;height:4px;margin-top:2px;'>"
                                    f"<div style='width:{_bar_r2_dx}%;background:{_r2_color_dx};height:4px;border-radius:2px;'></div></div>"
                                    f"</td>"
                                    f"</tr>"
                                )
                            _th_p1, _th_p2 = ("Target", "Target Range Precision") if _is2b else ("타겟", "목표 범위 정밀도")
                            st.markdown("<div style='max-height:380px; overflow-y:auto;'>", unsafe_allow_html=True)
                            st.markdown(
                                f"<div style='background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;padding:12px 14px;'>"
                                f"<table style='width:100%;border-collapse:collapse;'>"
                                f"<thead><tr style='border-bottom:1px solid #1e3a5f;'>"
                                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:left;'>{_th_p1}</th>"
                                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;'>{_th_p2}</th>"
                                f"</tr></thead>"
                                f"<tbody>{_msr_rows_dx}</tbody>"
                                f"</table>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                            st.markdown("</div>", unsafe_allow_html=True)

                        _ox_status_dx = st.session_state.get('sim_dx_result_x')
                        _na_means_status_dx = st.session_state.get('sim_dx_na_means_used', {})
                        _loss_rows_dx = ""
                        _total_loss_dx = 0.0
                        if _ox_status_dx is not None:
                            for _i, _v in enumerate(X_list):
                                _v_low = _v.lower()
                                _pv = float(_ox_status_dx[_i])
                                if _v in _na_means_status_dx:
                                    _fixed_lbl_dx = "Fixed at Mean" if _is2b else "평균값 고정"
                                    _fixed_badge_dx = "Fixed" if _is2b else "고정"
                                    _loss_rows_dx += (
                                        f"<tr>"
                                        f"<td style='padding:4px 8px;color:#64748b;font-weight:600;'>{_v}</td>"
                                        f"<td style='padding:4px 8px;color:#64748b;text-align:center;'>{_pv:.3f}</td>"
                                        f"<td style='padding:4px 8px;color:#475569;text-align:center;font-size:0.72rem;'>{_fixed_lbl_dx}</td>"
                                        f"<td style='padding:4px 8px;color:#475569;text-align:center;'>—</td>"
                                        f"<td style='padding:4px 8px;text-align:center;'><span style='background:#1e293b;color:#f59e0b;font-size:0.7rem;padding:1px 6px;border-radius:3px;'>{_fixed_badge_dx}</span></td>"
                                        f"</tr>"
                                    )
                                    continue
                                _tgt_range = st.session_state.get(f'sim_tgt_{_v_low}_s_val', (None, None))
                                if _tgt_range[0] is None: continue
                                _lo, _hi = float(_tgt_range[0]), float(_tgt_range[1])
                                _over = max(0, _lo - _pv) + max(0, _pv - _hi)
                                _loss_v = _over ** 2
                                _total_loss_dx += _loss_v
                                _spec_str = f"{_lo:.3f}~{_hi:.3f}"
                                _in_spec = _lo - 1e-6 <= _pv <= _hi + 1e-6
                                _loss_color = "#10b981" if _loss_v == 0 else "#f87171"
                                if _is2b:
                                    _status_badge = (
                                        f"<span style='background:#0a2010;color:#10b981;font-size:0.7rem;padding:1px 6px;border-radius:3px;'>✅ Achieved</span>"
                                        if _in_spec else
                                        f"<span style='background:#2d0f0f;color:#f87171;font-size:0.7rem;padding:1px 6px;border-radius:3px;'>⚠️ Deviated</span>"
                                    )
                                else:
                                    _status_badge = (
                                        f"<span style='background:#0a2010;color:#10b981;font-size:0.7rem;padding:1px 6px;border-radius:3px;'>✅ 달성</span>"
                                        if _in_spec else
                                        f"<span style='background:#2d0f0f;color:#f87171;font-size:0.7rem;padding:1px 6px;border-radius:3px;'>⚠️ 이탈</span>"
                                    )
                                _loss_rows_dx += (
                                    f"<tr>"
                                    f"<td style='padding:4px 8px;color:#e2e8f0;font-weight:700;'>{_v}</td>"
                                    f"<td style='padding:4px 8px;color:#ffffff;text-align:center;font-weight:600;'>{_pv:.3f}</td>"
                                    f"<td style='padding:4px 8px;color:#94a3b8;text-align:center;font-size:0.8rem;'>{_spec_str}</td>"
                                    f"<td style='padding:4px 8px;color:{_loss_color};text-align:center;font-weight:600;'>{_loss_v:.4f}</td>"
                                    f"<td style='padding:4px 8px;text-align:center;'>{_status_badge}</td>"
                                    f"</tr>"
                                )
                        if _loss_rows_dx:
                            _loss_exp_title_dx = (f"▸ Loss Status by Target  |  Total Loss: {_total_loss_dx:.4f} {'✅' if _total_loss_dx < 0.01 else '⚠️'}" if _is2b
                                                    else f"▸ 타겟별 손실 현황  |  전체 손실: {_total_loss_dx:.4f} {'✅' if _total_loss_dx < 0.01 else '⚠️'}")
                            with st.expander(_loss_exp_title_dx, expanded=False):
                                st.markdown("<div style='max-height:380px; overflow-y:auto;'>", unsafe_allow_html=True)
                                _lh1d, _lh2d, _lh3d, _lh4d, _lh5d = (("Target", "Predicted", "Spec", "Loss", "Judgement") if _is2b
                                                                       else ("타겟", "예측값", "스펙", "손실", "판정"))
                                _total_loss_lbl_dx = "Total Loss" if _is2b else "전체 손실 합계"
                                _perfect_lbl_dx = "← Perfect ✅" if _is2b else "← 완벽 ✅"
                                st.markdown(
                                    f"<div style='background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;padding:12px 14px;'>"
                                    f"<table style='width:100%;border-collapse:collapse;'>"
                                    f"<thead><tr style='border-bottom:1px solid #1e3a5f;'>"
                                    f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:left;'>{_lh1d}</th>"
                                    f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:center;'>{_lh2d}</th>"
                                    f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:center;'>{_lh3d}</th>"
                                    f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:center;'>{_lh4d}</th>"
                                    f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:center;'>{_lh5d}</th>"
                                    f"</tr></thead>"
                                    f"<tbody>{_loss_rows_dx}</tbody>"
                                    f"</table>"
                                    f"<div style='border-top:1px solid #1e3a5f;margin-top:6px;padding-top:6px;"
                                    f"display:flex;justify-content:space-between;align-items:center;'>"
                                    f"<span style='font-size:0.75rem;color:#94a3b8;'>{_total_loss_lbl_dx}</span>"
                                    f"<span style='font-size:0.9rem;font-weight:700;color:{'#10b981' if _total_loss_dx < 0.01 else '#f87171'};'>"
                                    f"{_total_loss_dx:.4f} {_perfect_lbl_dx if _total_loss_dx < 0.01 else ''}</span>"
                                    f"</div></div>",
                                    unsafe_allow_html=True
                                )
                                st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                        # ── (스왑) 상단 카드: 예상 설계/공정 변수 값 (34개) ──
                        _card1_title_dx = "▣ Expected Design/Process Variable Values (Optimal Point Within Target Range)" if _is2b else "▣ 예상 설계/공정 변수 값 (목표 범위 내 최적점)"
                        st.markdown(f"<p style='font-size:0.82rem;font-weight:700;color:#10b981;margin:10px 0 6px 0;'>{_card1_title_dx}</p>", unsafe_allow_html=True)
                        ox_dx = st.session_state['sim_dx_result_x']
                        _na_x_res_dx = st.session_state.get('na_x_vars', [])
                        _na_mode_used_dx = st.session_state.get('sim_dx_na_mode_used', '범위 탐색')
                        cols_x_card_dx = st.columns(3)
                        for idx, v_name in enumerate(X_list):
                            _xv_dx = ox_dx[idx] if ox_dx[idx] is not None else 0.0
                            val_display = f"{_xv_dx:.3f}"
                            _is_fixed_dx = v_name in _na_x_res_dx and "평균값" in _na_mode_used_dx
                            if v_name in _na_x_res_dx:
                                _badge_txt_dx = ("Fixed" if _is_fixed_dx else "Spec N/A") if _is2b else ("고정값" if _is_fixed_dx else "스펙N/A")
                                _sub_color_dx = "#f59e0b" if _is_fixed_dx else "#64748b"
                                cols_x_card_dx[idx % 3].markdown(
                                    f"<div style='padding:8px;background:#1a1a2e;border:1px dashed #475569;"
                                    f"border-radius:4px;margin-bottom:6px;opacity:0.8;'>"
                                    f"<span style='color:#64748b;font-size:0.72rem;'>{v_name}</span>"
                                    f"<span style='float:right;background:#334155;color:{_sub_color_dx};font-size:0.65rem;"
                                    f"padding:1px 5px;border-radius:3px;'>{_badge_txt_dx}</span><br>"
                                    f"<strong style='font-size:1.05rem;color:#94a3b8;'>{val_display}</strong></div>",
                                    unsafe_allow_html=True
                                )
                            else:
                                cols_x_card_dx[idx % 3].markdown(
                                    f"<div style='padding:8px;background:#1e293b;border-radius:4px;margin-bottom:6px;'>"
                                    f"<span style='color:#94a3b8;font-size:0.72rem;'>{v_name}</span><br>"
                                    f"<strong style='font-size:1.05rem;color:#ffffff;'>{val_display}</strong></div>",
                                    unsafe_allow_html=True
                                )

                        st.metric(L_G['opt_conf'], f"{st.session_state['sim_dx_confidence']}%")

                        # 예측 데이터 다운로드 (설계/공정 변수 결과, 34개)
                        x_result_dict_dx = {X_list[i]: [ox_dx[i]] for i in range(len(X_list))}
                        df_xres_export_dx = pd.DataFrame(x_result_dict_dx)
                        col_pred_sel_dx, col_pred_trigger_dx = st.columns([1, 1])
                        with col_pred_sel_dx:
                            file_format_pred_dx = st.selectbox(L_G['dl_format'], ["Excel (.xlsx)", "Database (.db)"],
                                                            key="fmt_dx_xres", label_visibility="collapsed")
                        with col_pred_trigger_dx:
                            if "Excel" in file_format_pred_dx:
                                buffer_p_dx = io.BytesIO()
                                with pd.ExcelWriter(buffer_p_dx) as writer: df_xres_export_dx.to_excel(writer, index=False, sheet_name='Design_Process_Variables')
                                st.download_button(label=L_G['dl_btn_pred'], data=buffer_p_dx.getvalue(), file_name="dx_design_process_variables.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_btn_dx_xres_trigger")
                            else:
                                conn_p_dx = sqlite3.connect(":memory:")
                                df_xres_export_dx.to_sql("design_process_variables", conn_p_dx, index=False, if_exists="replace")
                                backup_conn_p_dx = sqlite3.connect("temp_dx_xres.db")
                                conn_p_dx.backup(backup_conn_p_dx); backup_conn_p_dx.close(); conn_p_dx.close()
                                with open("temp_dx_xres.db", "rb") as f: db_bytes_p_dx = f.read()
                                st.download_button(label=L_G['dl_btn_pred'], data=db_bytes_p_dx, file_name="dx_design_process_variables.db", mime="application/x-sqlite3", key="dl_btn_dx_xres_db_trigger")
                        st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)

                        # ── (스왑) 하단 카드: 예상 타겟 값 (품질, 스펙 달성 결과) ──
                        _card2_title_dx = "Expected Target Values (Quality Spec Achievement Result)" if _is2b else "예상 타겟 값 (품질 스펙 달성 결과)"
                        st.markdown(f"<div class='glass-card'><div class='glass-card-title' style='color:#10b981;'>{_card2_title_dx}</div>", unsafe_allow_html=True)
                        st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)
                        _na_tgts_dx = st.session_state.get('na_spec_targets', [])
                        cols_y_card_dx = st.columns(3)
                        for idx, tgt in enumerate(valid_tgts):
                            p_val = st.session_state[f'sim_dx_pred_{tgt.lower()}']
                            val_display = f"{p_val:.3f}" if isinstance(p_val, float) else "0.000"
                            if tgt in _na_tgts_dx:
                                cols_y_card_dx[idx % 3].markdown(
                                    f"<div style='padding:8px;background:#1a1a2e;border:1px dashed #475569;"
                                    f"border-radius:4px;margin-bottom:6px;opacity:0.8;'>"
                                    f"<span style='color:#64748b;font-size:0.72rem;'>Predicted {tgt}</span>"
                                    f"<span style='float:right;background:#334155;color:#94a3b8;font-size:0.65rem;"
                                    f"padding:1px 5px;border-radius:3px;'>N/A</span><br>"
                                    f"<strong style='font-size:1.05rem;color:#94a3b8;'>{val_display}</strong></div>",
                                    unsafe_allow_html=True
                                )
                            else:
                                _sp_lo, _sp_hi = spec_limits.get(tgt, (None, None))
                                if _sp_lo is not None and isinstance(p_val, float) and (p_val < _sp_lo or p_val > _sp_hi):
                                    _dev_txt_dx = "Deviated" if _is2b else "이탈"
                                    cols_y_card_dx[idx % 3].markdown(
                                        f"<div style='padding:8px;background:#2d0f0f;border:1px solid #ef4444;"
                                        f"border-radius:4px;margin-bottom:6px;'>"
                                        f"<span style='color:#f87171;font-size:0.72rem;'>⚠️ {tgt} [{_dev_txt_dx}]</span><br>"
                                        f"<strong style='font-size:1.05rem;color:#ef4444;'>{val_display}</strong>"
                                        f"<span style='color:#f87171;font-size:0.68rem;'> ({_sp_lo}~{_sp_hi})</span></div>",
                                        unsafe_allow_html=True
                                    )
                                else:
                                    cols_y_card_dx[idx % 3].markdown(
                                        f"<div style='padding:8px;background:#1e293b;border-radius:4px;margin-bottom:6px;'>"
                                        f"<span style='color:#94a3b8;font-size:0.72rem;'>Predicted {tgt}</span><br>"
                                        f"<strong style='font-size:1.05rem;color:#ffffff;'>{val_display}</strong></div>",
                                        unsafe_allow_html=True
                                    )
                        y_pred_dict_dx = {tgt: [st.session_state[f'sim_dx_pred_{tgt.lower()}']] for tgt in valid_tgts}
                        df_ypred_export_dx = pd.DataFrame(y_pred_dict_dx)
                        col_dl_sel_dx, col_dl_trigger_dx = st.columns([1, 1])
                        with col_dl_sel_dx:
                            file_format_dx = st.selectbox(L_G['dl_format'], ["Excel (.xlsx)", "Database (.db)"],
                                                       key="fmt_dx_ypred", label_visibility="collapsed")
                        with col_dl_trigger_dx:
                            if "Excel" in file_format_dx:
                                buffer_dx = io.BytesIO()
                                with pd.ExcelWriter(buffer_dx) as writer: df_ypred_export_dx.to_excel(writer, index=False, sheet_name='Predicted_Quality')
                                st.download_button(label=L_G['dl_btn_spec'], data=buffer_dx.getvalue(), file_name="dx_predicted_quality_targets.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_btn_dx_ypred_trigger")
                            else:
                                conn_dx = sqlite3.connect(":memory:")
                                df_ypred_export_dx.to_sql("predicted_quality_targets", conn_dx, index=False, if_exists="replace")
                                backup_conn_dx = sqlite3.connect("temp_dx_ypred.db")
                                conn_dx.backup(backup_conn_dx); backup_conn_dx.close(); conn_dx.close()
                                with open("temp_dx_ypred.db", "rb") as f: db_bytes_dx = f.read()
                                st.download_button(label=L_G['dl_btn_spec'], data=db_bytes_dx, file_name="dx_predicted_quality_targets.db", mime="application/x-sqlite3", key="dl_btn_dx_ypred_db_trigger")
                        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.get('sim_dx_result_x') is not None:
            _is2c = st.session_state.get('lang', 'KO') == 'EN'
            st.markdown(f"### □ {'Result Analysis & Diagnosis' if _is2c else '결과 분석 및 진단'}")
            has_warning_dx = False
            feasibility_info_dx = st.session_state.get('sim_dx_feasibility', {})

            result_cards_dx = []
            _na_tgts_res_dx = st.session_state.get('na_spec_targets', [])
            for tgt in valid_tgts:
                pred_val_dx = st.session_state.get(f'sim_dx_pred_{tgt.lower()}')
                if pred_val_dx is None: continue

                if tgt in _na_tgts_res_dx:
                    if _is2c:
                        result_cards_dx.append(('na',
                            f"⬜ **{tgt}** — Spec N/A\n\n"
                            f"Predicted: **{pred_val_dx:.3f}**\n\n"
                            f"Excluded from optimization (no spec set)"
                        ))
                    else:
                        result_cards_dx.append(('na',
                            f"⬜ **{tgt}** — 스펙 N/A\n\n"
                            f"예측값: **{pred_val_dx:.3f}**\n\n"
                            f"스펙 미설정으로 최적화 계산에서 제외"
                        ))
                    continue

                t_range_dx = spec_limits.get(tgt, (None, None))
                feas_dx = feasibility_info_dx.get(tgt, {})
                overlap_dx2 = feas_dx.get('overlap_ratio', 1.0)
                pred_min_dx2 = feas_dx.get('pred_min', None)
                pred_max_dx2 = feas_dx.get('pred_max', None)

                if overlap_dx2 < 0.1 and pred_min_dx2 is not None:
                    if _is2c:
                        result_cards_dx.append(('error',
                            f"🚫 **{tgt}** Infeasible\n\n"
                            f"Predicted range: **{pred_min_dx2:.2f}~{pred_max_dx2:.2f}**\n\n"
                            f"Target spec: {t_range_dx[0]}~{t_range_dx[1]}\n\n"
                            f"Prediction: **{pred_val_dx:.3f}**"
                        ))
                    else:
                        result_cards_dx.append(('error',
                            f"🚫 **{tgt}** 달성 불가\n\n"
                            f"예측 범위: **{pred_min_dx2:.2f}~{pred_max_dx2:.2f}**\n\n"
                            f"설정 스펙: {t_range_dx[0]}~{t_range_dx[1]}\n\n"
                            f"예측 결과: **{pred_val_dx:.3f}**"
                        ))
                    has_warning_dx = True
                elif t_range_dx[0] is not None and t_range_dx[1] is not None and (pred_val_dx > t_range_dx[1] or pred_val_dx < t_range_dx[0]):
                    if _is2c:
                        result_cards_dx.append(('warning', f"⚠️ **{tgt}** Out of Spec\n\nPredicted **{pred_val_dx:.3f}**\n\nTarget spec {t_range_dx[0]}~{t_range_dx[1]}"))
                    else:
                        result_cards_dx.append(('warning', f"⚠️ **{tgt}** 스펙 이탈\n\n예측값 **{pred_val_dx:.3f}**\n\n설정 스펙 {t_range_dx[0]}~{t_range_dx[1]}"))
                    has_warning_dx = True
                else:
                    if _is2c:
                        result_cards_dx.append(('success', f"✅ **{tgt}** Achieved\n\nPredicted **{pred_val_dx:.3f}**\n\nTarget spec {t_range_dx[0]}~{t_range_dx[1]}"))
                    else:
                        result_cards_dx.append(('success', f"✅ **{tgt}** 정상 도달\n\n예측값 **{pred_val_dx:.3f}**\n\n설정 스펙 {t_range_dx[0]}~{t_range_dx[1]}"))

            RESULT_COLS_PER_ROW_DX = 5
            for i in range(0, len(result_cards_dx), RESULT_COLS_PER_ROW_DX):
                row_cards_dx = result_cards_dx[i:i + RESULT_COLS_PER_ROW_DX]
                grid_cols_dx = st.columns(RESULT_COLS_PER_ROW_DX)
                for gc_dx, (rtype_dx, msg_dx) in zip(grid_cols_dx, row_cards_dx):
                    with gc_dx:
                        if rtype_dx == 'na':
                            _p_dx = msg_dx.replace('**','').split('\n\n')
                            _t_dx = _p_dx[0] if _p_dx else ''
                            gc_dx.markdown(
                                f"<div style='background:#1e293b;border:1px dashed #475569;"
                                f"border-radius:6px;padding:8px 12px;font-size:0.82rem;margin-bottom:6px;'>"
                                f"<span style='color:#94a3b8;font-weight:700;font-size:0.80rem;'>{_t_dx}</span><br>"
                                f"<span style='color:#64748b;font-size:0.78rem;'>&nbsp;</span>"
                                f"<span style='color:#475569;font-size:0.72rem;display:block;'>&nbsp;</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            _parts_dx = msg_dx.split('\n\n')
                            _title_dx = _parts_dx[0].replace('**','') if _parts_dx else ''
                            _pred_dx  = _parts_dx[1].replace('**','') if len(_parts_dx) > 1 else ''
                            _spec_dx  = _parts_dx[2].replace('**','') if len(_parts_dx) > 2 else ''
                            if rtype_dx == 'success':
                                _bg_dx2, _bd_dx2, _tc_dx2, _vc_dx2 = '#0a2010', '#10b981', '#6ee7b7', '#ffffff'
                            elif rtype_dx == 'warning':
                                _bg_dx2, _bd_dx2, _tc_dx2, _vc_dx2 = '#1a1200', '#f59e0b', '#fcd34d', '#ffffff'
                            else:
                                _bg_dx2, _bd_dx2, _tc_dx2, _vc_dx2 = '#2d0f0f', '#ef4444', '#f87171', '#ffffff'
                            gc_dx.markdown(
                                f"<div style='background:{_bg_dx2};border:1px solid {_bd_dx2};"
                                f"border-radius:6px;padding:8px 12px;font-size:0.82rem;margin-bottom:6px;'>"
                                f"<span style='color:{_tc_dx2};font-weight:700;font-size:0.80rem;'>{_title_dx}</span><br>"
                                f"<span style='color:{_vc_dx2};font-size:0.78rem;'>{_pred_dx}</span>"
                                f"<span style='color:#64748b;font-size:0.72rem;display:block;'>{_spec_dx}</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

            if has_warning_dx:
                if _is2c:
                    st.info(
                        "**Reliability-Guaranteed Optimization Applied**\n\n"
                        "Infeasible targets are converted to soft weights in the loss function, so the "
                        "optimizer is automatically adjusted to avoid drifting to extreme values "
                        "(Feasibility-Aware Adaptive Optimization). The current result is the most realistic "
                        "best condition derivable within the design/process target range you specified."
                    )
                else:
                    st.info(
                        "**신뢰성 보장 최적화 적용됨**\n\n"
                        "달성 불가 타겟은 손실 함수에서 소프트 가중치(Soft Weight)로 전환되어, "
                        "옵티마이저가 극단적인 값으로 이탈하지 않도록 자동 조정되었습니다(Feasibility-Aware Adaptive Optimization). "
                        "현재 결과는 설정하신 설계/공정 목표 범위 내에서 가장 현실적으로 도출된 최선의 조건입니다."
                    )
            else:
                if _is2c:
                    st.info("□  **All Valid Targets Achieved**\n\nWithin the design/process variable target range you specified, all targets are predicted to fully reach the quality spec range.")
                else:
                    st.info("□  **전체 유효 타겟 정상 도달**\n\n설정하신 설계/공정 변수 목표 범위 안에서 모든 타겟이 품질 스펙 범위 내에 완벽히 도달할 수 있는 것으로 분석되었습니다.")

        if st.session_state.get('sim_dx_result_x') is not None:
            st.markdown("---")
            st.markdown(f"### □ {'Process Improvement Guide (Result Diagnosis Report)' if _is2c else '공정 개선 가이드 (결과 진단 리포트)'}")
            _fi_col2, _ = st.columns([1.3, 1.2], gap="large")
            with _fi_col2:
                _fi_btn2 = st.button(("□ Generate FI Diagnosis + LLM Integrated Process Guide" if _is2c else "□ FI 진단 + LLM 통합 공정 가이드 생성"), key="btn_combined_tab2", type="primary")
            if _fi_btn2:
                with st.spinner("Generating Feature Importance analysis + LLM guideline integration... (10-20 sec)" if _is2c else "Feature Importance 분석 + LLM 가이드라인 통합 생성 중... (10~20초 소요)"):
                    for _tgt in valid_tgts:
                        st.session_state[f'dx_{_tgt.lower()}_s_val'] = spec_limits.get(_tgt, (0.0, 1.0))
                    dx_pred_kpis_now = {tgt: st.session_state[f'sim_dx_pred_{tgt.lower()}'] for tgt in valid_tgts if st.session_state.get(f'sim_dx_pred_{tgt.lower()}') is not None}
                    current_p_specs_dx = {X_list[i]: st.session_state['sim_dx_result_x'][i] for i in range(len(X_list))}
                    dx_combined = generate_combined_report(
                        process_specs=current_p_specs_dx,
                        predicted_kpis=dx_pred_kpis_now,
                        feasibility_info=st.session_state.get('sim_dx_feasibility', {}),
                        confidence_score=st.session_state.get('sim_dx_confidence', 0),
                        mode="Simulation",
                        range_key_prefix='dx_'
                    )
                    st.session_state['sim_dx_combined_report_text'] = dx_combined

            if st.session_state.get('sim_dx_combined_report_text'):
                with st.expander(("□ Integrated Process Analysis Report (FI Diagnosis + LLM)" if _is2c else "□ 통합 공정 분석 보고서 (FI 진단 + LLM)"), expanded=True):
                    st.markdown(
                        f"<div class='scrollable-box' style='height:600px;'>"
                        f"{st.session_state['sim_dx_combined_report_text'].replace(chr(10), '<br>')}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    st.download_button(
                        label=("□ Download Integrated Report" if _is2c else "□ 통합 보고서 다운로드"),
                        data=st.session_state['sim_dx_combined_report_text'],
                        file_name=f"JOINT_AI_DX_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain",
                        key="dl_combined_tab2"
                    )

    with tab3:
        _df3  = st.session_state.get('df_caulking', pd.DataFrame())
        _fi3  = st.session_state.get('feature_importance', {})
        _mm3  = st.session_state.get('model_metadata', {})
        _vt3  = st.session_state.get('valid_target_vars', target_vars)
        is_en3 = st.session_state.get('lang', 'KO') == 'EN'

        # ── ① 원시 데이터 ───────────────────────────────────────────
        _raw_lbl = "+ Raw Data" if is_en3 else "+ 원시 데이터"
        with st.expander(_raw_lbl, expanded=False):
            if not _df3.empty:
                st.markdown(
                    "<p style='color:#cbd5e1;font-size:0.83rem;'>"
                    + ("All accumulated production log data. Defect columns represent measured values." if is_en3 else
                       "업로드된 파일과 이력이 누적된 전체 데이터입니다. 불량/품질 컬럼은 실측값입니다.")
                    + "</p>", unsafe_allow_html=True
                )
                st.dataframe(_df3, use_container_width=True)
            else:
                st.info("데이터가 없습니다. 사이드바에서 데이터를 업로드하세요." if not is_en3 else "No data. Upload files from the sidebar.")

        # ── ② 품질 타겟 & 이탈 분포 ────────────────────────────────
        _dist_lbl = "+ Quality Distribution & Out-of-Spec Analysis" if is_en3 else "+ 품질 타겟 분포 & 이탈 분석"
        with st.expander(_dist_lbl, expanded=False):
            st.markdown(
                "<details style='margin-bottom:12px;'>"
                f"<summary style='font-size:0.72rem;color:#64748b;cursor:pointer;padding:4px 0;list-style:none;'>{'▶ Learn what this means and how it is analyzed' if is_en3 else '▶ 이 항목의 의미와 분석 방법 보기'}</summary>"
                "<div style='background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;padding:12px 16px;margin-top:8px;font-size:0.72rem;color:#cbd5e1;line-height:1.8;'>"
                + (
                "<b style='color:#38bdf8;'>Quality Distribution & Out-of-Spec Analysis</b>?<br>"
                "Shows how the measured values of a selected quality item (BT, RT, ABAMS, etc.) are distributed, "
                "and highlights samples outside the set spec range in red.<br><br>"
                "<b style='color:#a3e635;'>Key things to check</b><br>"
                "· <b>Out-of-spec sample count & ratio</b> — number of spec deviations relative to the total. Immediate process review is recommended if over 10%.<br>"
                "· <b>Distribution skew</b> — if most values are clustered near one edge of the spec, the process center may need adjustment.<br>"
                "· <b>Red zone</b> — the out-of-spec region. Check the correlation heatmap for the process variables driving that range."
                if is_en3 else
                "<b style='color:#38bdf8;'>품질 타겟 분포 & 이탈 분석</b>이란?<br>"
                "선택한 품질 항목(BT, RT, ABAMS 등)의 실측값이 어떤 범위에 얼마나 분포하는지 보여주며, "
                "설정된 스펙 범위를 벗어난 샘플을 빨간색으로 강조합니다.<br><br>"
                "<b style='color:#a3e635;'>주요 확인 포인트</b><br>"
                "· <b>이탈 샘플 수 & 비율</b> — 전체 대비 스펙 이탈 건수. 10% 초과 시 공정 즉시 점검 권장.<br>"
                "· <b>분포 치우침</b> — 대부분의 값이 스펙 한쪽 끝에 몰려 있으면 공정 중심값 조정이 필요합니다.<br>"
                "· <b>빨간색 구간</b> — 스펙 이탈 구간. 해당 범위를 유발하는 공정 변수를 상관관계 히트맵에서 확인하세요."
                )
                + "</div></details>",
                unsafe_allow_html=True
            )
            if _df3.empty:
                st.info("데이터 없음" if not is_en3 else "No data available.")
            else:
                _tgt_cols = [c for c in _df3.columns if c in target_vars]
                if not _tgt_cols:
                    st.info("품질 타겟 컬럼이 없습니다." if not is_en3 else "No quality target columns found in data.")
                else:
                    _sel_lbl = "불량 항목 선택" if not is_en3 else "Select Quality Target"
                    _sel_h = st.selectbox(_sel_lbl, _tgt_cols,
                                          format_func=lambda k: TARGET_GLOSSARY.get(k, k),
                                          key="t3_hist_sel")
                    _hd = pd.to_numeric(_df3[_sel_h], errors='coerce').dropna()
                    if _hd.empty:
                        st.warning(f"{_sel_h} 데이터가 없습니다." if not is_en3 else f"No data for {_sel_h}.")
                    else:
                        # 스펙 범위
                        _sp_lo, _sp_hi = spec_limits.get(_sel_h, (None, None))
                        _hm  = float(_hd.mean())
                        _hs  = float(_hd.std())
                        _hn  = len(_hd)
                        # 이탈 샘플 수
                        if _sp_lo is not None and _sp_hi is not None:
                            _out = int((((_hd < _sp_lo) | (_hd > _sp_hi)).sum()))
                            _out_pct = _out / _hn * 100
                            if is_en3:
                                _insight_txt = (
                                    f"Mean: <b style='color:#38bdf8;'>{_hm:.3f}</b> &nbsp;|&nbsp; "
                                    f"Std Dev: <b>{_hs:.3f}</b> &nbsp;|&nbsp; "
                                    f"Spec Range: <b>{_sp_lo}~{_sp_hi}</b> &nbsp;|&nbsp; "
                                    f"Out-of-Spec: <b style='color:{'#f87171' if _out_pct>20 else '#ffab00' if _out_pct>5 else '#10b981'};'>"
                                    f"{_out} ({_out_pct:.1f}%)</b>"
                                )
                                _advice = ("⚠️ High out-of-spec rate. Prioritize checking this quality target." if _out_pct > 20 else
                                           "Some samples are in the caution zone. Strengthen monitoring." if _out_pct > 5 else
                                           "Most samples are within the spec range.")
                            else:
                                _insight_txt = (
                                    f"평균: <b style='color:#38bdf8;'>{_hm:.3f}</b> &nbsp;|&nbsp; "
                                    f"표준편차: <b>{_hs:.3f}</b> &nbsp;|&nbsp; "
                                    f"스펙 범위: <b>{_sp_lo}~{_sp_hi}</b> &nbsp;|&nbsp; "
                                    f"이탈 샘플: <b style='color:{'#f87171' if _out_pct>20 else '#ffab00' if _out_pct>5 else '#10b981'};'>"
                                    f"{_out}개 ({_out_pct:.1f}%)</b>"
                                )
                                _advice = ("⚠️ 이탈 비율이 높습니다. 해당 품질 타겟을 우선 점검하세요." if _out_pct > 20 else
                                           "주의 구간 샘플이 일부 있습니다. 모니터링을 강화하세요." if _out_pct > 5 else
                                           "대부분의 샘플이 스펙 범위 내에 있습니다.")
                        else:
                            _out, _out_pct = 0, 0.0
                            if is_en3:
                                _insight_txt = f"Mean: <b style='color:#38bdf8;'>{_hm:.3f}</b> &nbsp;|&nbsp; Std Dev: <b>{_hs:.3f}</b> &nbsp;|&nbsp; n={_hn}"
                                _advice = "No spec range set — showing data distribution only."
                            else:
                                _insight_txt = f"평균: <b style='color:#38bdf8;'>{_hm:.3f}</b> &nbsp;|&nbsp; 표준편차: <b>{_hs:.3f}</b> &nbsp;|&nbsp; n={_hn}"
                                _advice = "스펙 범위 미설정 — 데이터 분포만 표시합니다."

                        st.markdown(
                            f"<div style='background:#0a1628;border:1px solid #1e3a5f;border-left:3px solid #38bdf8;"
                            f"border-radius:6px;padding:10px 14px;margin-bottom:10px;font-size:0.82rem;'>"
                            f"<span style='color:#e2e8f0;'>{_insight_txt}</span><br>"
                            f"<span style='color:#cbd5e1;font-size:0.78rem;'>→ {_advice}</span></div>",
                            unsafe_allow_html=True
                        )
                        # 히스토그램
                        _bins_n = min(15, max(5, _hn // 3))
                        _cnt, _edg = np.histogram(_hd.values.astype(float), bins=_bins_n)
                        _mx_c = int(_cnt.max()) if _cnt.max() > 0 else 1
                        _bar_h = ""
                        for _i, _c in enumerate(_cnt):
                            _lo, _hi2 = float(_edg[_i]), float(_edg[_i+1])
                            _mid = (_lo + _hi2) / 2
                            _bp  = int(_c) / _mx_c * 100
                            # 스펙 기준 색상 (스펙 있으면 이탈=빨강, 정상=파랑; 없으면 파랑)
                            if _sp_lo is not None and _sp_hi is not None:
                                _bc = "#f87171" if (_lo < _sp_lo or _hi2 > _sp_hi) else "#38bdf8"
                            else:
                                _bc = "#38bdf8"
                            _bar_h += (
                                f"<div style='display:flex;align-items:center;margin-bottom:4px;gap:8px;'>"
                                f"<span style='color:#cbd5e1;font-size:11px;width:100px;text-align:right;'>{_lo:.3f}~{_hi2:.3f}</span>"
                                f"<div style='flex:1;background:#1e293b;border-radius:3px;height:18px;'>"
                                f"<div style='width:{_bp:.1f}%;background:{_bc};height:18px;border-radius:3px;"
                                f"display:flex;align-items:center;padding-left:6px;'>"
                                f"<span style='color:#fff;font-size:11px;'>{int(_c)}</span></div></div></div>"
                            )
                        if is_en3:
                            _sp_note = f"● Blue: within spec ({_sp_lo}~{_sp_hi}) &nbsp; ● Red: out of spec" if _sp_lo is not None else "● Blue: measured value distribution"
                            _dist_word = "Distribution"
                        else:
                            _sp_note = f"● 파란색: 스펙 내 ({_sp_lo}~{_sp_hi}) &nbsp; ● 빨간색: 스펙 이탈" if _sp_lo is not None else "● 파란색: 측정값 분포"
                            _dist_word = "분포"
                        st.markdown(
                            f"<div style='background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;padding:16px;'>"
                            f"<div style='font-size:13px;color:#e2e8f0;font-weight:600;margin-bottom:12px;'>"
                            f"{TARGET_GLOSSARY.get(_sel_h, _sel_h)} {_dist_word} (n={_hn})</div>"
                            f"{_bar_h}"
                            f"<div style='color:#94a3b8;font-size:11px;margin-top:8px;'>{_sp_note}</div>"
                            f"</div>", unsafe_allow_html=True
                        )

        # ── ③ 변수 상관관계 히트맵 ────────────────────────────────
        _corr_lbl = "+ Process Variable Correlation Heatmap" if is_en3 else "+ 변수 상관관계 히트맵"
        with st.expander(_corr_lbl, expanded=False):
            st.markdown(
                "<details style='margin-bottom:12px;'>"
                f"<summary style='font-size:0.72rem;color:#64748b;cursor:pointer;padding:4px 0;list-style:none;'>{'▶ Learn what this means and how it is analyzed' if is_en3 else '▶ 이 항목의 의미와 분석 방법 보기'}</summary>"
                "<div style='background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;padding:12px 16px;margin-top:8px;font-size:0.72rem;color:#cbd5e1;line-height:1.8;'>"
                "<b style='color:#38bdf8;'>변수 상관관계 히트맵</b>이란?<br>"
                "34개 공정 변수(입력)와 품질 타겟(출력) 사이의 선형 상관계수(r)를 색상으로 표현한 표입니다. "
                "어떤 공정 변수가 품질에 가장 큰 영향을 미치는지 한눈에 파악할 수 있습니다.<br><br>"
                "<b style='color:#a3e635;'>주요 확인 포인트</b><br>"
                "· <b style='color:#38bdf8;'>파란색(양의 상관, r > 0.3)</b> — 이 변수가 증가하면 품질값도 증가하는 경향.<br>"
                "· <b style='color:#f87171;'>빨간색(음의 상관, r &lt; -0.3)</b> — 이 변수가 증가하면 품질값이 감소하는 경향.<br>"
                "· <b>|r| ≥ 0.5</b> — 강한 상관관계. 역최적화 시 이 변수들을 우선 조정 대상으로 삼으세요.<br>"
                "· <b>|r| &lt; 0.3</b> — 선형 상관 약함. 비선형 관계이거나 해당 변수의 영향이 적을 수 있습니다."
                "</div></details>",
                unsafe_allow_html=True
            )
            if _df3.empty:
                st.info("데이터 없음" if not is_en3 else "No data available.")
            else:
                _num_cols = _df3.select_dtypes(include=[np.number]).columns.tolist()
                _proc_c   = [c for c in _num_cols if c not in target_vars]
                _tgt_c    = [c for c in _num_cols if c in target_vars]
                if not _proc_c or not _tgt_c:
                    st.info("상관관계 계산에 필요한 공정 변수 또는 품질 타겟 컬럼이 충분하지 않습니다." if not is_en3 else "Insufficient columns for correlation.")
                else:
                    _corr_all = _df3[_proc_c + _tgt_c].corr()
                    _sc       = _corr_all.loc[_proc_c, _tgt_c]
                    _top10    = _sc.abs().max(axis=1).sort_values(ascending=False).head(10).index.tolist()
                    _sc       = _sc.loc[_top10]
                    _flat     = _sc.abs().stack()
                    if not _flat.empty:
                        _tp  = _flat.idxmax()
                        _tv  = float(_sc.loc[_tp[0], _tp[1]])
                        if is_en3:
                            _dir = "positive" if _tv > 0 else "negative"
                        else:
                            _dir = "양의 방향으로" if _tv > 0 else "음의 방향으로"
                        # 강한 상관 쌍 수
                        _strong = [(r,c,float(_sc.loc[r,c])) for r in _sc.index for c in _sc.columns if abs(float(_sc.loc[r,c])) >= 0.5]
                        if is_en3:
                            _advice_c = (f"{len(_strong)} strong correlated variable-quality pairs with |r| ≥ 0.5 — key targets for process control." if _strong
                                         else "No strong linear correlation (|r| ≥ 0.5) found. Non-linear interactions may dominate.")
                            _strongest_lbl = "Strongest correlation:"
                        else:
                            _advice_c = (f"|r| ≥ 0.5 강한 상관 변수-품질 쌍 {len(_strong)}개 — 공정 관리의 핵심 타겟입니다." if _strong
                                         else "강한 선형 상관(|r| ≥ 0.5)이 없습니다. 비선형 상호작용이 지배적일 수 있습니다.")
                            _strongest_lbl = "가장 강한 상관관계:"
                        _dir_txt = f"({_dir}, r={_tv:.2f})" if is_en3 else f"({_dir}으로, r={_tv:.2f})"
                        st.markdown(
                            f"<div style='background:#0a1628;border:1px solid #1e3a5f;border-left:3px solid #38bdf8;"
                            f"border-radius:6px;padding:10px 14px;margin-bottom:10px;font-size:0.82rem;'>"
                            f"<span style='color:#e2e8f0;'>{_strongest_lbl} <b style='color:#38bdf8;'>{_tp[0]}</b> → "
                            f"<b style='color:#38bdf8;'>{_tp[1]}</b> {_dir_txt}</span><br>"
                            f"<span style='color:#cbd5e1;font-size:0.78rem;'>→ {_advice_c}</span></div>",
                            unsafe_allow_html=True
                        )
                    _hdr = "".join([f"<th style='padding:5px 7px;font-size:11px;color:#cbd5e1;text-align:center;white-space:nowrap;'>{c}</th>" for c in _sc.columns])
                    _rows_c = ""
                    for _vr in _sc.index:
                        _cells = f"<td style='padding:5px 7px;font-size:11px;color:#e2e8f0;font-weight:600;white-space:nowrap;'>{_vr}</td>"
                        for _vc in _sc.columns:
                            _vc_val = float(_sc.loc[_vr, _vc])
                            _ic = abs(_vc_val)
                            if _vc_val > 0.3:   _bg, _tc = f"rgba(56,189,248,{min(_ic,0.9):.2f})", "#000"
                            elif _vc_val < -0.3: _bg, _tc = f"rgba(239,68,68,{min(_ic,0.9):.2f})", "#fff"
                            else:                _bg, _tc = "#1e293b", "#6b7fa3"
                            _cells += f"<td style='padding:5px 7px;text-align:center;background:{_bg};color:{_tc};font-size:11px;'>{_vc_val:.2f}</td>"
                        _rows_c += f"<tr>{_cells}</tr>"
                    _corr_title = ("Process Variable ↔ Quality Item Correlation Coefficients (Top 10 Process Variables)" if is_en3
                                    else "공정 변수 ↔ 불량 항목 상관관계수 (상위 10개 공정 변수)")
                    st.markdown(
                        f"<div style='background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;padding:16px;overflow-x:auto;'>"
                        f"<div style='color:#e2e8f0;font-size:0.9rem;font-weight:600;margin-bottom:12px;'>{_corr_title}</div>"
                        f"<table style='border-collapse:collapse;width:100%;'>"
                        f"<thead><tr><th style='padding:5px 7px;'></th>{_hdr}</tr></thead>"
                        f"<tbody>{_rows_c}</tbody></table>"
                        f"<div style='color:#6b7fa3;font-size:0.72rem;margin-top:8px;'>{'● Blue: positive correlation &nbsp; ● Red: negative correlation &nbsp; ● Gray: weak correlation' if is_en3 else '● 파란색: 양의 상관 &nbsp; ● 빨간색: 음의 상관 &nbsp; ● 회색: 상관 약함'}</div>"
                        f"</div>", unsafe_allow_html=True
                    )

        # ── ④ 변수 민감도 분석 (모델 기반) ──────────────────────────
        _sens_lbl = "+ Variable Sensitivity Analysis (Model-based)" if is_en3 else "+ 변수 민감도 분석 (모델 기반)"
        with st.expander(_sens_lbl, expanded=False):
            st.markdown(
                "<details style='margin-bottom:12px;'>"
                f"<summary style='font-size:0.72rem;color:#64748b;cursor:pointer;padding:4px 0;list-style:none;'>{'▶ Learn what this means and how it is analyzed' if is_en3 else '▶ 이 항목의 의미와 분석 방법 보기'}</summary>"
                "<div style='background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;padding:12px 16px;margin-top:8px;font-size:0.72rem;color:#cbd5e1;line-height:1.8;'>"
                "<b style='color:#38bdf8;'>변수 민감도 분석</b>이란?<br>"
                "상관관계 히트맵은 데이터 전체의 <b>선형</b> 관계만 보여주지만, 이 분석은 실제 <b>학습된 예측 모델</b>에 "
                "각 공정 변수를 기준점(데이터 중앙값)에서 살짝(±5%) 흔들어보고, 예측값이 어느 쪽으로 얼마나 움직이는지 "
                "직접 확인합니다. XGBoost 같은 비선형 모델의 실제 반응을 그대로 반영합니다.<br><br>"
                "<b style='color:#a3e635;'>주요 확인 포인트</b><br>"
                "· <b style='color:#38bdf8;'>↑ (양의 방향)</b> — 이 변수를 늘리면 예측값도 증가.<br>"
                "· <b style='color:#f87171;'>↓ (음의 방향)</b> — 이 변수를 늘리면 예측값이 감소.<br>"
                "· <b>민감도 크기</b> — 값이 클수록 예측 결과에 미치는 영향이 큼. 우선 조정 대상 변수를 고를 때 참고하세요.<br>"
                "· 상관관계 히트맵과 방향이 다르게 나온다면, 그 변수는 <b>비선형 관계</b>를 가질 가능성이 높습니다."
                "</div></details>",
                unsafe_allow_html=True
            )
            if _df3.empty or st.session_state.get('scaler') is None:
                st.info("데이터 및 모델 학습이 필요합니다." if not is_en3 else "Data and trained models are required.")
            else:
                _sens_tgt_opts = [t for t in target_vars if st.session_state.get(f'model_{t.lower()}') is not None]
                if not _sens_tgt_opts:
                    st.info("학습된 예측 모델이 없습니다." if not is_en3 else "No trained models available.")
                else:
                    _sens_sel_lbl = "품질 타겟 선택" if not is_en3 else "Select Quality Target"
                    _sens_sel = st.selectbox(_sens_sel_lbl, _sens_tgt_opts,
                                              format_func=lambda k: TARGET_GLOSSARY.get(k, k),
                                              key="t3_sens_sel")
                    _sens_model = st.session_state[f'model_{_sens_sel.lower()}']
                    _sens_scaler = st.session_state['scaler']

                    # 기준점: 각 공정 변수의 데이터 중앙값
                    _baseline_x = []
                    _span_x = {}
                    for v in X_list:
                        if v in _df3.columns:
                            _col_v = pd.to_numeric(_df3[v], errors='coerce').dropna()
                        else:
                            _col_v = pd.Series(dtype=float)
                        if not _col_v.empty:
                            _baseline_x.append(float(_col_v.median()))
                            _span_x[v] = max(float(_col_v.max() - _col_v.min()), 1e-9)
                        else:
                            _lo_v, _hi_v = db.get(v, (0.0, 1.0))
                            _baseline_x.append(float((_lo_v + _hi_v) / 2))
                            _span_x[v] = max(_hi_v - _lo_v, 1e-9)

                    _sens_rows = []
                    for _i, v in enumerate(X_list):
                        _delta = _span_x[v] * 0.05
                        _x_up = list(_baseline_x);   _x_up[_i]   += _delta
                        _x_down = list(_baseline_x); _x_down[_i] -= _delta
                        _q_up   = _sens_scaler.transform(pd.DataFrame([_x_up], columns=X_list))
                        _q_down = _sens_scaler.transform(pd.DataFrame([_x_down], columns=X_list))
                        _p_up   = float(_sens_model.predict(_q_up)[0])
                        _p_down = float(_sens_model.predict(_q_down)[0])
                        _sensitivity = (_p_up - _p_down) / 2.0
                        _sens_rows.append((v, _sensitivity))

                    _sens_rows.sort(key=lambda r: abs(r[1]), reverse=True)
                    _max_abs_sens = max(abs(r[1]) for r in _sens_rows) if _sens_rows else 1e-9
                    _max_abs_sens = max(_max_abs_sens, 1e-9)

                    _top_var, _top_val = _sens_rows[0]
                    if is_en3:
                        _top_dir = "positive (increases with higher value)" if _top_val > 0 else "negative (decreases with higher value)"
                        _top_msg = f"→ Adjusting {_top_var} causes the largest response in {_sens_sel}. Consider it a priority adjustment target."
                        _top_lbl = "Most sensitive variable:"
                    else:
                        _top_dir = "양의 방향(늘리면 증가)" if _top_val > 0 else "음의 방향(늘리면 감소)"
                        _top_msg = f"→ {_top_var} 조정 시 {_sens_sel} 값이 가장 크게 반응합니다. 우선 조정 대상으로 고려하세요."
                        _top_lbl = "가장 민감한 변수:"
                    st.markdown(
                        f"<div style='background:#0a1628;border:1px solid #1e3a5f;border-left:3px solid #38bdf8;"
                        f"border-radius:6px;padding:10px 14px;margin-bottom:10px;font-size:0.82rem;'>"
                        f"<span style='color:#e2e8f0;'>{_top_lbl} <b style='color:#38bdf8;'>{_top_var}</b> → "
                        f"<b style='color:#38bdf8;'>{_sens_sel}</b> ({_top_dir}, Δ={_top_val:+.4f})</span><br>"
                        f"<span style='color:#cbd5e1;font-size:0.78rem;'>{_top_msg}</span></div>",
                        unsafe_allow_html=True
                    )

                    _rows_html = ""
                    for v, sens in _sens_rows[:15]:
                        _bar_pct = min(int(abs(sens) / _max_abs_sens * 100), 100)
                        _is_pos = sens > 0
                        _dir_arrow = "▲" if _is_pos else "▼" if sens < 0 else "—"
                        _dir_color = "#38bdf8" if _is_pos else "#f87171" if sens < 0 else "#64748b"
                        if is_en3:
                            _dir_txt = "Increases" if _is_pos else "Decreases" if sens < 0 else "No effect"
                        else:
                            _dir_txt = "늘리면 증가" if _is_pos else "늘리면 감소" if sens < 0 else "영향 없음"
                        _rows_html += (
                            f"<tr>"
                            f"<td style='padding:5px 8px;font-size:0.8rem;color:#e2e8f0;font-weight:700;white-space:nowrap;'>{v}</td>"
                            f"<td style='padding:5px 8px;font-size:0.78rem;color:{_dir_color};font-weight:700;white-space:nowrap;'>{_dir_arrow} {_dir_txt}</td>"
                            f"<td style='padding:5px 8px;min-width:140px;'>"
                            f"<span style='color:{_dir_color};font-weight:700;font-family:monospace;font-size:0.8rem;'>{sens:+.4f}</span>"
                            f"<div style='background:#1e293b;border-radius:2px;height:5px;margin-top:2px;'>"
                            f"<div style='width:{_bar_pct}%;background:{_dir_color};height:5px;border-radius:2px;'></div></div>"
                            f"</td>"
                            f"</tr>"
                        )
                    if is_en3:
                        _sens_table_title = f"{_sens_sel} — Top 15 Sensitive Variables (prediction response to ±5% change from data median)"
                        _th_s1, _th_s2, _th_s3 = "Process Variable", "Direction", "Sensitivity (Δ Predicted)"
                        _sens_footer = ("▲ Blue: variable increase → prediction increases &nbsp; ▼ Red: variable increase → prediction decreases &nbsp;|&nbsp; "
                                        "This reflects the local ±5% response around the baseline (median); direction may change in other ranges.")
                    else:
                        _sens_table_title = f"{_sens_sel} 민감도 상위 15개 변수 (데이터 중앙값 기준 ±5% 변화 시 예측 반응)"
                        _th_s1, _th_s2, _th_s3 = "공정 변수", "방향성", "민감도 (Δ예측값)"
                        _sens_footer = ("▲ 파란색: 변수 증가 → 예측값 증가 &nbsp; ▼ 빨간색: 변수 증가 → 예측값 감소 &nbsp;|&nbsp; "
                                        "기준점(중앙값)에서 ±5% 국소 변화에 대한 반응이라 구간이 달라지면 방향이 바뀔 수 있습니다.")
                    st.markdown(
                        f"<div style='background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;padding:16px;overflow-x:auto;'>"
                        f"<div style='color:#e2e8f0;font-size:0.9rem;font-weight:600;margin-bottom:12px;'>"
                        f"{_sens_table_title}</div>"
                        f"<table style='width:100%;border-collapse:collapse;'>"
                        f"<thead><tr style='border-bottom:1px solid #1e3a5f;'>"
                        f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:left;'>{_th_s1}</th>"
                        f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:left;'>{_th_s2}</th>"
                        f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:left;'>{_th_s3}</th>"
                        f"</tr></thead>"
                        f"<tbody>{_rows_html}</tbody>"
                        f"</table>"
                        f"<div style='color:#6b7fa3;font-size:0.72rem;margin-top:8px;'>{_sens_footer}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

        # ── ⑤ 시계열 트렌드 차트 ──────────────────────────────────
        _trend_lbl = "+ Time-Series Trend Chart" if is_en3 else "+ 시계열 트렌드 차트"
        with st.expander(_trend_lbl, expanded=False):
            st.markdown(
                "<details style='margin-bottom:12px;'>"
                f"<summary style='font-size:0.72rem;color:#64748b;cursor:pointer;padding:4px 0;list-style:none;'>{'▶ Learn what this means and how it is analyzed' if is_en3 else '▶ 이 항목의 의미와 분석 방법 보기'}</summary>"
                "<div style='background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;padding:12px 16px;margin-top:8px;font-size:0.72rem;color:#cbd5e1;line-height:1.8;'>"
                + (
                "<b style='color:#38bdf8;'>Time-Series Trend Chart</b>?<br>"
                "Shows the change in quality and process values over the row order of the data (measurement or "
                "chronological order) as a line graph. Useful for checking whether the process is stable or "
                "whether quality has been degrading from a certain point.<br><br>"
                "<b style='color:#a3e635;'>Key things to check</b><br>"
                "· <b style='color:#f87171;'>↑ Upward trend</b> — the second-half average is higher than the first half. Quality value is trending upward.<br>"
                "· <b style='color:#10b981;'>↓ Downward trend</b> — the second-half average is lower than the first half. Quality value is trending downward.<br>"
                "· <b>If the trend heads toward a spec limit</b> — possible process drift. Check the root-cause variable immediately.<br>"
                "· <b>Chart values are normalized 0~1</b> — to compare multiple items at once, each item is scaled to a relative value with its own min=0 and max=1."
                if is_en3 else
                "<b style='color:#38bdf8;'>시계열 트렌드 차트</b>이란?<br>"
                "데이터 행 순서(측정 순서 또는 시간 순서)에 따라 품질값과 공정값의 변화 추이를 선 그래프로 보여줍니다. "
                "공정이 안정적인지, 특정 시점부터 품질이 나빠지고 있는지를 파악하는 데 유용합니다.<br><br>"
                "<b style='color:#a3e635;'>주요 확인 포인트</b><br>"
                "· <b style='color:#f87171;'>↑ 상승 추세</b> — 후반부 평균이 전반부보다 높음. 품질값이 증가 방향으로 변화 중.<br>"
                "· <b style='color:#10b981;'>↓ 하락 추세</b> — 후반부 평균이 전반부보다 낮음. 품질값이 감소 방향으로 변화 중.<br>"
                "· <b>추세가 스펙 한계 방향이면</b> — 공정 드리프트 가능성. 즉시 원인 변수 점검 필요.<br>"
                "· <b>차트 값은 0~1 정규화</b> — 여러 항목을 동시에 비교하기 위해 각 항목의 최솟값=0, 최댓값=1로 변환한 상대값입니다."
                )
                + "</div></details>",
                unsafe_allow_html=True
            )
            if _df3.empty:
                st.info("데이터 없음" if not is_en3 else "No data available.")
            else:
                _tgt_t  = [c for c in _df3.columns if c in target_vars]
                _proc_t = [c for c in _df3.select_dtypes(include=[np.number]).columns if c not in target_vars]
                # 추세 분석 (전반부 vs 후반부)
                _tr_ins = []
                for _tc3 in _tgt_t:
                    _td = pd.to_numeric(_df3[_tc3], errors='coerce').dropna()
                    if len(_td) >= 4:
                        _fh = float(_td.iloc[:len(_td)//2].mean())
                        _lh = float(_td.iloc[len(_td)//2:].mean())
                        _dl = _lh - _fh
                        if abs(_dl) >= 0.001:
                            if is_en3:
                                _dir2 = "↑ Upward" if _dl > 0 else "↓ Downward"
                            else:
                                _dir2 = "↑ 상승 추세" if _dl > 0 else "↓ 하락 추세"
                            _col2 = "#f87171" if _dl > 0 else "#10b981"
                            _tr_ins.append(f"<b style='color:{_col2};'>{_tc3}</b>: {_dir2} (Δ{_dl:+.3f})")
                if not _tr_ins and not _tgt_t:
                    st.info("품질 타겟 컬럼이 없어 추세 분석을 할 수 없습니다." if not is_en3 else "No target columns for trend analysis.")
                else:
                    if _tr_ins:
                        _tr_title = ("▸ Quality Risk Trend (first-half vs. second-half average comparison)" if is_en3
                                      else "▸ 품질 리스크 추세 (데이터 전반부 vs. 후반부 평균 비교)")
                        st.markdown(
                            f"<div style='background:#0a1628;border:1px solid #1e3a5f;border-left:3px solid #38bdf8;"
                            f"border-radius:6px;padding:10px 14px;margin-bottom:10px;'>"
                            f"<div style='font-size:0.78rem;color:#38bdf8;font-weight:600;margin-bottom:6px;'>"
                            f"{_tr_title}</div>"
                            + "".join([f"<div style='font-size:0.80rem;color:#e2e8f0;padding:2px 0;'>{t}</div>" for t in _tr_ins])
                            + "</div>", unsafe_allow_html=True
                        )
                    else:
                        _no_trend_txt = ("✅ No significant trend change (first-half vs. second-half difference is minimal)" if is_en3
                                          else "✅ 유의미한 추세 변화 없음 (전반부 vs 후반부 차이 미미)")
                        st.markdown(
                            "<div style='background:#0a1628;border:1px solid #1e3a5f;border-left:3px solid #10b981;"
                            f"border-radius:6px;padding:10px 14px;margin-bottom:10px;font-size:0.82rem;color:#10b981;'>"
                            f"{_no_trend_txt}</div>",
                            unsafe_allow_html=True
                        )
                    _all_t = _tgt_t + _proc_t
                    _tr_sel = st.multiselect(
                        "트렌드 확인할 항목 선택 (복수 선택 가능)" if not is_en3 else "Select items for trend (multi-select)",
                        _all_t,
                        default=_tgt_t[:3] if len(_tgt_t) >= 3 else _tgt_t,
                        format_func=lambda k: TARGET_GLOSSARY.get(k, k),
                        key="t3_trend_sel"
                    )
                    if _tr_sel:
                        _tdf = _df3[_tr_sel].reset_index(drop=True)
                        _COLS = ["#38bdf8","#a3e635","#ffab00","#f87171","#c084fc","#fb923c","#34d399","#f472b6","#60a5fa","#fbbf24"]
                        _ldefs = []
                        for _ci3, _col3 in enumerate(_tr_sel):
                            _cd = pd.to_numeric(_tdf[_col3], errors='coerce').dropna()
                            if _cd.empty: continue
                            _mn3, _mx3 = float(_cd.min()), float(_cd.max())
                            _sp3b = _mx3 - _mn3 if _mx3 != _mn3 else 1.0
                            _ldefs.append({'col': _col3, 'data': (_cd-_mn3)/_sp3b, 'raw': _cd, 'color': _COLS[_ci3 % len(_COLS)]})
                        if _ldefs:
                            W, H, PAD = 900, 240, 40
                            _svg = ""
                            for _ld in _ldefs:
                                _pts = []
                                for _xi, (_, _yv) in enumerate(_ld['data'].items()):
                                    _x = PAD + (_xi / max(len(_ld['data'])-1, 1)) * (W-2*PAD)
                                    _y = PAD + (1-float(_yv)) * (H-2*PAD)
                                    _pts.append(f"{_x:.1f},{_y:.1f}")
                                if _pts:
                                    _svg += f"<polyline points='{' '.join(_pts)}' fill='none' stroke='{_ld['color']}' stroke-width='2' opacity='0.85'/>"
                                    _lx2, _ly2 = float(_pts[-1].split(',')[0]), float(_pts[-1].split(',')[1])
                                    _svg += f"<circle cx='{_lx2}' cy='{_ly2}' r='4' fill='{_ld['color']}'/>"
                                    _svg += f"<text x='{min(_lx2+6,W-60)}' y='{_ly2+4}' fill='{_ld['color']}' font-size='10'>{float(_ld['raw'].iloc[-1]):.3f}</text>"
                            _leg = ""
                            for _li, _ld in enumerate(_ldefs):
                                _lx3 = PAD + _li * 130
                                _leg += f"<rect x='{_lx3}' y='{H+8}' width='14' height='8' fill='{_ld['color']}' rx='2'/>"
                                _leg += f"<text x='{_lx3+18}' y='{H+16}' fill='#cbd5e1' font-size='10'>{TARGET_GLOSSARY.get(_ld['col'],_ld['col'])[:12]}</text>"
                            st.markdown(
                                f"<div style='background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;padding:16px;overflow-x:auto;'>"
                                f"<div style='color:#e2e8f0;font-size:0.88rem;font-weight:600;margin-bottom:8px;'>{'Trend Chart (Normalized 0~1)' if is_en3 else '트렌드 차트 (정규화 0~1 표시)'}</div>"
                                f"<svg viewBox='0 0 {W} {H+36}' style='width:100%;max-height:280px;'>"
                                f"<line x1='{PAD}' y1='{PAD}' x2='{PAD}' y2='{H-PAD}' stroke='#1e3a5f' stroke-width='1'/>"
                                f"<line x1='{PAD}' y1='{H-PAD}' x2='{W-PAD}' y2='{H-PAD}' stroke='#1e3a5f' stroke-width='1'/>"
                                f"{_svg}{_leg}</svg></div>",
                                unsafe_allow_html=True
                            )

    with tab4:
        _run4 = False
        _is4 = st.session_state.get('lang', 'KO') == 'EN'

        # AI 추천값 vs 현재(TEST) 값의 차이 순위를 미리 계산 (조정 처방 표 정렬에도 동일하게 재사용)
        _t4_ai_ref = None
        if st.session_state.get('opt_result_x') is not None:
            _t4_ai_ref = list(st.session_state['opt_result_x'])
        elif st.session_state.get('sim_dx_result_x') is not None:
            _t4_ai_ref = list(st.session_state['sim_dx_result_x'])

        _df_t4_rank = st.session_state.get('df_imputed_ref')
        _t4_test_ref = []
        for v in X_list:
            if _df_t4_rank is not None and v in _df_t4_rank.columns:
                _cv_r = pd.to_numeric(_df_t4_rank[v], errors='coerce')
                _t4_test_ref.append(float(_cv_r.mean()) if not _cv_r.dropna().empty else float((db.get(v,(0,1))[0]+db.get(v,(0,1))[1])/2))
            else:
                _t4_test_ref.append(float((db.get(v,(0,1))[0]+db.get(v,(0,1))[1])/2))

        if _t4_ai_ref is not None:
            _t4_diff_rank = {v: abs(_t4_ai_ref[i] - _t4_test_ref[i]) for i, v in enumerate(X_list)}
        else:
            _t4_diff_rank = {v: 0.0 for v in X_list}

        t4_l, t4_r = st.columns([1.3, 1.2], gap="large")

        with t4_l:
            _t4_card1_title = "Reset Target Spec" if _is4 else "목표 스펙 재설정"
            st.markdown(f"<div class='glass-card'><div class='glass-card-title'>{_t4_card1_title}</div>", unsafe_allow_html=True)
            if _is4:
                st.markdown(
                    "<div style='font-size:0.78rem;color:#64748b;margin-bottom:10px;'>"
                    "Change the quality spec range and instantly see <b style='color:#38bdf8;'>which design/process "
                    "variable to adjust, in which direction, and by how much</b> to bring it back within range. "
                    "(Instead of Tab1's heavy algorithm search, this gives a fast sensitivity-based first-order estimate)"
                    "</div>", unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div style='font-size:0.78rem;color:#64748b;margin-bottom:10px;'>"
                    "품질 스펙 범위를 바꿔보면, <b style='color:#38bdf8;'>어떤 설계/공정 변수를 어느 방향으로 얼마큼 "
                    "조정해야</b> 그 범위 안에 들어오는지 즉시 계산해드립니다. (Tab1의 무거운 알고리즘 탐색 대신, "
                    "민감도 기반 1차 근사로 빠르게 처방합니다)"
                    "</div>", unsafe_allow_html=True
                )

            _t4_tgt_opts = [t for t in valid_tgts if st.session_state.get(f'model_{t.lower()}') is not None]
            if not _t4_tgt_opts:
                st.info("No trained prediction models available." if _is4 else "학습된 예측 모델이 없습니다.")
            else:
                _t4_sel = st.selectbox("Select Quality Target" if _is4 else "품질 타겟 선택", _t4_tgt_opts,
                                        format_func=lambda k: TARGET_GLOSSARY.get(k, k),
                                        key="t4_target_sel")

                _sp_lo4, _sp_hi4 = spec_limits.get(_t4_sel, (0.0, 1.0))
                _span4 = max(_sp_hi4 - _sp_lo4, 0.01)
                _slider_min4 = round(_sp_lo4 - _span4 * 0.5, 4)
                _slider_max4 = round(_sp_hi4 + _span4 * 0.5, 4)
                _step4 = max(round(_span4 / 200, 4), 0.001)

                _t4_range_lbl = (f"{_t4_sel} New Target Range (Base Spec: {_sp_lo4:.2f} ~ {_sp_hi4:.2f})" if _is4
                                  else f"{_t4_sel} 새 목표 Range (기본 스펙: {_sp_lo4:.2f} ~ {_sp_hi4:.2f})")
                st.markdown(f"<p style='font-size:0.85rem; font-weight:600; color:#38bdf8; margin-bottom:5px;'>{_t4_range_lbl}</p>", unsafe_allow_html=True)
                _t4_prefix = _t4_sel.lower()
                _t4_key = f"t4_range_{_t4_prefix}_s_val"
                st.session_state.setdefault(_t4_key, (float(_sp_lo4), float(_sp_hi4)))
                st.session_state.setdefault(f"t4_range_{_t4_prefix}_n_min", float(st.session_state[_t4_key][0]))
                st.session_state.setdefault(f"t4_range_{_t4_prefix}_n_max", float(st.session_state[_t4_key][1]))
                _c4a, _c4b, _c4c = st.columns([1.8, 0.6, 0.6])
                with _c4a:
                    st.slider(f"{_t4_sel} Range4", float(_slider_min4), float(_slider_max4), step=_step4,
                              format="%.3f", label_visibility="collapsed", key=_t4_key,
                              on_change=on_t4_slider_change, args=(_t4_prefix,))
                with _c4b:
                    st.number_input("Min", step=_step4, format="%.3f", key=f"t4_range_{_t4_prefix}_n_min",
                                     on_change=on_t4_min_change, args=(_t4_prefix,))
                with _c4c:
                    st.number_input("Max", step=_step4, format="%.3f", key=f"t4_range_{_t4_prefix}_n_max",
                                     on_change=on_t4_max_change, args=(_t4_prefix,))

                st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='glass-card-title' style='font-size:0.85rem;'>{'Baseline Setting' if _is4 else '기준점 설정'}</div>", unsafe_allow_html=True)
                if _is4:
                    _t4_base_opts = ["Data Median"]
                    if st.session_state.get('opt_result_x') is not None:
                        _t4_base_opts.insert(0, "Latest Backward Optimization Result (Tab1)")
                    if st.session_state.get('sim_dx_result_x') is not None:
                        _t4_base_opts.insert(0, "Latest Forward Optimization Result (Tab2)")
                else:
                    _t4_base_opts = ["데이터 중앙값 기준"]
                    if st.session_state.get('opt_result_x') is not None:
                        _t4_base_opts.insert(0, "최근 역방향 최적화 결과 (Tab1)")
                    if st.session_state.get('sim_dx_result_x') is not None:
                        _t4_base_opts.insert(0, "최근 순방향 최적화 결과 (Tab2)")
                _t4_base_mode = st.radio("Baseline" if _is4 else "기준점", _t4_base_opts, index=0, horizontal=True,
                                          key="t4_base_mode", label_visibility="collapsed")

                _run4 = st.button("⚡ Compute Prescription (Instant)" if _is4 else "⚡ 처방 계산 (즉시)", type="primary", key="t4_run_btn")
            st.markdown("</div>", unsafe_allow_html=True)

        with t4_r:
            _t4_card2_title = "Real-time Deviation Diagnosis & Adjustment Prescription" if _is4 else "실시간 이탈 진단 & 조정 처방"
            st.markdown(f"<div class='glass-card'><div class='glass-card-title' style='color:#3b82f6;'>{_t4_card2_title}</div>", unsafe_allow_html=True)

            if not _t4_tgt_opts:
                st.info("Please load data on the left first." if _is4 else "좌측에서 데이터를 먼저 로드하세요.")
            elif _run4:
                _df4 = st.session_state.get('df_imputed_ref')
                _model4 = st.session_state[f'model_{_t4_sel.lower()}']
                _scaler4 = st.session_state['scaler']
                _new_lo, _new_hi = st.session_state[_t4_key]

                # 기준점(baseline_x) 구성
                if _t4_base_mode in ("최근 순방향 최적화 결과 (Tab2)", "Latest Forward Optimization Result (Tab2)") and st.session_state.get('sim_dx_result_x') is not None:
                    _baseline_x4 = list(st.session_state['sim_dx_result_x'])
                elif _t4_base_mode in ("최근 역방향 최적화 결과 (Tab1)", "Latest Backward Optimization Result (Tab1)") and st.session_state.get('opt_result_x') is not None:
                    _baseline_x4 = list(st.session_state['opt_result_x'])
                elif _df4 is not None and not _df4.empty:
                    _baseline_x4 = [float(pd.to_numeric(_df4[v], errors='coerce').median()) if v in _df4.columns else float((db.get(v,(0,1))[0]+db.get(v,(0,1))[1])/2) for v in X_list]
                else:
                    _baseline_x4 = [float((db.get(v,(0,1))[0]+db.get(v,(0,1))[1])/2) for v in X_list]

                _span_x4 = {}
                for v in X_list:
                    if _df4 is not None and v in _df4.columns:
                        _cv = pd.to_numeric(_df4[v], errors='coerce').dropna()
                        _span_x4[v] = max(float(_cv.max() - _cv.min()), 1e-9) if not _cv.empty else max(db.get(v,(0,1))[1]-db.get(v,(0,1))[0], 1e-9)
                    else:
                        _span_x4[v] = max(db.get(v,(0,1))[1]-db.get(v,(0,1))[0], 1e-9)

                _q_base4 = _scaler4.transform(pd.DataFrame([_baseline_x4], columns=X_list))
                _pred_base4 = float(_model4.predict(_q_base4)[0])

                _in_spec4 = _new_lo <= _pred_base4 <= _new_hi
                _gap4 = 0.0 if _in_spec4 else (_new_lo - _pred_base4 if _pred_base4 < _new_lo else _new_hi - _pred_base4)

                if _is4:
                    _status_badge4 = (
                        f"<span style='background:#0a2010;color:#10b981;font-size:0.75rem;padding:2px 8px;border-radius:4px;'>✅ In Spec</span>"
                        if _in_spec4 else
                        f"<span style='background:#2d0f0f;color:#f87171;font-size:0.75rem;padding:2px 8px;border-radius:4px;'>⚠️ Out of Spec</span>"
                    )
                    _baseline_pred_lbl = f"Baseline Predicted Value ({_t4_sel})"
                    _new_range_lbl = "New Target Range"
                    _gap_lbl = "Gap"
                else:
                    _status_badge4 = (
                        f"<span style='background:#0a2010;color:#10b981;font-size:0.75rem;padding:2px 8px;border-radius:4px;'>✅ 스펙 내</span>"
                        if _in_spec4 else
                        f"<span style='background:#2d0f0f;color:#f87171;font-size:0.75rem;padding:2px 8px;border-radius:4px;'>⚠️ 이탈</span>"
                    )
                    _baseline_pred_lbl = f"기준점 예측값 ({_t4_sel})"
                    _new_range_lbl = "새 목표 Range"
                    _gap_lbl = "부족분(gap)"
                st.markdown(
                    f"<div style='background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;padding:12px 14px;margin-bottom:12px;'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                    f"<span style='color:#94a3b8;font-size:0.8rem;'>{_baseline_pred_lbl}</span>{_status_badge4}</div>"
                    f"<div style='font-size:1.3rem;font-weight:700;color:#ffffff;margin-top:4px;'>{_pred_base4:.3f}</div>"
                    f"<div style='font-size:0.78rem;color:#64748b;margin-top:2px;'>{_new_range_lbl}: {_new_lo:.3f} ~ {_new_hi:.3f}"
                    + (f" &nbsp;|&nbsp; {_gap_lbl}: <b style='color:#f59e0b;'>{_gap4:+.4f}</b>" if not _in_spec4 else "")
                    + f"</div></div>",
                    unsafe_allow_html=True
                )

                if _in_spec4:
                    if _is4:
                        st.success(f"{_t4_sel} is already within the new target range at the current baseline. No adjustment needed.")
                    else:
                        st.success(f"현재 기준점에서 이미 {_t4_sel}이(가) 새 목표 Range 안에 있습니다. 별도 조정이 필요 없습니다.")
                else:
                    _rows4 = []
                    for _i, v in enumerate(X_list):
                        _delta_probe = _span_x4[v] * 0.05
                        _x_up = list(_baseline_x4);   _x_up[_i]   += _delta_probe
                        _x_down = list(_baseline_x4); _x_down[_i] -= _delta_probe
                        _p_up = float(_model4.predict(_scaler4.transform(pd.DataFrame([_x_up], columns=X_list)))[0])
                        _p_down = float(_model4.predict(_scaler4.transform(pd.DataFrame([_x_down], columns=X_list)))[0])
                        _sens4 = (_p_up - _p_down) / (2.0 * _delta_probe)
                        if abs(_sens4) < 1e-9:
                            continue
                        _needed_delta = _gap4 / _sens4
                        _new_val4 = _baseline_x4[_i] + _needed_delta
                        _x_after = list(_baseline_x4); _x_after[_i] = _new_val4
                        _p_after = float(_model4.predict(_scaler4.transform(pd.DataFrame([_x_after], columns=X_list)))[0])
                        _lo_v4, _hi_v4 = db.get(v, (0.0, 1.0))
                        _feasible4 = _lo_v4 <= _new_val4 <= _hi_v4
                        _rows4.append((v, _needed_delta, _baseline_x4[_i], _new_val4, _p_after, _feasible4))

                    _rows4.sort(key=lambda r: _t4_diff_rank.get(r[0], 0.0), reverse=True)
                    _rows4 = _rows4[:10]

                    _presc_title = ("▣ Adjustment Prescription (sorted by largest difference, same order as "
                                     "'AI Recommended vs Current(TEST) Value Comparison' below)" if _is4 else
                                     "▣ 조정 처방 (아래 'AI 추천값 vs 현재(TEST) 값 비교'와 동일하게, 차이가 큰 순으로 정렬)")
                    st.markdown(
                        f"<p style='font-size:0.82rem;font-weight:700;color:#10b981;margin:6px 0 8px 0;'>{_presc_title}</p>",
                        unsafe_allow_html=True
                    )
                    _rows_html4 = ""
                    for v, delta, cur_v, new_v, p_after, feasible in _rows4:
                        if _is4:
                            _dir_arrow4 = "▲ Increase" if delta > 0 else "▼ Decrease"
                        else:
                            _dir_arrow4 = "▲ 증가" if delta > 0 else "▼ 감소"
                        _dir_color4 = "#38bdf8" if delta > 0 else "#f87171"
                        _after_in_spec = _new_lo <= p_after <= _new_hi
                        if _is4:
                            _after_badge = (
                                "<span style='background:#0a2010;color:#10b981;font-size:0.68rem;padding:1px 6px;border-radius:3px;'>✅ In Spec</span>"
                                if _after_in_spec else
                                "<span style='background:#2d0f0f;color:#f87171;font-size:0.68rem;padding:1px 6px;border-radius:3px;'>⚠️ Not Met</span>"
                            )
                            _feas_badge = (
                                "<span style='background:#0a2010;color:#10b981;font-size:0.68rem;padding:1px 6px;border-radius:3px;'>✅ Within Data Range</span>"
                                if feasible else
                                "<span style='background:#2d0f0f;color:#f59e0b;font-size:0.68rem;padding:1px 6px;border-radius:3px;'>⚠️ Exceeds Data Range</span>"
                            )
                        else:
                            _after_badge = (
                                "<span style='background:#0a2010;color:#10b981;font-size:0.68rem;padding:1px 6px;border-radius:3px;'>✅ 스펙 내</span>"
                                if _after_in_spec else
                                "<span style='background:#2d0f0f;color:#f87171;font-size:0.68rem;padding:1px 6px;border-radius:3px;'>⚠️ 미달</span>"
                            )
                            _feas_badge = (
                                "<span style='background:#0a2010;color:#10b981;font-size:0.68rem;padding:1px 6px;border-radius:3px;'>✅ 데이터 범위 내</span>"
                                if feasible else
                                "<span style='background:#2d0f0f;color:#f59e0b;font-size:0.68rem;padding:1px 6px;border-radius:3px;'>⚠️ 데이터 범위 초과</span>"
                            )
                        _rows_html4 += (
                            f"<tr>"
                            f"<td style='padding:6px 8px;color:#e2e8f0;font-weight:700;'>{v}</td>"
                            f"<td style='padding:6px 8px;color:{_dir_color4};font-weight:700;font-size:0.8rem;white-space:nowrap;'>{_dir_arrow4}</td>"
                            f"<td style='padding:6px 8px;color:#ffffff;font-family:monospace;font-size:0.8rem;'>{delta:+.4f}<br>"
                            f"<span style='color:#64748b;font-size:0.68rem;'>{cur_v:.3f} → {new_v:.3f}</span></td>"
                            f"<td style='padding:6px 8px;font-size:0.78rem;'>{p_after:.3f}<br>{_after_badge}</td>"
                            f"<td style='padding:6px 8px;'>{_feas_badge}</td>"
                            f"</tr>"
                        )
                    if _is4:
                        _th_var, _th_dir, _th_delta, _th_after, _th_feas = "Variable", "Direction", "Needed Movement", "Predicted After Adjustment", "Feasibility"
                        _footnote4 = ("※ This is a sensitivity-based first-order estimate — error can grow for large "
                                       "changes or values beyond the data range. Re-verify precise values using Tab1/Tab2 optimization search.")
                    else:
                        _th_var, _th_dir, _th_delta, _th_after, _th_feas = "변수", "방향", "필요 이동량", "조정 후 예측", "실현 가능성"
                        _footnote4 = ("※ 민감도 기반 1차 근사입니다 — 변화량이 크거나 데이터 범위를 초과하면 오차가 커질 수 있습니다. "
                                       "정밀한 확정값은 Tab1/Tab2 최적화 탐색으로 재확인하세요.")
                    st.markdown(
                        f"<div style='background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;padding:12px 14px;'>"
                        f"<table style='width:100%;border-collapse:collapse;'>"
                        f"<thead><tr style='border-bottom:1px solid #1e3a5f;'>"
                        f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:left;'>{_th_var}</th>"
                        f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:left;'>{_th_dir}</th>"
                        f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:left;'>{_th_delta}</th>"
                        f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:left;'>{_th_after}</th>"
                        f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:left;'>{_th_feas}</th>"
                        f"</tr></thead>"
                        f"<tbody>{_rows_html4}</tbody>"
                        f"</table>"
                        f"<div style='color:#6b7fa3;font-size:0.72rem;margin-top:8px;'>{_footnote4}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.info("Set a new target Range on the left and click '⚡ Compute Prescription'." if _is4
                        else "좌측에서 새 목표 Range를 설정하고 '⚡ 처방 계산' 버튼을 눌러주세요.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        _t4_cmp_title = "### □ AI Recommended vs Current (TEST) Value Comparison — Identify Priority Changes" if _is4 else "### □ AI 추천값 vs 현재(TEST) 값 비교 — 우선 변경 대상 파악"
        st.markdown(_t4_cmp_title)
        if _is4:
            st.markdown(
                "<div style='font-size:0.78rem;color:#64748b;margin-bottom:10px;'>"
                "Compares the AI-recommended design/process variable values side by side with the values "
                "currently in use (TEST). The <b style='color:#facc15;'>larger the difference (AI-TEST), the "
                "higher the priority</b> for a real-world change — check exactly which variables to change, and "
                "by how much, if you want to satisfy the spec by changing as few variables as possible."
                "</div>", unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='font-size:0.78rem;color:#64748b;margin-bottom:10px;'>"
                "AI가 도출한 추천 설계/공정 변수 값과, 현재 실제 사용 중인(TEST) 값을 나란히 비교합니다. "
                "차이(AI-TEST)가 <b style='color:#facc15;'>큰 변수일수록 실물 변경 시 우선순위</b>가 높습니다 — "
                "적은 변수만 바꿔서 스펙을 만족시키고 싶을 때, 어떤 변수를 얼마나 바꿀지 바로 확인하세요."
                "</div>", unsafe_allow_html=True
            )

        _ai_src_opts = []
        if st.session_state.get('opt_result_x') is not None:
            _ai_src_opts.append("Tab1 Backward Optimization Result" if _is4 else "Tab1 역방향 최적화 결과")
        if st.session_state.get('sim_dx_result_x') is not None:
            _ai_src_opts.append("Tab2 Forward Optimization Result" if _is4 else "Tab2 순방향 최적화 결과")

        if not _ai_src_opts:
            st.info("Please run an optimization in Tab1 or Tab2 first to generate an AI recommendation." if _is4
                    else "먼저 Tab1 또는 Tab2에서 최적화를 실행해 AI 추천값을 만들어주세요.")
        else:
            _cmp_c1, _cmp_c2, _cmp_c3 = st.columns([1, 1, 0.6])
            with _cmp_c1:
                _ai_src_sel = st.selectbox("AI Recommendation Source" if _is4 else "AI 추천값 소스", _ai_src_opts, key="t4_ai_src_sel")
            with _cmp_c2:
                if _is4:
                    _test_src_sel = st.radio("Current (TEST) Value Basis", ["Use Data Average", "Manual Input"], horizontal=True, key="t4_test_src_mode")
                else:
                    _test_src_sel = st.radio("현재(TEST) 값 기준", ["데이터 평균값 사용", "직접 입력"], horizontal=True, key="t4_test_src_mode")
            with _cmp_c3:
                _top_n_sel = st.number_input("Number of Priority Items to Show" if _is4 else "우선 변경 대상 표시 개수", min_value=1, max_value=34, value=5, step=1, key="t4_top_n")

            _ai_x = list(st.session_state['opt_result_x']) if _ai_src_sel.startswith("Tab1") else list(st.session_state['sim_dx_result_x'])

            _df_t4c = st.session_state.get('df_imputed_ref')
            _test_x = []
            for v in X_list:
                if _df_t4c is not None and v in _df_t4c.columns:
                    _cv = pd.to_numeric(_df_t4c[v], errors='coerce')
                    _test_x.append(float(_cv.mean()) if not _cv.dropna().empty else float((db.get(v,(0,1))[0]+db.get(v,(0,1))[1])/2))
                else:
                    _test_x.append(float((db.get(v,(0,1))[0]+db.get(v,(0,1))[1])/2))

            if _test_src_sel in ("직접 입력", "Manual Input"):
                _manual_exp_lbl = "▸ Manually Enter Current (TEST) Values (34 variables)" if _is4 else "▸ 현재(TEST) 값 직접 입력 (34개 변수)"
                with st.expander(_manual_exp_lbl, expanded=False):
                    st.markdown("<div style='max-height:380px; overflow-y:auto; padding-right:10px;'>", unsafe_allow_html=True)
                    for _i, v in enumerate(X_list):
                        st.session_state.setdefault(f"t4_test_val_{v.lower()}", _test_x[_i])
                        _new_val = st.number_input(v, value=float(st.session_state[f"t4_test_val_{v.lower()}"]),
                                                    step=0.001, format="%.3f", key=f"t4_test_val_{v.lower()}")
                        _test_x[_i] = _new_val
                    st.markdown("</div>", unsafe_allow_html=True)

            # 차이 계산 (AI - TEST), 절댓값 큰 순 정렬
            _diff_rows = []
            for _i, v in enumerate(X_list):
                _ai_v = float(_ai_x[_i])
                _test_v = float(_test_x[_i])
                _diff_rows.append((v, _ai_v, _test_v, _ai_v - _test_v))
            _diff_rows.sort(key=lambda r: abs(r[3]), reverse=True)
            _top_vars = set(r[0] for r in _diff_rows[:int(_top_n_sel)])

            _rows_html_diff = ""
            _diff_export_rows = []
            for v, ai_v, test_v, diff_v in _diff_rows:
                _is_top = v in _top_vars
                if _is4:
                    _diff_export_rows.append({
                        "Design/Process Variable": v, "AI Recommended": round(ai_v, 4), "Current (TEST)": round(test_v, 4),
                        "Difference (AI-TEST)": round(diff_v, 4), "Priority Change": "Y" if _is_top else ""
                    })
                else:
                    _diff_export_rows.append({
                        "설계/공정 변수": v, "AI 추천값": round(ai_v, 4), "현재(TEST) 값": round(test_v, 4),
                        "차이(AI-TEST)": round(diff_v, 4), "우선 변경 대상": "Y" if _is_top else ""
                    })
                _row_bg = "background:#3d3106;" if _is_top else ""
                _diff_color = "#facc15" if _is_top else ("#38bdf8" if diff_v > 0 else "#f87171" if diff_v < 0 else "#64748b")
                _sign_txt = "(+)" if diff_v > 0 else "(-)" if diff_v < 0 else ""
                _priority_txt = "★ Priority" if _is4 else "★ 우선 변경"
                _priority_badge = (
                    f"<span style='background:#78350f;color:#fde047;font-size:0.62rem;padding:1px 6px;"
                    f"border-radius:3px;margin-left:6px;'>{_priority_txt}</span>" if _is_top else ""
                )
                _rows_html_diff += (
                    f"<tr style='{_row_bg}'>"
                    f"<td style='padding:5px 8px;color:#e2e8f0;font-weight:700;white-space:nowrap;'>{v}{_priority_badge}</td>"
                    f"<td style='padding:5px 8px;color:#94a3b8;text-align:center;font-family:monospace;'>{ai_v:.3f}</td>"
                    f"<td style='padding:5px 8px;color:#94a3b8;text-align:center;font-family:monospace;'>{test_v:.3f}</td>"
                    f"<td style='padding:5px 8px;color:{_diff_color};text-align:center;font-weight:700;font-family:monospace;'>{_sign_txt} {abs(diff_v):.3f}</td>"
                    f"</tr>"
                )

            if _is4:
                _th1, _th2, _th3, _th4 = "Design/Process Variable", "AI Recommended", "Current (TEST) Value", "Difference (AI-TEST)"
                _diff_footer = f"★ Yellow highlight = top {int(_top_n_sel)} variables with the largest difference (priority for real-world change)"
            else:
                _th1, _th2, _th3, _th4 = "설계/공정 변수", "AI 추천값", "현재(TEST) 값", "차이 (AI-TEST)"
                _diff_footer = f"★ 노란색 강조 = 차이가 가장 큰 상위 {int(_top_n_sel)}개 변수 (실물 변경 시 우선순위)"
            st.markdown(
                f"<div style='background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;padding:12px 14px;max-height:420px;overflow-y:auto;'>"
                f"<table style='width:100%;border-collapse:collapse;'>"
                f"<thead><tr style='border-bottom:1px solid #1e3a5f;'>"
                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:left;'>{_th1}</th>"
                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:center;'>{_th2}</th>"
                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:center;'>{_th3}</th>"
                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:center;'>{_th4}</th>"
                f"</tr></thead>"
                f"<tbody>{_rows_html_diff}</tbody>"
                f"</table>"
                f"<div style='color:#6b7fa3;font-size:0.72rem;margin-top:8px;'>{_diff_footer}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

            # 스펙 판정 비교 (AI 값 vs TEST 값 기준 예측)
            st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
            _spec_cmp_title = "▣ Quality Prediction Comparison: AI Value vs Current (TEST) Value" if _is4 else "▣ AI 값 vs 현재(TEST) 값 기준 품질 예측 비교"
            st.markdown(f"<p style='font-size:0.85rem;font-weight:700;color:#10b981;'>{_spec_cmp_title}</p>", unsafe_allow_html=True)
            _scaler_c = st.session_state['scaler']
            _q_ai_c = _scaler_c.transform(pd.DataFrame([_ai_x], columns=X_list))
            _q_test_c = _scaler_c.transform(pd.DataFrame([_test_x], columns=X_list))
            _spec_rows_html = ""
            _spec_export_rows = []
            for tgt in valid_tgts:
                _mdl_c = st.session_state.get(f'model_{tgt.lower()}')
                if _mdl_c is None:
                    continue
                _p_ai = float(_mdl_c.predict(_q_ai_c)[0])
                _p_test = float(_mdl_c.predict(_q_test_c)[0])
                _lo_c, _hi_c = spec_limits.get(tgt, (None, None))
                if _lo_c is None:
                    _spec_str_c = "-"
                    _judge_ai_txt, _judge_test_txt = "-", "-"
                    _jc_ai, _jc_test = "#64748b", "#64748b"
                else:
                    _spec_str_c = f"{_lo_c}~{_hi_c}"
                    _ok_ai = _lo_c <= _p_ai <= _hi_c
                    _ok_test = _lo_c <= _p_test <= _hi_c
                    _judge_ai_txt = "OK" if _ok_ai else "NG"
                    _judge_test_txt = "OK" if _ok_test else "NG"
                    _jc_ai = "#10b981" if _ok_ai else "#f87171"
                    _jc_test = "#10b981" if _ok_test else "#f87171"
                if _is4:
                    _spec_export_rows.append({
                        "Target": tgt, "AI Predicted": round(_p_ai, 4), "AI Judgement": _judge_ai_txt,
                        "TEST Predicted": round(_p_test, 4), "TEST Judgement": _judge_test_txt, "Spec.": _spec_str_c
                    })
                else:
                    _spec_export_rows.append({
                        "타겟": tgt, "AI 예측값": round(_p_ai, 4), "AI 판정": _judge_ai_txt,
                        "TEST 예측값": round(_p_test, 4), "TEST 판정": _judge_test_txt, "Spec.": _spec_str_c
                    })
                _spec_rows_html += (
                    f"<tr>"
                    f"<td style='padding:5px 8px;color:#e2e8f0;font-weight:700;'>{tgt}</td>"
                    f"<td style='padding:5px 8px;text-align:center;font-family:monospace;color:#38bdf8;font-weight:700;'>{_p_ai:.3f}</td>"
                    f"<td style='padding:5px 8px;text-align:center;'><span style='color:{_jc_ai};font-weight:700;'>{_judge_ai_txt}</span></td>"
                    f"<td style='padding:5px 8px;text-align:center;font-family:monospace;color:#94a3b8;'>{_p_test:.3f}</td>"
                    f"<td style='padding:5px 8px;text-align:center;'><span style='color:{_jc_test};font-weight:700;'>{_judge_test_txt}</span></td>"
                    f"<td style='padding:5px 8px;text-align:center;color:#64748b;font-size:0.78rem;'>{_spec_str_c}</td>"
                    f"</tr>"
                )
            if _is4:
                _sh1, _sh2, _sh3, _sh4, _sh5, _sh6 = "Target", "AI Predicted", "Judgement", "TEST Predicted", "Judgement", "Spec."
            else:
                _sh1, _sh2, _sh3, _sh4, _sh5, _sh6 = "타겟", "AI 예측값", "판정", "TEST 예측값", "판정", "Spec."
            st.markdown(
                f"<div style='background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;padding:12px 14px;'>"
                f"<table style='width:100%;border-collapse:collapse;'>"
                f"<thead><tr style='border-bottom:1px solid #1e3a5f;'>"
                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:left;'>{_sh1}</th>"
                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:center;'>{_sh2}</th>"
                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:center;'>{_sh3}</th>"
                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:center;'>{_sh4}</th>"
                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:center;'>{_sh5}</th>"
                f"<th style='padding:4px 8px;font-size:0.68rem;color:#64748b;text-align:center;'>{_sh6}</th>"
                f"</tr></thead>"
                f"<tbody>{_spec_rows_html}</tbody>"
                f"</table></div>",
                unsafe_allow_html=True
            )

            # 다운로드 (설계/공정 변수 차이 표 + 품질 예측 비교 표, 시트 2개)
            st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
            _df_diff_export = pd.DataFrame(_diff_export_rows)
            _df_spec_export = pd.DataFrame(_spec_export_rows)
            _dl_col1, _dl_col2 = st.columns([1, 1])
            with _dl_col1:
                _dl_fmt4 = st.selectbox(L_G['dl_format'], ["Excel (.xlsx)", "Database (.db)"],
                                         key="t4_cmp_dl_fmt", label_visibility="collapsed")
            _dl_btn_lbl4 = "□ Download Comparison Result" if _is4 else "□ 비교 결과 다운로드"
            with _dl_col2:
                if "Excel" in _dl_fmt4:
                    _buf4 = io.BytesIO()
                    with pd.ExcelWriter(_buf4) as _writer4:
                        _df_diff_export.to_excel(_writer4, index=False, sheet_name='AI_vs_TEST_Diff')
                        _df_spec_export.to_excel(_writer4, index=False, sheet_name='Spec_Judgement')
                    st.download_button(
                        label=_dl_btn_lbl4,
                        data=_buf4.getvalue(),
                        file_name=f"AI_vs_TEST_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="t4_cmp_dl_xlsx"
                    )
                else:
                    _conn4 = sqlite3.connect(":memory:")
                    _df_diff_export.to_sql("ai_vs_test_diff", _conn4, index=False, if_exists="replace")
                    _df_spec_export.to_sql("spec_judgement", _conn4, index=False, if_exists="replace")
                    _backup_conn4 = sqlite3.connect("temp_t4_cmp.db")
                    _conn4.backup(_backup_conn4); _backup_conn4.close(); _conn4.close()
                    with open("temp_t4_cmp.db", "rb") as f:
                        _db_bytes4 = f.read()
                    st.download_button(
                        label=_dl_btn_lbl4,
                        data=_db_bytes4,
                        file_name=f"AI_vs_TEST_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.db",
                        mime="application/x-sqlite3",
                        key="t4_cmp_dl_db"
                    )

else:
    st.info(L_G['engine_inactive'])
