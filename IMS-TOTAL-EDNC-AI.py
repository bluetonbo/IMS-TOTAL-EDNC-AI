import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
import sqlite3
import json
import os
import time
from datetime import datetime
from groq import Groq

GROQ_API_KEY = "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"

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



def generate_ai_report(defect_results, optimized_params, num_actions=3, lang="ko"):
    try:
        client = Groq(api_key=GROQ_API_KEY)

        # 사전 지정 조치 사항 (한국어 원문을 기준으로 사용)
        preset_text = ""
        if PRESET_ACTIONS_KO:
            preset_text = "\n\n[사전 지정 분석 항목 - 아래 항목들을 번호 순서대로 빠짐없이 먼저 분석하여 답변하세요]:\n"
            preset_text += "\n".join(f"{i+1}. {item}" for i, item in enumerate(PRESET_ACTIONS_KO))

        fact_block = _build_fact_block(defect_results)

        prompt_ko = f"""당신은 20년 경력의 사출 성형 공정 전문가입니다.

{fact_block}

[파라미터]: {optimized_params}{preset_text}

<중요 - 반드시 지켜야 할 원칙>
- 위 [사실 확인]에 없는 내용을 추측하거나 새로 만들어내지 마세요.
- 확실하지 않은 내용은 "데이터 부족으로 판단 불가"라고 답하세요.
- 본문에 등장하는 모든 수치(%, 값)는 위 [사실 확인]/[파라미터]에 있는 값과 정확히 일치해야 하며,
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
        "err_load": "Error loading file: ",
        "err_vars": "Could not find 10 defect variables in the uploaded data.",
        "warn_upload": "Please upload the Current Data (1) and either Historical (2) or CAE (3) data.",
        "main_title_1": "Total Injection ",
        "main_title_2": "AI Solution System",
        "main_desc": "Comprehensive Defect Diagnostic & Multi-Objective Optimization System v6.6 (10 Key Defects)",
        "m_status": "System Status",
        "m_vars": "Analyzed Variables",
        "m_reliability": "Expert Reliability",
        "m_opt": "Optimization Status / Algorithm",
        "status_active": "Operational",
        "status_standby": "Standby",
        "info_standby": "Please upload the converted data in the sidebar and start AI learning.",
        "tab_diag": "[ Diagnostic & Optimization ]",
        "tab_master": "[ Master Data ]",
        "sec_a": "A. Current Injection Parameters",
        "sec_c": "C. Defect Weights & Expert Constraints",
        "sec_c_sub2": "2. Expert Constraint Settings",
        "lbl_constant": "Select Variables to Keep Constant",
        "lbl_target": " Target",
        "lbl_expert_rel": "Expert Guideline Reliability (%)",
        "sec_d": "D. Intelligent Diagnosis & Optimization",
        "btn_diagnose": "Diagnose Current Risk",
        "btn_optimize": "Optimize Conditions",
        "opt_converged": "Converged",
        "opt_failed": "Failed",
        "dash_title": "AI Intelligent Dashboard",
        "opt_success_msg": "AI Recommendation Derived Successfully",
        "warn_need_optimize": "⚠ To generate the AI Expert Report, please click the 'Optimize Conditions' button above first to complete optimization.",
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
        "access_title": "사출 성형 AI 시스템 접속",
        "enter_pwd": "비밀번호 입력",
        "connect_sys": "시스템 연결",
        "invalid_pwd": "비밀번호가 올바르지 않습니다. 다시 시도해 주세요.",
        "data_mgmt": "데이터 관리",
        "upload_1": "1. 현재 최적 조건 데이터",
        "upload_2": "2. 누적 이력 데이터",
        "upload_3": "3. CAE 해석 데이터",
        "run_ai": "학습 초기화 및 데이터 통합 학습 실행",
        "err_load": "파일 로드 오류: ",
        "err_vars": "업로드된 데이터에서 10대 불량 변수를 찾을 수 없습니다.",
        "warn_upload": "현재 데이터(1)와 함께 이력 데이터(2) 또는 CAE 데이터(3)를 업로드해 주세요.",
        "main_title_1": "통합 사출 ",
        "main_title_2": "AI 솔루션 시스템",
        "main_desc": "종합 불량 진단 및 다목적 최적화 시스템 v6.6 (10대 핵심 불량)",
        "m_status": "시스템 상태",
        "m_vars": "분석된 변수",
        "m_reliability": "전문가 신뢰도",
        "m_opt": "최적화 상태 / 사용 알고리즘",
        "status_active": "가동 중",
        "status_standby": "대기 중",
        "info_standby": "사이드바에 변환된 데이터를 업로드하고 AI 학습을 시작해 주세요.",
        "tab_diag": "[ 진단 및 최적화 ]",
        "tab_master": "[ 마스터 데이터 ]",
        "sec_a": "A. 현재 사출 조건 파라미터",
        "sec_c": "C. 불량 가중치 및 전문가 제약 조건",
        "sec_c_sub2": "2. 전문 제약 조건 설정",
        "lbl_constant": "고정 상태를 유지할 변수 선택",
        "lbl_target": " 목표치",
        "lbl_expert_rel": "전문가 가이드라인 신뢰도 (%)",
        "sec_d": "D. 지능형 진단 및 최적화",
        "btn_diagnose": "현재 리스크 진단",
        "btn_optimize": "역추론 최적화 탐색 실행",
        "opt_converged": "수렴 완료",
        "opt_failed": "최적화 실패",
        "dash_title": "AI 지능형 대시보드",
        "opt_success_msg": "AI 추천 조건 도출 완료",
        "warn_need_optimize": "⚠ AI 전문가 리포트를 생성하려면 먼저 위의 '조건 최적화' 버튼을 눌러 최적화를 완료해 주세요.",
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
    st.session_state.lang = "en"

L = LANG_DICT[st.session_state.lang]

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

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
        </style>
    """, unsafe_allow_html=True)

    col_space, col_lang = st.columns([9, 1])
    with col_lang:
        if st.button("KO / EN", key="lang_btn_auth"):
            st.session_state.lang = "ko" if st.session_state.lang == "en" else "en"
            st.rerun()

    _, center, _ = st.columns([0.5, 2, 0.5])
    with center:
        st.markdown(f"<br><br><h2 style='text-align: center;'>{L['access_title']}</h2>", unsafe_allow_html=True)
        pwd = st.text_input(L['enter_pwd'], type="password")
        if st.button(L['connect_sys']):
            if pwd == "ednc1234":
                st.session_state.authenticated = True
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
        'df_injection': pd.DataFrame(),
        'ui_display_vars': [],
        'global_process_vars': [],
        'global_bounds': {},
        'expert_constraints': {},
        'current_inputs': {},
        'defect_weights': {k: 1.0 for k in TARGET_VARS.keys()},
        'defect_switches': {k: True for k in TARGET_VARS.keys()},
        'ver': 0,
        'expert_reliability': 0.0,
        'last_res_val': None,
        'last_defect_risks': {},
        'last_opt_df': None,
        'optimization_success': "N/A",
        'selected_algorithm': "N/A",
        'prepared_db_file': None,
        'data_changed_since_save': False,
        'show_feature_guide': False
    })

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Noto+Sans+KR:wght@300;400;700&display=swap');
    .stApp {
        background-color: #0b0c10 !important;
        color: #e1e1e1 !important;
        font-family: 'Inter', sans-serif;
    }
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
    .metric-label { color: #94a3b8; font-size: 0.85rem; margin-bottom: 5px; }
    .metric-value { color: #ffffff; font-size: 1.2rem; font-weight: 700; }
    .section-title {
        display: flex;
        align-items: center;
        color: #00e5ff !important;
        font-weight: 600 !important;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        margin-top: 1.5rem;
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
    .stDownloadButton>button {
        background: linear-gradient(180deg, #2e7d32 0%, #1b5e20 100%) !important;
        border: 1px solid #2e7d32 !important;
    }
    h1 { color: #ffffff !important; font-weight: 800 !important; letter-spacing: -0.04em; }
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
    </style>
""", unsafe_allow_html=True)


# =================================================================
# 1. 사이드바 (데이터 관리)
# =================================================================
with st.sidebar:
    st.markdown(f"<h2 style='color:#FFFFFF; font-size:1.5rem;'>{L['data_mgmt']}</h2>", unsafe_allow_html=True)

    with st.sidebar.form(key='data_upload_form'):
        u1 = st.file_uploader(L['upload_1'], type=['csv', 'xlsx'])
        u2 = st.file_uploader(L['upload_2'], type=['csv', 'xlsx', 'db'])
        u3 = st.file_uploader(L['upload_3'], type=['csv', 'xlsx'])
        sub_btn = st.form_submit_button(L['run_ai'])

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

        df_i, df_v, df_r = load_data(u1), load_data(u2), load_data(u3)

        if df_i is not None and (df_v is not None or df_r is not None):
            # 1. 데이터 정리
            for df in [df_i, df_v, df_r]:
                if df is not None:
                    df.rename(columns=OLD_TO_NEW_MAP, inplace=True)
                    df.drop(columns=df.columns[df.columns.duplicated()], inplace=True)

            df_comb = pd.concat(
                [df for df in [df_i, df_v, df_r] if df is not None],
                ignore_index=True
            )
            df_comb.drop_duplicates(ignore_index=True, inplace=True)

            available_targets = [t for t in TARGET_VARS.keys() if t in df_comb.columns]

            if not available_targets:
                st.sidebar.error(L['err_vars'])
            else:
                df_comb.dropna(subset=available_targets, inplace=True)
                vars_list = [c for c in df_comb.columns if c not in TARGET_VARS.keys() and c != 'vars']

                if not vars_list or df_comb.empty:
                    st.sidebar.error("데이터에 분석 가능한 변수가 없거나 데이터가 비어 있습니다.")
                else:
                    models_dict, scalers_dict = {}, {}
                    
                    # 학습 진행 상황 UI 추가
                    st.sidebar.markdown("<br>", unsafe_allow_html=True)
                    prog_text = st.sidebar.empty()
                    prog_bar = st.sidebar.progress(0)
                    total_targets = len(available_targets)

                    for idx, target in enumerate(available_targets):
                        # 프로그레스 바 텍스트 업데이트
                        pct = int(((idx + 1) / total_targets) * 100)
                        prog_text.markdown(f"▸ **{L['learning_progress']} ({idx+1}/{total_targets}): {target} ({pct}%)**")
                        prog_bar.progress((idx + 1) / total_targets)
                        time.sleep(0.1) # 시각적 피드백을 위한 짧은 대기

                        t_series = (
                            df_comb[target].iloc[:, 0]
                            if isinstance(df_comb[target], pd.DataFrame)
                            else df_comb[target]
                        )
                        target_vals = np.where(t_series >= DEFECT_THRESHOLD, 1, 0)

                        if vars_list and (len(np.unique(target_vals)) >= 2):
                            scaler = MinMaxScaler().fit(df_comb[vars_list])
                            model = LogisticRegression(max_iter=1000).fit(
                                scaler.transform(df_comb[vars_list]), target_vals
                            )
                            models_dict[target] = model
                            scalers_dict[target] = scaler

                    bounds_dict = {
                        v: (int(np.floor(df_comb[v].min())), int(np.ceil(df_comb[v].max())) + 1)
                        for v in vars_list
                    }

                    st.session_state.update({
                        'models': models_dict,
                        'scalers': scalers_dict,
                        'df_injection': df_comb,
                        'global_process_vars': vars_list,
                        'global_bounds': bounds_dict,
                        'ui_display_vars': [
                            c for c in df_comb.columns
                            if c not in TARGET_VARS.keys() and c != 'vars'
                        ],
                        'prepared_db_file': None,
                        'data_changed_since_save': True
                    })

                    init_row = df_i.iloc[0].to_dict()
                    for v in vars_list:
                        st.session_state['current_inputs'][v] = int(round(float(init_row.get(v, 0))))

                    st.rerun()
        else:
            st.sidebar.warning(L['warn_upload'])

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<h3 style='color:#00e5ff; font-size:1.1rem;'>{L['db_export_title']}</h3>",
        unsafe_allow_html=True
    )

    if not st.session_state['df_injection'].empty:
        if st.sidebar.button(L['db_prepare_btn'], key="btn_create_db_snapshot"):
            today_str = datetime.now().strftime("%Y%m%d")

            if st.session_state.get('data_changed_since_save', True) or st.session_state['prepared_db_file'] is None:
                idx = 1
                while True:
                    candidate = f"{today_str}-{idx}.db"
                    if not os.path.exists(candidate):
                        final_filename = candidate
                        break
                    idx += 1
                st.session_state['prepared_db_file'] = final_filename
            else:
                final_filename = st.session_state['prepared_db_file']

            try:
                existing_df = pd.DataFrame()
                if os.path.exists(final_filename):
                    try:
                        conn_old = sqlite3.connect(final_filename)
                        df_old_raw = pd.read_sql_query("SELECT vars FROM production_log", conn_old)
                        conn_old.close()
                        existing_df = pd.json_normalize([json.loads(x) for x in df_old_raw['vars']])
                    except Exception:
                        existing_df = pd.DataFrame()

                df_to_save = st.session_state['df_injection'].copy()
                if 'vars' in df_to_save.columns:
                    df_to_save = df_to_save.drop(columns=['vars'], errors='ignore')

                if not existing_df.empty:
                    df_to_save = pd.concat(
                        [existing_df, df_to_save], ignore_index=True
                    ).drop_duplicates(ignore_index=True)
                else:
                    df_to_save = df_to_save.drop_duplicates(ignore_index=True)

                conn = sqlite3.connect(final_filename)
                df_to_save['vars'] = df_to_save.apply(
                    lambda row: json.dumps(row.to_dict()), axis=1
                )
                df_to_save[['vars']].to_sql("production_log", conn, if_exists="replace", index=False)
                conn.close()

                st.session_state['data_changed_since_save'] = False
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

        if st.session_state.get('prepared_db_file'):
            target_file = st.session_state['prepared_db_file']
            if os.path.exists(target_file):
                try:
                    with open(target_file, "rb") as f:
                        db_bytes = f.read()

                    if not st.session_state.get('data_changed_since_save', True):
                        st.sidebar.markdown(
                            f"<span style='color:#a3e635; font-size:0.85rem;'>{L['db_current_latest']}</span>",
                            unsafe_allow_html=True
                        )

                    st.sidebar.markdown(f"✓ {L['db_prepared_msg']} `{target_file}`")
                    st.sidebar.download_button(
                        label=L['db_pc_download'],
                        data=db_bytes,
                        file_name=target_file,
                        mime="application/x-sqlite3",
                        key="db_final_download_action"
                    )
                except Exception as e:
                    st.sidebar.error(f"File Load Error: {e}")
    else:
        st.sidebar.warning(L['db_save_empty'])


# =================================================================
# 2. 메인 화면
# =================================================================
col_title, col_lang_switch = st.columns([8.5, 1.5])
with col_title:
    st.markdown(
        f"<h1 style='text-align: left; margin-bottom: 0;'>"
        f"{L['main_title_1']}<span style='color:#00e5ff;'>{L['main_title_2']}</span></h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='color:#64748b; margin-bottom: 1rem;'>{L['main_desc']}</p>",
        unsafe_allow_html=True
    )
with col_lang_switch:
    if st.button("KO / EN", key="lang_btn_main"):
        st.session_state.lang = "ko" if st.session_state.lang == "en" else "en"
        st.rerun()

is_active = len(st.session_state.get('models', {})) > 0
status_text = L['status_active'] if is_active else L['status_standby']
dot_color = "#00e5ff" if is_active else "#64748b"
var_count = len(st.session_state.get('ui_display_vars', []))
exp_weight = int(st.session_state.get('expert_reliability', 0.0) * 100)

opt_status = st.session_state.get('optimization_success', "N/A")
algo_info = st.session_state.get('selected_algorithm', "N/A")

if opt_status == "Converged":
    opt_display = f"{L['opt_converged']} ({algo_info})"
elif opt_status == "Failed":
    opt_display = f"{L['opt_failed']}"
else:
    opt_display = "N/A"

m1, m2, m3, m4 = st.columns(4)
m1.markdown(
    f'''<div class="metric-container">
        <div class="metric-label">{L['m_status']}</div>
        <div class="metric-value">
            <span style="color:{dot_color}; margin-right:5px;">●</span>
            <span style="color:#FFFFFF;">{status_text}</span>
        </div>
    </div>''',
    unsafe_allow_html=True
)
m2.markdown(
    f'''<div class="metric-container">
        <div class="metric-label">{L['m_vars']}</div>
        <div class="metric-value">{var_count} EA</div>
    </div>''',
    unsafe_allow_html=True
)
m3.markdown(
    f'''<div class="metric-container">
        <div class="metric-label">{L['m_reliability']}</div>
        <div class="metric-value">{exp_weight}%</div>
    </div>''',
    unsafe_allow_html=True
)
m4.markdown(
    f'''<div class="metric-container">
        <div class="metric-label">{L['m_opt']}</div>
        <div class="metric-value" style="font-size: 1.05rem;">{opt_display}</div>
    </div>''',
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

if not is_active:
    st.info(L['info_standby'])

if is_active:
    t1, t2 = st.tabs([L['tab_diag'], L['tab_master']])

    with t1:
        # A. 현재 사출 조건 파라미터 입력 (최적화 후 결과값이 여기에 연동됨)
        st.markdown(
            f'<div class="section-title"><span class="square-icon"></span>{L["sec_a"]}</div>',
            unsafe_allow_html=True
        )
        cols = st.columns(3)
        for i, var in enumerate(st.session_state['ui_display_vars']):
            with cols[i % 3]:
                curr_val = st.session_state['current_inputs'].get(var, 0)
                # 세션 상태의 값을 직접 슬라이더에 바인딩하고, ver 값을 활용해 UI 강제 업데이트
                st.session_state['current_inputs'][var] = st.slider(
                    f"{var}",
                    0,
                    int(curr_val * 2) if curr_val > 0 else 100,
                    int(curr_val),
                    step=1,
                    key=f"sl_{var}_{st.session_state['ver']}"
                )

        st.divider()

        # C. 불량 가중치 및 전문가 제약 조건
        st.markdown(
            f'<div class="section-title"><span class="square-icon"></span>{L["sec_c"]}</div>',
            unsafe_allow_html=True
        )
        active_targets = list(st.session_state['models'].keys())
        w_cols = st.columns(3)
        for idx, target_key in enumerate(active_targets):
            with w_cols[idx % 3]:
                is_on = st.checkbox(
                    f"{TARGET_VARS[target_key]}",
                    value=st.session_state['defect_switches'].get(target_key, True),
                    key=f"onoff_{target_key}"
                )
                st.session_state['defect_switches'][target_key] = is_on
                st.session_state['defect_weights'][target_key] = st.slider(
                    "",
                    0.0, 5.0,
                    float(st.session_state['defect_weights'].get(target_key, 1.0)),
                    step=0.5,
                    disabled=not is_on,
                    key=f"weight_{target_key}"
                )
                st.markdown("<br>", unsafe_allow_html=True)

        st.write(L['sec_c_sub2'])
        selected_expert_vars = st.multiselect(
            L['lbl_constant'],
            options=st.session_state['ui_display_vars'],
            default=list(st.session_state['expert_constraints'].keys())
        )
        if selected_expert_vars:
            cols_b = st.columns(3)
            for i, v_name in enumerate(selected_expert_vars):
                with cols_b[i % 3]:
                    st.session_state['expert_constraints'].setdefault(
                        v_name, {'limit': st.session_state['current_inputs'].get(v_name, 0)}
                    )
                    st.session_state['expert_constraints'][v_name]['limit'] = st.number_input(
                        f"{v_name}{L['lbl_target']}",
                        value=int(st.session_state['expert_constraints'][v_name]['limit']),
                        step=1
                    )

        st.session_state['expert_reliability'] = (
            st.slider(L['lbl_expert_rel'], 0, 100, int(st.session_state['expert_reliability'] * 100)) / 100.0
        )
        st.divider()

        # D. 지능형 진단 및 최적화
        def calculate_total_risk(input_vals_list):
            all_v = st.session_state['global_process_vars']
            df_input = pd.DataFrame([input_vals_list], columns=all_v)
            total_weighted_risk = 0.0
            weight_sum = 0.0
            for target_key, model in st.session_state['models'].items():
                if st.session_state['defect_switches'].get(target_key, True):
                    scaler = st.session_state['scalers'][target_key]
                    prob = model.predict_proba(scaler.transform(df_input))[0, 1]
                    weight = st.session_state['defect_weights'][target_key]
                    total_weighted_risk += prob * weight
                    weight_sum += weight
            weight_sum = weight_sum if weight_sum > 0 else 1e-9
            avg_defect_risk = total_weighted_risk / weight_sum
            penalty = sum(
                abs(input_vals_list[list(all_v).index(v)] - c['limit']) / (c['limit'] + 1e-9)
                for v, c in st.session_state['expert_constraints'].items()
                if v in all_v
            )
            return min(1.0, avg_defect_risk + (penalty * st.session_state['expert_reliability']))

        def get_individual_risks(input_vals_list):
            all_v = st.session_state['global_process_vars']
            df_input = pd.DataFrame([input_vals_list], columns=all_v)
            risks = {}
            for target_key, model in st.session_state['models'].items():
                scaler = st.session_state['scalers'][target_key]
                risks[target_key] = model.predict_proba(scaler.transform(df_input))[0, 1]
            return risks

        st.markdown(
            f'<div class="section-title"><span class="square-icon"></span>{L["sec_d"]}</div>',
            unsafe_allow_html=True
        )
        c_btn1, c_btn2 = st.columns(2)

        with c_btn1:
            if st.button(L['btn_diagnose'], type="primary"):
                all_v = st.session_state['global_process_vars']
                input_vals = [float(st.session_state['current_inputs'].get(v, 0.0)) for v in all_v]
                st.session_state['last_res_val'] = calculate_total_risk(input_vals)
                st.session_state['last_defect_risks'] = get_individual_risks(input_vals)
                st.session_state['last_opt_df'] = None
                st.session_state['optimization_success'] = "N/A"
                st.session_state['selected_algorithm'] = "N/A"
                st.session_state['show_feature_guide'] = False

                new_row = {v: st.session_state['current_inputs'].get(v, 0.0) for v in all_v}
                for target_key, r_val in st.session_state['last_defect_risks'].items():
                    new_row[target_key] = r_val

                new_df = pd.DataFrame([new_row])
                st.session_state['df_injection'] = pd.concat(
                    [st.session_state['df_injection'], new_df], ignore_index=True
                )
                st.session_state['data_changed_since_save'] = True
                st.rerun()

        with c_btn2:
            if st.button(L['btn_optimize']):
                all_v = st.session_state['global_process_vars']
                x0 = [float(st.session_state['current_inputs'].get(v, 0.0)) for v in all_v]
                bnds = [st.session_state['global_bounds'].get(v, (0, 100)) for v in all_v]

                algorithms = ['L-BFGS-B', 'SLSQP', 'Powell', 'Nelder-Mead']
                best_fun = float('inf')
                best_res = None
                chosen_algo = "None"
                
                # 역추론 최적화 탐색 진행 상황 표시 UI
                st.markdown("<br>", unsafe_allow_html=True)
                opt_prog_text = st.empty()
                opt_prog_bar = st.progress(0)
                opt_prog_detail = st.empty()  # 스텝별 상세 진행 상황(현재 위험도 등) 출력용

                for i, algo in enumerate(algorithms):
                    pct = int(((i + 1) / len(algorithms)) * 100)
                    opt_prog_text.markdown(f"▸ **{L['opt_progress']} ({i+1}/{len(algorithms)}): {algo} ({pct}%)**")
                    opt_prog_bar.progress((i + 1) / len(algorithms))
                    opt_prog_detail.empty()  # 이전 알고리즘의 상세 로그가 남아있지 않도록 초기화
                    time.sleep(0.2) # 시각적 피드백

                    state = {'iter': 0}

                    # ---------------------------------------------------------
                    # [추가] 실시간 상세 진행 콜백 — 매 스텝마다 현재 위험도를 보여줍니다.
                    # ---------------------------------------------------------
                    def callback_min(xk, *args):
                        state['iter'] += 1
                        val = calculate_total_risk(xk)
                        opt_prog_detail.markdown(
                            f"&nbsp;&nbsp; ↳ → **[{algo}]** {L['opt_step_local']} ({L['opt_step_label']}: {state['iter']}) "
                            f"| {L['opt_current_risk']}: <span style='color:#00e5ff;'>{val*100:.2f}%</span>",
                            unsafe_allow_html=True
                        )

                    try:
                        res_temp = minimize(
                            calculate_total_risk, x0,
                            method=algo, bounds=bnds,
                            callback=callback_min,
                            options={'maxiter': 500}
                        )
                        if res_temp.success and res_temp.fun < best_fun:
                            best_fun = res_temp.fun
                            best_res = res_temp
                            chosen_algo = algo
                    except Exception:
                        continue

                # 하이브리드 멀티스타트(무작위 시작점) 단계도 동일하게 상세 진행 표시
                opt_prog_text.markdown(f"▸ **{L['opt_progress']}: Hybrid Multi-Start (L-BFGS-B)**")
                opt_prog_detail.empty()  # 이전 알고리즘의 상세 로그 초기화
                state = {'iter': 0}

                def callback_global(xk, *args):
                    state['iter'] += 1
                    val = calculate_total_risk(xk)
                    opt_prog_detail.markdown(
                        f"&nbsp;&nbsp; ↳ ⇒ **[Hybrid Multi-Start]** {L['opt_step_global']} ({L['opt_step_label']}: {state['iter']}) "
                        f"| {L['opt_current_risk']}: <span style='color:#a3e635;'>{val*100:.2f}%</span>",
                        unsafe_allow_html=True
                    )

                try:
                    random_x0 = [np.random.uniform(b[0], b[1]) for b in bnds]
                    res_global = minimize(
                        calculate_total_risk, random_x0,
                        method='L-BFGS-B', bounds=bnds,
                        callback=callback_global
                    )
                    if res_global.success and res_global.fun < best_fun:
                        best_fun = res_global.fun
                        best_res = res_global
                        chosen_algo = "Hybrid Multi-Start (L-BFGS-B)"
                except Exception:
                    pass
                
                opt_prog_text.empty()
                opt_prog_bar.empty()
                opt_prog_detail.empty()

                if best_res is not None:
                    final_x = [
                        np.clip(val, bnds[i][0], bnds[i][1])
                        for i, val in enumerate(best_res.x)
                    ]
                    opt_dict = {v: int(round(val)) for v, val in zip(all_v, final_x)}

                    st.session_state['last_res_val'] = calculate_total_risk(final_x)
                    st.session_state['last_defect_risks'] = get_individual_risks(final_x)
                    st.session_state['last_opt_df'] = pd.DataFrame([
                        {v: opt_dict.get(v, 0) for v in st.session_state['ui_display_vars']}
                    ])
                    st.session_state['optimization_success'] = "Converged"
                    st.session_state['selected_algorithm'] = chosen_algo
                    st.session_state['show_feature_guide'] = False

                    # 최적화된 파라미터 값을 현재 입력 상태에 덮어씌우고 버전을 올려 슬라이더 연동 처리
                    st.session_state['current_inputs'].update(opt_dict)
                    st.session_state['ver'] += 1

                    new_row = {v: opt_dict.get(v, 0) for v in all_v}
                    for target_key, r_val in st.session_state['last_defect_risks'].items():
                        new_row[target_key] = r_val

                    new_df = pd.DataFrame([new_row])
                    st.session_state['df_injection'] = pd.concat(
                        [st.session_state['df_injection'], new_df], ignore_index=True
                    )
                    st.session_state['data_changed_since_save'] = True
                    st.rerun()
                else:
                    st.session_state['optimization_success'] = "Failed"
                    st.session_state['selected_algorithm'] = "N/A"

        # AI 전문가 리포트 & 공정 진단 가이드 리포트 — 항상 표시 구역
        st.divider()
        
        # 공정 개선 가이드 (결과 진단 리포트) 추가 구역
        st.markdown(f"<h3 style='font-size: 1.1rem; color: #e1e1e1;'>▪ {L['guide_title']}</h3>", unsafe_allow_html=True)
        if st.button(f"▸ {L['btn_feature_guide']}", key="btn_feature_guide_trigger"):
            if st.session_state.get('last_res_val') is not None:
                st.session_state['show_feature_guide'] = True
            else:
                st.warning("진단 또는 최적화를 먼저 실행해 주세요.")
                
        if st.session_state.get('show_feature_guide', False) and st.session_state.get('last_res_val') is not None:
            risks = st.session_state['last_defect_risks']
            normal_count = sum(1 for r in risks.values() if r < DEFECT_THRESHOLD)
            out_spec_count = len(risks) - normal_count

            success_status = L['guide_all_success'] if out_spec_count == 0 else f"주의 필요 ({out_spec_count}개 이탈)"
            success_msg = L['guide_success_msg'] if out_spec_count == 0 else L['guide_partial_msg']
            icon = "✓" if out_spec_count == 0 else "⚠"

            guide_html = f"""
            <div style="background-color:#12141d; border:1px solid #2d3142; border-radius:10px; padding:20px 24px; margin-top:12px; margin-bottom: 24px;">
                <h3 style="margin-top:0; color:#e1e1e1;">## ▪ {L['guide_subtitle']}</h3>
                <blockquote style="border-left: 4px solid #10b981; padding-left: 10px; color:#94a3b8; font-size:0.95rem; background-color:#1a1c24; padding:10px;">
                    > {L['guide_pred_rel']}: <b>100.0%</b> | {L['guide_unachievable']}: <b>0개</b> | {L['guide_out_of_spec']}: <b>{out_spec_count}개</b> | {L['guide_normal']}: <b>{normal_count}개</b>
                </blockquote>
                <hr style="border-color:#2d3142; margin: 16px 0;">
                <h4 style="color:#10b981; margin-bottom: 12px;">### {icon} {success_status}</h4>
                <p style="line-height:1.6; font-size:0.95rem; color:#e1e1e1;">
                    {success_msg}
                </p>
            </div>
            """
            st.markdown(guide_html, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.session_state['last_opt_df'] is None:
            st.warning(L['warn_need_optimize'])
            st.button(L['btn_ai_report'], disabled=True, key="btn_report_disabled")
        else:
            st.success(L['opt_success_msg'])
            if st.button(L['btn_ai_report'], key="btn_report_active"):
                with st.spinner(L['spinner_analyzing']):
                    no_diag_text = "No diagnosis" if st.session_state.lang == "en" else "진단 없음"
                    results = st.session_state.get('last_defect_risks', no_diag_text)
                    params = st.session_state['last_opt_df'].to_dict(orient='records')
                    report = generate_ai_report(
                        results, params, num_actions=NUM_ACTIONS,
                        lang=st.session_state.lang
                    )

                    # <br> 태그 및 공백 정리 후 보기 좋게 렌더링
                    import re
                    cleaned = report
                    cleaned = re.sub(r'<br\s*/?>', '\n', cleaned)   # <br> → 줄바꿈
                    cleaned = re.sub(r'<[^>]+>', '', cleaned)        # 기타 HTML 태그 제거
                    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # 3줄 이상 공백 → 2줄

                    # 줄 단위로 파싱하여 번호 항목 강조
                    lines = cleaned.strip().split('\n')
                    html_lines = []
                    for line in lines:
                        line = line.strip()
                        if not line:
                            html_lines.append('<div style="margin:6px 0;"></div>')
                        elif re.match(r'^\d+[\.\)].', line):  # 1. 또는 1) 로 시작
                            html_lines.append(
                                f'<div style="margin:10px 0 4px 0; color:#00e5ff; font-weight:700; font-size:0.92rem;">{line}</div>'
                            )
                        else:
                            html_lines.append(
                                f'<div style="margin:2px 0 2px 12px; color:#e1e1e1; font-size:0.88rem; line-height:1.6;">{line}</div>'
                            )

                    report_html = f"""
                    <div style="background-color:#12141d; border:1px solid #2d3142;
                                border-radius:10px; padding:20px 24px; margin-top:12px;">
                        <div style="color:#94a3b8; font-size:0.8rem; margin-bottom:12px;
                                    letter-spacing:0.05em;">{L['report_box_title']}</div>
                        {''.join(html_lines)}
                    </div>"""
                    st.markdown(report_html, unsafe_allow_html=True)

        if st.session_state['last_res_val'] is not None:
            st.divider()
            val = st.session_state['last_res_val']
            total_risk_percent = int(round(val * 100))
            total_color = (
                "#00e5ff" if total_risk_percent < 30
                else "#ffab00" if total_risk_percent < 70
                else "#ff5252"
            )
            st.markdown(
                f"""<div style='background-color:#12141d; padding:25px; border-radius:10px;
                    border:1px solid {total_color}44;'>
                    <h4 style='margin-top:0; color:#94a3b8;'>{L['dash_title']}</h4>
                    <h2 style='color:{total_color}; font-size:3rem; margin:0;'>
                        {total_risk_percent}<span style='font-size:1.2rem;'>%</span>
                    </h2>
                </div>""",
                unsafe_allow_html=True
            )

            for target_key, full_name in TARGET_VARS.items():
                if target_key in st.session_state['last_defect_risks']:
                    r_val = st.session_state['last_defect_risks'][target_key]
                    r_perc = int(round(r_val * 100))
                    is_active_target = st.session_state['defect_switches'].get(target_key, True)
                    bar_color = (
                        "#00e5ff" if r_perc < 30
                        else "#ffab00" if r_perc < 70
                        else "#ff5252"
                    )
                    opacity_style = "opacity: 1.0;" if is_active_target else "opacity: 0.25;"
                    st.markdown(
                        f"""<div style="margin-bottom: 12px; {opacity_style}">
                            <span style="font-size:0.95rem; font-weight:600; color:#ffffff;">{full_name}</span>
                            <div class="custom-progress-container">
                                <div class="custom-progress-bar"
                                    style="width: {r_perc}%; background: {bar_color};">
                                    {r_perc}%
                                </div>
                            </div>
                        </div>""",
                        unsafe_allow_html=True
                    )

            if st.session_state['last_opt_df'] is not None:
                df = st.session_state['last_opt_df'].astype(int)
                headers = "".join([f"<th>{c}</th>" for c in df.columns])
                rows = "".join([f"<td>{v}</td>" for v in df.values[0]])
                st.markdown(
                    f"""<div style="overflow-x: auto;">
                        <table class="optimized-table">
                            <thead><tr>{headers}</tr></thead>
                            <tbody><tr>{rows}</tr></tbody>
                        </table>
                    </div>""",
                    unsafe_allow_html=True
                )
                csv = st.session_state['last_opt_df'].to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label=L['btn_download'],
                    data=csv,
                    file_name='total_optimized_params.csv',
                    mime='text/csv'
                )

    with t2:
        if not st.session_state['df_injection'].empty:
            st.dataframe(st.session_state['df_injection'], use_container_width=True)
