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



def generate_ai_report(defect_results, optimized_params, num_actions=3, lang="ko", is_optimized=True):
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
        "tab_diag": "▶  Diagnostic & Optimization",
        "tab_master": "▶  Master Data & Analytics",
        "sec_a": "A. Current Injection Parameters",
        "sec_c": "B. Defect Weights & Expert Constraints",
        "sec_c_sub2": "2. Expert Constraint Settings",
        "lbl_constant": "Select Variables to Keep Constant",
        "lbl_target": " Target",
        "lbl_expert_rel": "Expert Guideline Reliability (%)",
        "sec_d": "C. Intelligent Diagnosis & Optimization",
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
        "tab_diag": "▶  진단 및 최적화",
        "tab_master": "▶  마스터 데이터 & 분석",
        "sec_a": "A. 현재 사출 조건 파라미터",
        "sec_c": "B. 불량 가중치 및 전문가 제약 조건",
        "sec_c_sub2": "2. 전문 제약 조건 설정",
        "lbl_constant": "고정 상태를 유지할 변수 선택",
        "lbl_target": " 목표치",
        "lbl_expert_rel": "전문가 가이드라인 신뢰도 (%)",
        "sec_d": "C. 지능형 진단 및 최적화",
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
    .metric-label { color: #cbd5e1; font-size: 0.85rem; margin-bottom: 5px; }
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
        margin-bottom: 4px !important;
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

        # [추가] 데이터 로딩 진행 상황 표시
        load_prog    = st.sidebar.progress(0)
        load_status  = st.sidebar.empty()
        load_detail  = st.sidebar.empty()

        def _step(pct, msg, detail="", done_steps=None):
            load_prog.progress(pct, text=f"{pct}%")
            load_status.markdown(
                f"<div style='background:#1a1c24;border:1px solid #2d3142;border-left:3px solid #00e5ff;"
                f"border-radius:6px;padding:7px 12px;font-size:0.82rem;color:#e1e1e1;'>"
                f"▸ <b>{msg}</b></div>", unsafe_allow_html=True
            )

        done_steps = []
        _step(5, "Initializing...", "Checking uploaded files")

        _step(10, "Reading File 1 — Input Data", "Parsing CSV/XLSX...")
        df_i = load_data(u1)
        if df_i is not None:
            done_steps.append(f"File 1 loaded: {df_i.shape[0]} rows × {df_i.shape[1]} cols")

        _step(28, "Reading File 2 — Historical Data", "Parsing CSV/XLSX/DB...", done_steps)
        df_v = load_data(u2)
        if df_v is not None:
            done_steps.append(f"File 2 loaded: {df_v.shape[0]} rows × {df_v.shape[1]} cols")

        _step(46, "Reading File 3 — CAE Data", "Parsing CSV/XLSX...", done_steps)
        df_r = load_data(u3)
        if df_r is not None:
            done_steps.append(f"File 3 loaded: {df_r.shape[0]} rows × {df_r.shape[1]} cols")

        _step(60, "Merging & deduplicating datasets", "", done_steps)

        if df_i is not None and (df_v is not None or df_r is not None):
            # 1. 데이터 정리
            _step(65, "Cleaning & deduplicating...", "", done_steps)
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
                    reliability_dict = {}
                    algo_names_dict  = {}   # [추가] 타겟별 선택 알고리즘 이름
                    fi_dict          = {}   # [추가] 타겟별 Feature Importance
                    
                    # 학습 진행 상황 UI 추가
                    st.sidebar.markdown("<br>", unsafe_allow_html=True)
                    prog_text = st.sidebar.empty()
                    prog_bar  = st.sidebar.progress(0)
                    algo_text = st.sidebar.empty()   # [추가] 알고리즘 진행 표시용
                    total_targets = len(available_targets)

                    for idx, target in enumerate(available_targets):
                        # 프로그레스 바 텍스트 업데이트
                        pct = int(((idx + 1) / total_targets) * 100)
                        prog_text.markdown(
                            f"<span style='color:#e1e1e1; font-size:0.85rem;'>▸ <b>{L['learning_progress']} ({idx+1}/{total_targets}): {target} ({pct}%)</b></span>",
                            unsafe_allow_html=True
                        )
                        prog_bar.progress((idx + 1) / total_targets)
                        _step(70 + int(pct * 0.28), f"Training model {idx+1}/{total_targets}: {target}", "", done_steps)
                        time.sleep(0.1) # 시각적 피드백을 위한 짧은 대기

                        t_series = (
                            df_comb[target].iloc[:, 0]
                            if isinstance(df_comb[target], pd.DataFrame)
                            else df_comb[target]
                        )
                        target_vals = np.where(t_series >= DEFECT_THRESHOLD, 1, 0)

                        if vars_list and (len(np.unique(target_vals)) >= 2):
                            n_pos = int(target_vals.sum())
                            n_neg = int(len(target_vals) - n_pos)

                            chosen_C = _choose_regularization(n_pos, n_neg)
                            scaler   = MinMaxScaler().fit(df_comb[vars_list])
                            X_scaled = scaler.transform(df_comb[vars_list])

                            # [추가] 알고리즘별 진행 상황 콜백 정의
                            def _show_algo(algo_name, a_idx, a_total, _t=target):
                                if "Final fit" in str(algo_name):
                                    algo_text.markdown(
                                        f"<div style='background:#12141d;border:1px solid #2d3142;"
                                        f"border-left:3px solid #10b981;border-radius:5px;"
                                        f"padding:5px 10px;font-size:0.78rem;color:#a3e635;margin-top:3px;'>"
                                        f"✓ <b>{_t}</b> → {algo_name}</div>",
                                        unsafe_allow_html=True
                                    )
                                else:
                                    algo_text.markdown(
                                        f"<div style='background:#12141d;border:1px solid #2d3142;"
                                        f"border-left:3px solid #00e5ff;border-radius:5px;"
                                        f"padding:5px 10px;font-size:0.78rem;color:#cbd5e1;margin-top:3px;'>"
                                        f"  Testing <b style='color:#00e5ff;'>{algo_name}</b>"
                                        f" &nbsp;<span style='color:#cbd5e1;'>({a_idx}/{a_total})</span></div>",
                                        unsafe_allow_html=True
                                    )

                            # LR/RF/XGB/LGBM 4종 자동선택 (알고리즘 진행 콜백 포함)
                            best_model, best_algo, cv_score, fi = _auto_select_best_model(
                                X_scaled, target_vals, n_pos, n_neg, chosen_C,
                                algo_status_fn=_show_algo
                            )

                            models_dict[target]     = best_model
                            scalers_dict[target]    = scaler
                            algo_names_dict[target] = best_algo
                            if fi is not None:
                                fi_dict[target] = dict(zip(vars_list, fi.tolist() if hasattr(fi, 'tolist') else fi))

                            # [추가] 신뢰성 보강 3: 표본 부족 경고 (소수 클래스 5개 미만이면 저신뢰 표시)
                            low_sample = min(n_pos, n_neg) < 5

                            reliability_dict[target] = {
                                'cv_score': cv_score,
                                'n_pos': n_pos,
                                'n_neg': n_neg,
                                'n_total': n_pos + n_neg,
                                'C': chosen_C,
                                'low_sample': low_sample,
                                'algo': best_algo,   # [추가] 선택된 알고리즘 이름
                            }

                    bounds_dict = {
                        v: (float(df_comb[v].min()), float(df_comb[v].max()))
                        for v in vars_list
                    }

                    st.session_state.update({
                        'models': models_dict,
                        'scalers': scalers_dict,
                        'model_reliability': reliability_dict,
                        'model_algo_names': algo_names_dict,   # [추가]
                        'feature_importance': fi_dict,         # [추가]
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

                    load_prog.progress(100, text="✅ All done! AI engine ready.")
                    load_status.markdown("✅ **AI Engine ready.**")
                    algo_text.empty()

                    # [추가] rerun 후에도 유지되도록 세션에 요약 저장
                    st.session_state['algo_summary'] = {
                        t: {
                            'algo': algo_names_dict.get(t, 'N/A'),
                            'cv': reliability_dict.get(t, {}).get('cv_score')
                        }
                        for t in algo_names_dict
                    }
                    st.rerun()
        else:
            st.sidebar.warning(L['warn_upload'])

    # [추가] 학습 완료 후 알고리즘 선택 결과 — 항상 사이드바에 표시
    algo_summary = st.session_state.get('algo_summary', {})
    if algo_summary:
        summary_rows = ""
        for t_key, info in algo_summary.items():
            a_name = info.get('algo', 'N/A')
            cv     = info.get('cv')
            cv_str = f"{cv*100:.0f}%" if cv is not None else "N/A"
            summary_rows += (
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"padding:4px 2px;border-bottom:1px solid #23263a;'>"
                f"<span style='font-size:0.73rem;color:#cbd5e1;width:38%;'>{t_key}</span>"
                f"<span style='font-size:0.71rem;color:#a3e635;font-weight:600;width:40%;'>{a_name}</span>"
                f"<span style='font-size:0.70rem;color:#cbd5e1;width:22%;text-align:right;'>{cv_str}</span>"
                f"</div>"
            )
        st.sidebar.markdown(
            f"<div style='background:#12141d;border:1px solid #2d3142;"
            f"border-radius:8px;padding:10px 14px;margin-bottom:10px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"margin-bottom:8px;'>"
            f"<span style='font-size:0.78rem;color:#00e5ff;font-weight:600;'>Model Selection Result</span>"
            f"<span style='font-size:0.68rem;color:#cbd5e1;'>CV Accuracy</span>"
            f"</div>"
            f"<div style='display:flex;justify-content:space-between;font-size:0.67rem;"
            f"color:#64748b;padding-bottom:4px;border-bottom:1px solid #3f445e;margin-bottom:4px;'>"
            f"<span style='width:38%;'>Defect</span>"
            f"<span style='width:40%;'>Algorithm</span>"
            f"<span style='width:22%;text-align:right;'>Acc.</span>"
            f"</div>"
            f"{summary_rows}"
            f"</div>",
            unsafe_allow_html=True
        )

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
        f"<p style='color:#b0bec5; margin-bottom: 1rem;'>{L['main_desc']}</p>",
        unsafe_allow_html=True
    )
with col_lang_switch:
    if st.button("KO / EN", key="lang_btn_main"):
        st.session_state.lang = "ko" if st.session_state.lang == "en" else "en"
        st.rerun()

is_active = len(st.session_state.get('models', {})) > 0
status_text = L['status_active'] if is_active else L['status_standby']
dot_color = "#00e5ff" if is_active else "#b0bec5"
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

        # [추가] 섹션 A 슬라이더 ↔ Min/Max 숫자입력 콜백 함수
        def _on_sl_a_change(var, ver):
            val = st.session_state.get(f"sl_{var}_{ver}")
            if val is not None:
                st.session_state['current_inputs'][var] = val
                st.session_state[f"ni_a_{var}"] = float(val)

        def _on_ni_a_change(var, sl_min, sl_max, ver):
            raw = st.session_state.get(f"ni_a_{var}", sl_min)
            clamped = float(max(sl_min, min(raw, sl_max)))
            st.session_state['current_inputs'][var] = clamped
            st.session_state[f"sl_{var}_{ver}"] = clamped

        cols = st.columns(3)
        for i, var in enumerate(st.session_state['ui_display_vars']):
            with cols[i % 3]:
                curr_val = st.session_state['current_inputs'].get(var, 0)
                bounds   = st.session_state['global_bounds'].get(var, (0.0, 100.0))
                sl_min   = float(bounds[0])
                sl_max   = float(bounds[1])
                if sl_min == sl_max:
                    sl_max = sl_min + 1.0
                curr_clamped = float(max(sl_min, min(float(curr_val), sl_max)))
                step_v = max((sl_max - sl_min) / 100.0, 1.0) if (sl_max - sl_min) >= 100 else max((sl_max - sl_min) / 100.0, 0.1)

                # 숫자입력 초기값 세팅
                if f"ni_a_{var}" not in st.session_state:
                    st.session_state[f"ni_a_{var}"] = curr_clamped

                sl_col, ni_col = st.columns([3, 1])
                with sl_col:
                    st.session_state['current_inputs'][var] = st.slider(
                        f"{var}",
                        sl_min, sl_max, curr_clamped,
                        step=step_v,
                        key=f"sl_{var}_{st.session_state['ver']}",
                        on_change=_on_sl_a_change,
                        args=(var, st.session_state['ver'])
                    )
                with ni_col:
                    ni_val = st.number_input(
                        "Value",
                        min_value=sl_min, max_value=sl_max,
                        value=float(st.session_state['current_inputs'].get(var, curr_clamped)),
                        step=step_v,
                        format="%.2f",
                        key=f"ni_a_{var}",
                        on_change=_on_ni_a_change,
                        args=(var, sl_min, sl_max, st.session_state['ver']),
                        label_visibility="visible"
                    )
                    st.session_state['current_inputs'][var] = ni_val

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
                    opt_prog_text.markdown(
                        f"<span style='color:#e1e1e1; font-size:0.85rem;'>▸ <b>{L['opt_progress']} ({i+1}/{len(algorithms)}): {algo} ({pct}%)</b></span>",
                        unsafe_allow_html=True
                    )
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
                opt_prog_text.markdown(
                    f"<span style='color:#e1e1e1; font-size:0.85rem;'>▸ <b>{L['opt_progress']}: Hybrid Multi-Start (L-BFGS-B)</b></span>",
                    unsafe_allow_html=True
                )
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
                    <h4 style='margin-top:0; color:#cbd5e1;'>{L['dash_title']}</h4>
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

                    # [추가] 신뢰성 보강: 이 타겟 모델의 교차검증 신뢰도 · 표본 수를 함께 표시
                    rel = st.session_state.get('model_reliability', {}).get(target_key)
                    reliability_caption = ""
                    if rel is not None:
                        n_total = rel['n_total']
                        algo_name = rel.get('algo', 'LR')   # [추가] 알고리즘 이름
                        if rel['cv_score'] is not None:
                            rel_pct = int(round(rel['cv_score'] * 100))
                            rel_color = "#00e5ff" if rel_pct >= 80 else "#ffab00" if rel_pct >= 60 else "#ff5252"
                            reliability_caption = (
                                f"<span style='color:{rel_color};'>{L['reliability_label']}: {rel_pct}%</span>"
                                f" · {n_total}{L['reliability_samples']}"
                                f" · <span style='color:#a3e635; font-size:0.68rem;'>[{algo_name}]</span>"
                            )
                        else:
                            reliability_caption = (
                                f"<span style='color:#ff5252;'>{L['reliability_na']}</span>"
                                f" · {n_total}{L['reliability_samples']}"
                                f" · <span style='color:#a3e635; font-size:0.68rem;'>[{algo_name}]</span>"
                            )
                        if rel['low_sample']:
                            reliability_caption += f" · <span style='color:#ff5252;'>⚠ {L['reliability_low_sample']}</span>"

                    st.markdown(
                        f"""<div style="margin-bottom: 12px; {opacity_style}">
                            <span style="font-size:0.95rem; font-weight:600; color:#ffffff;">{full_name}</span>
                            <div class="custom-progress-container">
                                <div class="custom-progress-bar"
                                    style="width: {r_perc}%; background: {bar_color};">
                                    {r_perc}%
                                </div>
                            </div>
                            <div style="font-size:0.72rem; color:#cbd5e1; margin-top:2px;">{reliability_caption}</div>
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

        # ── 공정 개선 가이드 + AI 전문가 진단 통합 expander ─────────
        st.divider()
        _combined_lbl = (
            "+ Feature Importance-based Diagnosis & AI Expert Report"
            if st.session_state.lang == "en"
            else "+ Feature Importance 기반 공정 진단 가이드 & AI 전문가 진단"
        )
        with st.expander(_combined_lbl, expanded=False):

            # ── 1) Feature Importance 기반 공정 진단 가이드 ──────────
            if st.session_state.get('last_res_val') is not None:
                risks = st.session_state['last_defect_risks']
                normal_count   = sum(1 for r in risks.values() if r < DEFECT_THRESHOLD)
                out_spec_count = len(risks) - normal_count
                success_status = L['guide_all_success'] if out_spec_count == 0 else (
                    f"Caution Required ({out_spec_count} out of spec)" if st.session_state.lang == "en"
                    else f"주의 필요 ({out_spec_count}개 이탈)"
                )
                success_msg = L['guide_success_msg'] if out_spec_count == 0 else L['guide_partial_msg']
                icon = "✓" if out_spec_count == 0 else "⚠"
                unachievable_label = "Unachievable" if st.session_state.lang == "en" else "달성 불가"
                out_spec_label     = "Out of Spec"  if st.session_state.lang == "en" else "이탈"
                normal_label       = "Normal"       if st.session_state.lang == "en" else "정상"
                guide_html = f"""
                <div style="background-color:#12141d; border:1px solid #2d3142;
                            border-radius:10px; padding:16px 20px; margin-bottom:16px;">
                    <div style="font-size:0.82rem; color:#00e5ff; font-weight:700;
                                margin-bottom:10px;">▸ {L['guide_subtitle']}</div>
                    <div style="border-left:4px solid #10b981; padding:8px 12px;
                                background:#1a1c24; border-radius:4px;
                                font-size:0.82rem; color:#cbd5e1; margin-bottom:10px;">
                        {L['guide_pred_rel']}: <b>100.0%</b> &nbsp;|&nbsp;
                        {unachievable_label}: <b>0</b> &nbsp;|&nbsp;
                        {out_spec_label}: <b>{out_spec_count}</b> &nbsp;|&nbsp;
                        {normal_label}: <b>{normal_count}</b>
                    </div>
                    <div style="color:#10b981; font-size:0.88rem; font-weight:700;
                                margin-bottom:6px;">{icon} {success_status}</div>
                    <div style="font-size:0.85rem; color:#e1e1e1; line-height:1.6;">
                        {success_msg}
                    </div>
                </div>
                """
                st.markdown(guide_html, unsafe_allow_html=True)
            else:
                st.info("진단 또는 최적화를 먼저 실행해 주세요." if st.session_state.lang != "en"
                        else "Please run Diagnose or Optimize first.")

            st.markdown("<hr style='border-color:#2d3142; margin:8px 0 16px 0;'>",
                        unsafe_allow_html=True)

            # ── 2) AI 전문가 진단 ────────────────────────────────────
            if st.session_state['last_res_val'] is None:
                st.warning(L['warn_need_diagnose'])
            else:
                is_optimized = st.session_state['last_opt_df'] is not None
                st.success(L['opt_success_msg'] if is_optimized else L['diag_success_msg'])
                if st.button(L['btn_ai_report'], key="btn_report_active"):
                    with st.spinner(L['spinner_analyzing']):
                        import re
                        no_diag_text = "No diagnosis" if st.session_state.lang == "en" else "진단 없음"
                        results = st.session_state.get('last_defect_risks', no_diag_text)
                        if is_optimized:
                            params = st.session_state['last_opt_df'].to_dict(orient='records')
                        else:
                            params = [{v: st.session_state['current_inputs'].get(v, 0)
                                       for v in st.session_state['ui_display_vars']}]
                        report = generate_ai_report(
                            results, params, num_actions=NUM_ACTIONS,
                            lang=st.session_state.lang, is_optimized=is_optimized
                        )
                        cleaned = re.sub(r'<br\s*/?>', '\n', report)
                        cleaned = re.sub(r'<[^>]+>', '', cleaned)
                        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
                        lines = cleaned.strip().split('\n')
                        html_lines = []
                        for line in lines:
                            line = line.strip()
                            if not line:
                                html_lines.append('<div style="margin:6px 0;"></div>')
                            elif re.match(r'^\d+[\.\.].', line):
                                html_lines.append(
                                    f'<div style="margin:10px 0 4px 0; color:#00e5ff;'
                                    f' font-weight:700; font-size:0.92rem;">{line}</div>'
                                )
                            else:
                                html_lines.append(
                                    f'<div style="margin:2px 0 2px 12px; color:#e1e1e1;'
                                    f' font-size:0.88rem; line-height:1.6;">{line}</div>'
                                )
                        report_html = f"""
                        <div style="background-color:#12141d; border:1px solid #2d3142;
                                    border-radius:10px; padding:20px 24px; margin-top:12px;
                                    max-height:420px; overflow-y:auto;">
                            <div style="color:#cbd5e1; font-size:0.8rem; margin-bottom:12px;
                                        letter-spacing:0.05em;">{L['report_box_title']}</div>
                            {''.join(html_lines)}
                        </div>"""
                        st.markdown(report_html, unsafe_allow_html=True)

        # [추가] Feature Importance 시각화 섹션 - expander로 접기/펼치기
        fi_all = st.session_state.get('feature_importance', {})
        if fi_all:
            import streamlit.components.v1 as components
            st.divider()
            fi_title = "+ Feature Importance — Top Influential Variables per Defect" if st.session_state.lang == "en" else "+ Feature Importance — 불량별 주요 영향 변수"
            with st.expander(fi_title, expanded=False):
                fi_sel_label  = "Select Defect" if st.session_state.lang == "en" else "불량 항목 선택"
                fi_algo_label = "Algorithm"     if st.session_state.lang == "en" else "알고리즘"
                fi_top_label  = "Top 15 Variables" if st.session_state.lang == "en" else "상위 15개 변수"
                fi_target_keys = list(fi_all.keys())
                fi_sel = st.selectbox(
                    fi_sel_label,
                    options=fi_target_keys,
                    format_func=lambda k: TARGET_VARS.get(k, k),
                    key="fi_target_sel"
                )
                if fi_sel and fi_sel in fi_all:
                    fi_data   = fi_all[fi_sel]
                    fi_series = pd.Series(fi_data).sort_values(ascending=False).head(15)
                    algo_used = st.session_state.get('model_algo_names', {}).get(fi_sel, '')
                    max_val   = fi_series.max() if fi_series.max() > 0 else 1.0

                    bar_rows_html = ""
                    for var_name, imp_val in fi_series.items():
                        bar_pct   = imp_val / max_val * 100
                        bar_color = "#00e5ff" if bar_pct >= 60 else "#10b981" if bar_pct >= 30 else "#94a3b8"
                        bar_rows_html += f"""
                        <div style="margin-bottom:8px;">
                          <div style="display:flex;justify-content:space-between;font-size:13px;color:#e1e1e1;margin-bottom:3px;">
                            <span style="font-weight:600;">{var_name}</span>
                            <span style="color:#cbd5e1;">{imp_val:.4f}</span>
                          </div>
                          <div style="background:#1e293b;border-radius:3px;height:10px;">
                            <div style="width:{bar_pct:.1f}%;background:{bar_color};height:10px;border-radius:3px;"></div>
                          </div>
                        </div>"""

                    fi_html = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;background:#12141d;font-family:Inter,sans-serif;">
                    <div style="background:#12141d;border:1px solid #2d3142;border-radius:10px;padding:20px 24px;">
                      <div style="color:#cbd5e1;font-size:12px;margin-bottom:14px;">
                        {TARGET_VARS.get(fi_sel, fi_sel)} &middot; {fi_algo_label}: <span style="color:#a3e635;">{algo_used}</span> &middot; {fi_top_label}
                      </div>
                      {bar_rows_html}
                    </div></body></html>"""
                    components.html(fi_html, height=60 + len(fi_series) * 36, scrolling=False)

    with t2:
        if not st.session_state['df_injection'].empty:
            import streamlit.components.v1 as components
            df_view = st.session_state['df_injection'].copy()
            is_en   = st.session_state.lang == "en"

            # ── 상단 데이터 & 모델 현황 요약 카드 ───────────────────────
            target_cols_all  = [c for c in df_view.columns if c in TARGET_VARS]
            process_cols_all = [c for c in df_view.columns if c not in TARGET_VARS and c != 'vars']
            n_rows           = len(df_view)
            n_proc           = len(process_cols_all)
            n_tgt            = len(target_cols_all)

            # 모델 신뢰도 평균
            rel_dict  = st.session_state.get('model_reliability', {})
            cv_scores = [v['cv_score'] for v in rel_dict.values() if v.get('cv_score') is not None]
            avg_cv    = f"{np.mean(cv_scores)*100:.1f}%" if cv_scores else "N/A"

            # 고위험 불량 수 (마지막 진단 결과 기준)
            last_risks   = st.session_state.get('last_defect_risks', {})
            high_risk_n  = sum(1 for v in last_risks.values() if v >= DEFECT_THRESHOLD)
            high_risk_names = [TARGET_VARS.get(k, k) for k, v in last_risks.items() if v >= DEFECT_THRESHOLD]

            # 알고리즘별 선택 횟수
            algo_summary = st.session_state.get('algo_summary', {})
            from collections import Counter
            algo_counts  = Counter(info['algo'] for info in algo_summary.values())
            algo_dist_str = " / ".join([f"{k}: {v}" for k, v in algo_counts.most_common()])

            if is_en:
                card_items = [
                    ("Total Samples",          f"{n_rows} rows"),
                    ("Process Variables",       f"{n_proc} vars"),
                    ("Defect Targets",          f"{n_tgt} types"),
                    ("Avg. Model CV Accuracy",  avg_cv),
                    ("High-Risk Defects",       f"{high_risk_n} types" if last_risks else "Not diagnosed yet"),
                    ("Algorithm Distribution",  algo_dist_str if algo_dist_str else "Not trained yet"),
                ]
                summary_title = "Dataset & Model Overview"
                insight_title = "Key Insights"

                # 인사이트 메시지 생성
                insights = []
                if cv_scores:
                    best_tgt = max(rel_dict, key=lambda k: rel_dict[k].get('cv_score') or 0)
                    worst_tgt = min(rel_dict, key=lambda k: rel_dict[k].get('cv_score') or 1)
                    insights.append(f"Best model accuracy: <b style='color:#a3e635;'>{TARGET_VARS.get(best_tgt, best_tgt)}</b> ({rel_dict[best_tgt].get('cv_score',0)*100:.1f}%)")
                    worst_cv = rel_dict[worst_tgt].get('cv_score') or 0
                    if worst_cv < 0.7:
                        insights.append(f"Low accuracy warning: <b style='color:#ff5252;'>{TARGET_VARS.get(worst_tgt, worst_tgt)}</b> ({worst_cv*100:.1f}%) — consider collecting more data.")
                if high_risk_n > 0:
                    insights.append(f"<b style='color:#ffab00;'>{high_risk_n} high-risk defect(s)</b> detected in last diagnosis: {', '.join(h.split('(')[0].strip() for h in high_risk_names[:3])}")
                else:
                    if last_risks:
                        insights.append("<b style='color:#10b981;'>All defects below risk threshold</b> in last diagnosis.")
                if algo_counts:
                    top_algo = algo_counts.most_common(1)[0][0]
                    insights.append(f"Most selected algorithm: <b style='color:#00e5ff;'>{top_algo}</b> ({algo_counts[top_algo]}/{n_tgt} defects)")
                low_sample_targets = [k for k, v in rel_dict.items() if v.get('low_sample')]
                if low_sample_targets:
                    insights.append(f"<b style='color:#ff5252;'>Low sample warning</b> on: {', '.join(low_sample_targets)} — predictions may be unreliable.")
            else:
                card_items = [
                    ("전체 샘플 수",         f"{n_rows} 행"),
                    ("공정 변수 수",         f"{n_proc} 개"),
                    ("불량 타겟 수",         f"{n_tgt} 종"),
                    ("평균 모델 CV 정확도",   avg_cv),
                    ("고위험 불량",          f"{high_risk_n} 종" if last_risks else "아직 진단 안 함"),
                    ("알고리즘 분포",         algo_dist_str if algo_dist_str else "미학습"),
                ]
                summary_title = "데이터셋 & 모델 현황 요약"
                insight_title = "주요 인사이트"

                insights = []
                if cv_scores:
                    best_tgt = max(rel_dict, key=lambda k: rel_dict[k].get('cv_score') or 0)
                    worst_tgt = min(rel_dict, key=lambda k: rel_dict[k].get('cv_score') or 1)
                    insights.append(f"가장 높은 모델 정확도: <b style='color:#a3e635;'>{TARGET_VARS.get(best_tgt, best_tgt)}</b> ({rel_dict[best_tgt].get('cv_score',0)*100:.1f}%)")
                    worst_cv = rel_dict[worst_tgt].get('cv_score') or 0
                    if worst_cv < 0.7:
                        insights.append(f"정확도 낮음 주의: <b style='color:#ff5252;'>{TARGET_VARS.get(worst_tgt, worst_tgt)}</b> ({worst_cv*100:.1f}%) — 데이터 추가 수집 권장")
                if high_risk_n > 0:
                    insights.append(f"마지막 진단에서 <b style='color:#ffab00;'>고위험 불량 {high_risk_n}종</b> 감지: {', '.join(h.split('(')[0].strip() for h in high_risk_names[:3])}")
                else:
                    if last_risks:
                        insights.append("<b style='color:#10b981;'>마지막 진단에서 모든 불량이 기준치 이하</b>로 확인됨.")
                if algo_counts:
                    top_algo = algo_counts.most_common(1)[0][0]
                    insights.append(f"가장 많이 선택된 알고리즘: <b style='color:#00e5ff;'>{top_algo}</b> ({algo_counts[top_algo]}/{n_tgt}종 불량)")
                low_sample_targets = [k for k, v in rel_dict.items() if v.get('low_sample')]
                if low_sample_targets:
                    insights.append(f"<b style='color:#ff5252;'>표본 부족 경고</b>: {', '.join(low_sample_targets)} — 예측 신뢰도 낮을 수 있음")

            # 요약 카드 렌더링
            card_html = "".join([
                f"<div style='text-align:center;padding:10px 6px;background:#1a1c24;"
                f"border:1px solid #2d3142;border-radius:8px;'>"
                f"<div style='font-size:0.68rem;color:#cbd5e1;margin-bottom:4px;'>{label}</div>"
                f"<div style='font-size:0.95rem;font-weight:700;color:#ffffff;'>{val}</div>"
                f"</div>"
                for label, val in card_items
            ])
            st.markdown(
                f"<div style='background:#12141d;border:1px solid #2d3142;border-radius:10px;"
                f"padding:16px 20px;margin-bottom:10px;'>"
                f"<div style='font-size:0.82rem;color:#00e5ff;font-weight:700;margin-bottom:12px;'>"
                f"▸ {summary_title}</div>"
                f"<div style='display:grid;grid-template-columns:repeat(6,1fr);gap:8px;'>{card_html}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

            # 인사이트 박스
            if insights:
                insight_rows = "".join([
                    f"<div style='padding:5px 0;border-bottom:1px solid #23263a;"
                    f"font-size:0.82rem;color:#e1e1e1;line-height:1.5;'>• {msg}</div>"
                    for msg in insights
                ])
                st.markdown(
                    f"<div style='background:#12141d;border:1px solid #2d3142;"
                    f"border-left:3px solid #00e5ff;border-radius:10px;"
                    f"padding:14px 20px;margin-bottom:12px;'>"
                    f"<div style='font-size:0.82rem;color:#00e5ff;font-weight:700;margin-bottom:8px;'>"
                    f"▸ {insight_title}</div>"
                    f"{insight_rows}</div>",
                    unsafe_allow_html=True
                )

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            # ── Raw Data ── expander ───────────────────────────────────
            raw_label = "+ Raw Data" if is_en else "+ 원시 데이터"
            with st.expander(raw_label, expanded=False):
                if is_en:
                    st.markdown(
                        "<p style='color:#cbd5e1;font-size:0.83rem;'>"
                        "All accumulated data including uploaded files and diagnosis/optimization history. "
                        "Defect columns represent predicted probability (0~1).</p>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        "<p style='color:#cbd5e1;font-size:0.83rem;'>"
                        "업로드된 파일과 진단/최적화 이력이 누적된 전체 데이터입니다. "
                        "불량 컬럼은 예측 확률(0~1)을 나타냅니다.</p>",
                        unsafe_allow_html=True
                    )
                st.dataframe(df_view, use_container_width=True)

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            # ── Defect Distribution ── expander ───────────────────────
            dist_label = "+ Defect Probability Distribution" if is_en else "+ 불량률 분포 히스토그램"
            with st.expander(dist_label, expanded=False):
                target_cols = [c for c in df_view.columns if c in TARGET_VARS]
                if target_cols:
                    sel_label = "Select Defect" if is_en else "불량 항목 선택"
                    sel_hist = st.selectbox(
                        sel_label,
                        options=target_cols,
                        format_func=lambda k: TARGET_VARS.get(k, k),
                        key="hist_target_sel"
                    )
                    hist_data = df_view[sel_hist].dropna()
                    if not hist_data.empty:
                        # 결과 분석 인사이트 계산
                        h_mean   = hist_data.mean()
                        h_std    = hist_data.std()
                        h_high   = (hist_data >= DEFECT_THRESHOLD).sum()
                        h_pct    = h_high / len(hist_data) * 100

                        if is_en:
                            dist_insight = (
                                f"Mean risk: <b style='color:#00e5ff;'>{h_mean:.3f}</b> &nbsp;|&nbsp; "
                                f"Std dev: <b>{h_std:.3f}</b> &nbsp;|&nbsp; "
                                f"High-risk samples (≥{int(DEFECT_THRESHOLD*100)}%): "
                                f"<b style='color:#{'ff5252' if h_pct>30 else 'ffab00' if h_pct>10 else '10b981'};'>"
                                f"{h_high} ({h_pct:.1f}%)</b>"
                            )
                            if h_pct > 30:
                                dist_advice = "High proportion of danger-zone samples. Priority inspection recommended."
                            elif h_pct > 10:
                                dist_advice = "Some samples in the caution zone. Monitor this defect type closely."
                            else:
                                dist_advice = "Most samples are in the safe zone. Low defect risk overall."
                        else:
                            dist_insight = (
                                f"평균 리스크: <b style='color:#00e5ff;'>{h_mean:.3f}</b> &nbsp;|&nbsp; "
                                f"표준편차: <b>{h_std:.3f}</b> &nbsp;|&nbsp; "
                                f"고위험 샘플 (≥{int(DEFECT_THRESHOLD*100)}%): "
                                f"<b style='color:#{'ff5252' if h_pct>30 else 'ffab00' if h_pct>10 else '10b981'};'>"
                                f"{h_high}개 ({h_pct:.1f}%)</b>"
                            )
                            if h_pct > 30:
                                dist_advice = "위험 구간 샘플 비율이 높습니다. 해당 불량을 우선 점검하세요."
                            elif h_pct > 10:
                                dist_advice = "주의 구간 샘플이 일부 존재합니다. 이 불량 유형을 주의 깊게 모니터링하세요."
                            else:
                                dist_advice = "대부분의 샘플이 안전 구간에 있습니다. 전반적으로 낮은 불량 리스크입니다."

                        st.markdown(
                            f"<div style='background:#1a1c24;border:1px solid #2d3142;"
                            f"border-left:3px solid #00e5ff;border-radius:6px;"
                            f"padding:10px 14px;margin-bottom:10px;font-size:0.82rem;'>"
                            f"<span style='color:#e1e1e1;'>{dist_insight}</span><br>"
                            f"<span style='color:#cbd5e1;font-size:0.78rem;'>→ {dist_advice}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                        bins = min(20, max(5, len(hist_data) // 3))
                        counts, edges = np.histogram(hist_data.values.astype(float), bins=bins, range=(0.0, 1.0))
                        legend_txt = (
                            "<span style='color:#00e5ff;'>●</span> Blue: Safe(0~0.3) &nbsp; <span style='color:#ffab00;'>●</span> Orange: Caution(0.3~0.7) &nbsp; <span style='color:#ff5252;'>●</span> Red: Danger(0.7~1.0)"
                            if is_en else
                            "<span style='color:#00e5ff;'>●</span> 파란색: 안전(0~0.3) &nbsp; <span style='color:#ffab00;'>●</span> 주황색: 주의(0.3~0.7) &nbsp; <span style='color:#ff5252;'>●</span> 빨간색: 위험(0.7~1.0)"
                        )
                        dist_title = f"{TARGET_VARS.get(sel_hist, sel_hist)} {'Distribution' if is_en else '분포'} (n={len(hist_data)})"
                        max_count = int(counts.max()) if counts.max() > 0 else 1
                        bar_rows_h = ""
                        for i, cnt in enumerate(counts):
                            lo, hi  = float(edges[i]), float(edges[i+1])
                            mid     = (lo + hi) / 2.0
                            bp      = int(cnt) / max_count * 100
                            bc      = "#00e5ff" if mid < 0.3 else "#ffab00" if mid < 0.7 else "#ff5252"
                            bar_rows_h += f"""
                            <div style="display:flex;align-items:center;margin-bottom:5px;gap:8px;">
                              <span style="color:#cbd5e1;font-size:11px;width:80px;text-align:right;">{lo:.2f}~{hi:.2f}</span>
                              <div style="flex:1;background:#1e293b;border-radius:3px;height:18px;">
                                <div style="width:{bp:.1f}%;background:{bc};height:18px;border-radius:3px;display:flex;align-items:center;padding-left:6px;">
                                  <span style="color:#fff;font-size:11px;">{int(cnt)}</span>
                                </div>
                              </div>
                            </div>"""
                        hist_html = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;background:#12141d;font-family:Inter,sans-serif;">
                        <div style="background:#12141d;border:1px solid #2d3142;border-radius:10px;padding:20px 24px;">
                          <div style="color:#e1e1e1;font-size:14px;font-weight:600;margin-bottom:14px;">{dist_title}</div>
                          {bar_rows_h}
                          <div style="color:#cbd5e1;font-size:11px;margin-top:10px;">{legend_txt}</div>
                        </div></body></html>"""
                        components.html(hist_html, height=80 + len(counts) * 28, scrolling=False)

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            # ── Correlation Heatmap ── expander ───────────────────────
            corr_label = "+ Process Variable Correlation Heatmap" if is_en else "+ 변수 상관관계 히트맵"
            with st.expander(corr_label, expanded=False):
                numeric_cols = df_view.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    process_vars = [c for c in numeric_cols if c not in TARGET_VARS]
                    defect_vars  = [c for c in numeric_cols if c in TARGET_VARS]

                    if process_vars and defect_vars:
                        corr_df  = df_view[process_vars + defect_vars].corr()
                        sub_corr = corr_df.loc[process_vars, defect_vars]
                        top_vars = sub_corr.abs().max(axis=1).sort_values(ascending=False).head(10).index.tolist()
                        sub_corr = sub_corr.loc[top_vars]

                        # 상관관계 인사이트 자동 추출
                        flat = sub_corr.abs().stack()
                        top_pair = flat.idxmax()
                        top_val  = sub_corr.loc[top_pair[0], top_pair[1]]
                        top_dir  = ("positively" if top_val > 0 else "negatively") if is_en else ("양의 방향으로" if top_val > 0 else "음의 방향으로")
                        strong_pairs = [(r, c, sub_corr.loc[r, c]) for r in sub_corr.index for c in sub_corr.columns if abs(sub_corr.loc[r, c]) >= 0.5]
                        strong_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

                        if is_en:
                            corr_insight = (
                                f"Strongest correlation: <b style='color:#00e5ff;'>{top_pair[0]}</b> ↔ "
                                f"<b style='color:#00e5ff;'>{top_pair[1]}</b> "
                                f"({top_dir}, r={top_val:.2f})"
                            )
                            if strong_pairs:
                                corr_advice = f"{len(strong_pairs)} variable-defect pair(s) with |r| ≥ 0.5 — these variables are strong predictors and key targets for process control."
                            else:
                                corr_advice = "No strong linear correlations (|r| ≥ 0.5) found. Nonlinear interactions may be dominant — tree-based models handle this well."
                        else:
                            corr_insight = (
                                f"가장 강한 상관관계: <b style='color:#00e5ff;'>{top_pair[0]}</b> ↔ "
                                f"<b style='color:#00e5ff;'>{top_pair[1]}</b> "
                                f"({top_dir}, r={top_val:.2f})"
                            )
                            if strong_pairs:
                                corr_advice = f"|r| ≥ 0.5 이상의 강한 상관 변수-불량 쌍 {len(strong_pairs)}개 — 공정 관리의 핵심 타겟 변수입니다."
                            else:
                                corr_advice = "강한 선형 상관(|r| ≥ 0.5)이 없습니다. 비선형 상호작용이 지배적일 수 있습니다 — 트리 기반 모델이 유리합니다."

                        st.markdown(
                            f"<div style='background:#1a1c24;border:1px solid #2d3142;"
                            f"border-left:3px solid #00e5ff;border-radius:6px;"
                            f"padding:10px 14px;margin-bottom:10px;font-size:0.82rem;'>"
                            f"<span style='color:#e1e1e1;'>{corr_insight}</span><br>"
                            f"<span style='color:#cbd5e1;font-size:0.78rem;'>→ {corr_advice}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                        corr_title = "Process Variables ↔ Defect Correlation (Top 10)" if is_en else "공정 변수 ↔ 불량 항목 상관계수 (상위 10개 공정 변수)"
                        corr_leg   = "● Blue: Positive &nbsp; ● Red: Negative &nbsp; ● Gray: Weak" if is_en else "● 파란색: 양의 상관 &nbsp; ● 빨간색: 음의 상관 &nbsp; ● 회색: 상관 약함"

                        header_cells = "".join([f"<th style='padding:6px 8px;font-size:11px;color:#cbd5e1;text-align:center;white-space:nowrap;'>{c}</th>" for c in sub_corr.columns])
                        data_rows = ""
                        for var_r in sub_corr.index:
                            cells = f"<td style='padding:6px 8px;font-size:12px;color:#e1e1e1;font-weight:600;'>{var_r}</td>"
                            for var_c in sub_corr.columns:
                                val = float(sub_corr.loc[var_r, var_c])
                                intensity = abs(val)
                                if val > 0.3:
                                    bg = f"rgba(0,229,255,{min(intensity,0.9):.2f})"
                                    tc = "#000"
                                elif val < -0.3:
                                    bg = f"rgba(255,82,82,{min(intensity,0.9):.2f})"
                                    tc = "#fff"
                                else:
                                    bg = "#1e293b"
                                    tc = "#94a3b8"
                                cells += f"<td style='padding:6px 8px;text-align:center;background:{bg};color:{tc};font-size:12px;'>{val:.2f}</td>"
                            data_rows += f"<tr>{cells}</tr>"

                        st.markdown(
                            f"""<div style="background:#12141d;border:1px solid #2d3142;border-radius:10px;padding:20px 24px;margin-top:8px;overflow-x:auto;">
                                <div style="color:#e1e1e1;font-size:0.9rem;font-weight:600;margin-bottom:14px;">{corr_title}</div>
                                <table style="border-collapse:collapse;width:100%;">
                                    <thead><tr><th style="padding:6px 8px;"></th>{header_cells}</tr></thead>
                                    <tbody>{data_rows}</tbody>
                                </table>
                                <div style="color:#cbd5e1;font-size:0.72rem;margin-top:10px;">{corr_leg}</div>
                            </div>""",
                            unsafe_allow_html=True
                        )


            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            # ── 변수 민감도 분석 expander ─────────────────────────────
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            sens_label = "+ Variable Sensitivity Analysis" if is_en else "+ 변수 민감도 분석"
            with st.expander(sens_label, expanded=False):
                fi_all_t2 = st.session_state.get('feature_importance', {})
                models_t2  = st.session_state.get('models', {})
                scalers_t2 = st.session_state.get('scalers', {})
                proc_vars  = st.session_state.get('global_process_vars', [])
                cur_inputs = st.session_state.get('current_inputs', {})

                if not fi_all_t2 or not models_t2 or not proc_vars:
                    st.info("AI 모델 학습 후 진단/최적화를 먼저 실행해 주세요."
                            if not is_en else "Please run AI learning and diagnosis first.")
                else:
                    sens_tgt_lbl = "불량 항목 선택" if not is_en else "Select Defect Target"
                    sens_var_lbl = "분석 변수 선택" if not is_en else "Select Variable"
                    sens_tgt = st.selectbox(
                        sens_tgt_lbl,
                        options=list(models_t2.keys()),
                        format_func=lambda k: TARGET_VARS.get(k, k),
                        key="sens_tgt_sel"
                    )
                    sens_var = st.selectbox(
                        sens_var_lbl,
                        options=proc_vars,
                        key="sens_var_sel"
                    )

                    if sens_tgt and sens_var and sens_tgt in models_t2:
                        _model  = models_t2[sens_tgt]
                        _scaler = scalers_t2[sens_tgt]
                        _bounds = st.session_state.get('global_bounds', {})
                        v_min, v_max = _bounds.get(sens_var, (0, 100))
                        v_steps = np.linspace(v_min, v_max, 30)

                        # 현재 입력값 기준, 선택 변수만 변화시켜 리스크 계산
                        base_vals = [float(cur_inputs.get(v, 0)) for v in proc_vars]
                        var_idx   = proc_vars.index(sens_var)

                        sens_risks = []
                        for sv in v_steps:
                            trial = base_vals.copy()
                            trial[var_idx] = float(sv)
                            df_trial = pd.DataFrame([trial], columns=proc_vars)
                            prob = _model.predict_proba(_scaler.transform(df_trial))[0, 1]
                            sens_risks.append(prob)

                        # 민감도 지표: 변화 범위 대비 리스크 변화폭
                        risk_range = max(sens_risks) - min(sens_risks)
                        cur_risk   = sens_risks[int(len(sens_risks) * (
                            (cur_inputs.get(sens_var, v_min) - v_min) / max(v_max - v_min, 1e-9)
                        ))]

                        if is_en:
                            sens_insight = (
                                f"Variable: <b style='color:#00e5ff;'>{sens_var}</b> &nbsp;|&nbsp; "
                                f"Risk range: <b style='color:#ffab00;'>{min(sens_risks)*100:.1f}% ~ {max(sens_risks)*100:.1f}%</b> &nbsp;|&nbsp; "
                                f"Sensitivity: <b style='color:#{'ff5252' if risk_range>0.3 else 'ffab00' if risk_range>0.1 else '10b981'};'>"
                                f"{'HIGH' if risk_range>0.3 else 'MED' if risk_range>0.1 else 'LOW'}</b>"
                            )
                            sens_advice = (
                                f"This variable has a {'large' if risk_range>0.3 else 'moderate' if risk_range>0.1 else 'small'} "
                                f"impact on {TARGET_VARS.get(sens_tgt, sens_tgt)}. "
                                f"{'Priority control recommended.' if risk_range>0.3 else 'Monitor carefully.' if risk_range>0.1 else 'Low influence on this defect.'}"
                            )
                        else:
                            sens_insight = (
                                f"변수: <b style='color:#00e5ff;'>{sens_var}</b> &nbsp;|&nbsp; "
                                f"리스크 범위: <b style='color:#ffab00;'>{min(sens_risks)*100:.1f}% ~ {max(sens_risks)*100:.1f}%</b> &nbsp;|&nbsp; "
                                f"민감도: <b style='color:#{'ff5252' if risk_range>0.3 else 'ffab00' if risk_range>0.1 else '10b981'};'>"
                                f"{'높음' if risk_range>0.3 else '보통' if risk_range>0.1 else '낮음'}</b>"
                            )
                            sens_advice = (
                                f"이 변수는 {TARGET_VARS.get(sens_tgt, sens_tgt)} 불량에 "
                                f"{'큰' if risk_range>0.3 else '보통의' if risk_range>0.1 else '작은'} 영향을 미칩니다. "
                                f"{'우선 관리 대상입니다.' if risk_range>0.3 else '지속 모니터링을 권장합니다.' if risk_range>0.1 else '이 불량에 대한 영향이 낮습니다.'}"
                            )

                        st.markdown(
                            f"<div style='background:#1a1c24;border:1px solid #2d3142;"
                            f"border-left:3px solid #00e5ff;border-radius:6px;"
                            f"padding:10px 14px;margin-bottom:12px;font-size:0.82rem;'>"
                            f"<span style='color:#e1e1e1;'>{sens_insight}</span><br>"
                            f"<span style='color:#cbd5e1;font-size:0.78rem;'>→ {sens_advice}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                        # SVG 차트 렌더링
                        W, H, PAD = 860, 200, 40
                        cur_v   = float(cur_inputs.get(sens_var, v_min))
                        cur_idx = int((cur_v - v_min) / max(v_max - v_min, 1e-9) * (len(v_steps) - 1))
                        cur_idx = max(0, min(cur_idx, len(v_steps) - 1))

                        pts = []
                        for xi, rv in enumerate(sens_risks):
                            x = PAD + (xi / max(len(sens_risks) - 1, 1)) * (W - 2 * PAD)
                            y = PAD + (1 - rv) * (H - 2 * PAD)
                            pts.append((x, y))

                        polyline = " ".join([f"{x:.1f},{y:.1f}" for x, y in pts])
                        cx, cy   = pts[cur_idx]
                        threshold_y = PAD + (1 - DEFECT_THRESHOLD) * (H - 2 * PAD)

                        # 안전/위험 구역 배경
                        zone_safe   = f"<rect x='{PAD}' y='{threshold_y:.1f}' width='{W-2*PAD}' height='{H-PAD-threshold_y:.1f}' fill='rgba(16,185,129,0.06)'/>"
                        zone_danger = f"<rect x='{PAD}' y='{PAD}' width='{W-2*PAD}' height='{threshold_y-PAD:.1f}' fill='rgba(255,82,82,0.06)'/>"

                        # x축 레이블 (5개)
                        x_labels = ""
                        for ti in range(5):
                            xi2 = int(ti * (len(v_steps) - 1) / 4)
                            xp  = PAD + (xi2 / max(len(v_steps) - 1, 1)) * (W - 2 * PAD)
                            xv  = v_steps[xi2]
                            x_labels += f"<text x='{xp:.1f}' y='{H-PAD+18}' text-anchor='middle' fill='#64748b' font-size='10'>{xv:.1f}</text>"

                        # y축 레이블
                        y_labels = ""
                        for yi_pct in [0, 25, 50, 75, 100]:
                            yp = PAD + (1 - yi_pct / 100) * (H - 2 * PAD)
                            y_labels += f"<text x='{PAD-6}' y='{yp+4:.1f}' text-anchor='end' fill='#64748b' font-size='10'>{yi_pct}%</text>"

                        chart_title = (
                            f"{sens_var} → {TARGET_VARS.get(sens_tgt,'').split('(')[0].strip()} Risk Sensitivity"
                            if is_en else
                            f"{sens_var} 변화에 따른 {TARGET_VARS.get(sens_tgt,'').split('(')[0].strip()} 불량 리스크"
                        )
                        x_axis_lbl  = sens_var
                        cur_lbl     = f"Current: {cur_v:.1f}" if is_en else f"현재값: {cur_v:.1f}"
                        thres_lbl   = f"Threshold {int(DEFECT_THRESHOLD*100)}%" if is_en else f"기준선 {int(DEFECT_THRESHOLD*100)}%"

                        svg = f"""
                        <svg viewBox='0 0 {W} {H+50}' style='width:100%;max-height:280px;background:#12141d;border-radius:10px;border:1px solid #2d3142;'>
                          {zone_safe}{zone_danger}
                          <line x1='{PAD}' y1='{PAD}' x2='{PAD}' y2='{H-PAD}' stroke='#2d3142' stroke-width='1'/>
                          <line x1='{PAD}' y1='{H-PAD}' x2='{W-PAD}' y2='{H-PAD}' stroke='#2d3142' stroke-width='1'/>
                          <line x1='{PAD}' y1='{threshold_y:.1f}' x2='{W-PAD}' y2='{threshold_y:.1f}'
                                stroke='#ffab00' stroke-width='1' stroke-dasharray='5,4' opacity='0.7'/>
                          <text x='{W-PAD+4}' y='{threshold_y+4:.1f}' fill='#ffab00' font-size='10'>{thres_lbl}</text>
                          <polyline points='{polyline}' fill='none' stroke='#00e5ff' stroke-width='2.5'/>
                          <circle cx='{cx:.1f}' cy='{cy:.1f}' r='6' fill='#ff5252' stroke='#fff' stroke-width='1.5'/>
                          <text x='{min(cx+10, W-100):.1f}' y='{max(cy-10, PAD+12):.1f}' fill='#ff5252' font-size='11' font-weight='bold'>{cur_lbl}</text>
                          {x_labels}{y_labels}
                          <text x='{W//2}' y='{H+46}' text-anchor='middle' fill='#94a3b8' font-size='11'>{x_axis_lbl}</text>
                          <text x='{W//2}' y='18' text-anchor='middle' fill='#e1e1e1' font-size='12' font-weight='bold'>{chart_title}</text>
                        </svg>"""
                        st.markdown(svg, unsafe_allow_html=True)

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            # ── Trend Chart ── expander ───────────────────────────────
            trend_label = "+ Time-Series Trend Chart" if is_en else "+ 시계열 트렌드 차트"
            with st.expander(trend_label, expanded=False):
                target_cols_t  = [c for c in df_view.columns if c in TARGET_VARS]
                process_cols_t = [c for c in df_view.select_dtypes(include=[np.number]).columns if c not in TARGET_VARS]

                if target_cols_t or process_cols_t:
                    # 트렌드 인사이트: 불량 리스크 증가/감소 추세 자동 계산
                    trend_insights = []
                    for tc in target_cols_t:
                        tc_data = df_view[tc].dropna()
                        if len(tc_data) >= 3:
                            first_half = tc_data.iloc[:len(tc_data)//2].mean()
                            last_half  = tc_data.iloc[len(tc_data)//2:].mean()
                            delta      = last_half - first_half
                            if abs(delta) >= 0.05:
                                direction = ("↑ increasing" if delta > 0 else "↓ decreasing") if is_en else ("↑ 상승 추세" if delta > 0 else "↓ 하락 추세")
                                color     = "#ff5252" if delta > 0 else "#10b981"
                                trend_insights.append(
                                    f"<b style='color:{color};'>{tc}</b>: {direction} (Δ{delta:+.3f})"
                                )

                    if trend_insights:
                        title_txt = "Defect Risk Trend (first half vs. last half of data)" if is_en else "불량 리스크 추세 (데이터 전반부 vs 후반부 평균 비교)"
                        st.markdown(
                            f"<div style='background:#1a1c24;border:1px solid #2d3142;"
                            f"border-left:3px solid #00e5ff;border-radius:6px;"
                            f"padding:10px 14px;margin-bottom:10px;'>"
                            f"<div style='font-size:0.78rem;color:#00e5ff;font-weight:600;margin-bottom:6px;'>"
                            f"▸ {title_txt}</div>"
                            + "".join([f"<div style='font-size:0.80rem;color:#e1e1e1;padding:2px 0;'>{t}</div>" for t in trend_insights])
                            + "</div>",
                            unsafe_allow_html=True
                        )

                    trend_sel_l = "Select items to view trend (multiple selection)" if is_en else "트렌드 확인할 항목 선택 (복수 선택 가능)"
                    trend_chart_title = "Trend Chart (Normalized 0~1)" if is_en else "트렌드 차트 (정규화 0~1 표시)"
                    all_trend_cols = target_cols_t + process_cols_t
                    trend_sel = st.multiselect(
                        trend_sel_l,
                        options=all_trend_cols,
                        default=target_cols_t[:3] if len(target_cols_t) >= 3 else target_cols_t,
                        format_func=lambda k: TARGET_VARS.get(k, k),
                        key="trend_multisel"
                    )
                    if trend_sel:
                        trend_df = df_view[trend_sel].reset_index(drop=True)
                        trend_df.index = range(1, len(trend_df) + 1)

                        COLORS = ["#00e5ff","#a3e635","#ffab00","#ff5252","#c084fc",
                                  "#fb923c","#34d399","#f472b6","#60a5fa","#fbbf24"]

                        line_defs = []
                        for ci, col in enumerate(trend_sel):
                            col_data = trend_df[col].dropna()
                            if col_data.empty:
                                continue
                            col_min, col_max = col_data.min(), col_data.max()
                            span = col_max - col_min if col_max != col_min else 1.0
                            norm = (col_data - col_min) / span
                            line_defs.append({'col': col, 'data': norm, 'raw': col_data, 'color': COLORS[ci % len(COLORS)]})

                        if line_defs:
                            W, H, PAD = 900, 260, 40
                            svg_lines = ""
                            for ld in line_defs:
                                pts = []
                                for xi, (idx_v, y_v) in enumerate(ld['data'].items()):
                                    x = PAD + (xi / max(len(ld['data']) - 1, 1)) * (W - 2 * PAD)
                                    y = PAD + (1 - float(y_v)) * (H - 2 * PAD)
                                    pts.append(f"{x:.1f},{y:.1f}")
                                if pts:
                                    svg_lines += f"<polyline points='{' '.join(pts)}' fill='none' stroke='{ld['color']}' stroke-width='2' opacity='0.85'/>"
                                    last_x, last_y = float(pts[-1].split(',')[0]), float(pts[-1].split(',')[1])
                                    last_raw = ld['raw'].iloc[-1]
                                    svg_lines += f"<circle cx='{last_x}' cy='{last_y}' r='4' fill='{ld['color']}'/>"
                                    svg_lines += f"<text x='{min(last_x+6, W-60)}' y='{last_y+4}' fill='{ld['color']}' font-size='10'>{float(last_raw):.2f}</text>"

                            legend = ""
                            for li, ld in enumerate(line_defs):
                                lx = PAD + li * 130
                                legend += f"<rect x='{lx}' y='{H+8}' width='14' height='8' fill='{ld['color']}' rx='2'/>"
                                label  = TARGET_VARS.get(ld['col'], ld['col'])[:12]
                                legend += f"<text x='{lx+18}' y='{H+16}' fill='#cbd5e1' font-size='10'>{label}</text>"

                            svg_h = H + 36
                            st.markdown(
                                f"""<div style="background:#12141d;border:1px solid #2d3142;border-radius:10px;padding:16px 20px;margin-top:8px;overflow-x:auto;">
                                    <div style="color:#e1e1e1;font-size:0.9rem;font-weight:600;margin-bottom:10px;">{trend_chart_title}</div>
                                    <svg viewBox='0 0 {W} {svg_h}' style='width:100%;max-height:300px;'>
                                        <line x1='{PAD}' y1='{PAD}' x2='{PAD}' y2='{H-PAD}' stroke='#2d3142' stroke-width='1'/>
                                        <line x1='{PAD}' y1='{H-PAD}' x2='{W-PAD}' y2='{H-PAD}' stroke='#2d3142' stroke-width='1'/>
                                        {svg_lines}
                                        {legend}
                                    </svg>
                                </div>""",
                                unsafe_allow_html=True
                            )


            # ── 서브탭 1~4: expander로 이미 위에서 처리됨 (잔여 코드 제거)
