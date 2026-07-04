import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
import sqlite3
import json
import os
from datetime import datetime

# --- i18n Language Dictionary Definition ---
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
        "btn_download": "Download Optimal Parameters (.csv)",
        "db_save_empty": "No data available to save. Please run AI Learning first.",
        "db_pc_download": "📥 Download Saved DB File to PC Directly",
        "db_export_title": "💾 External Database Export",
        "db_prepare_btn": "⚙️ Generate & Save DB Snapshot",
        "db_prepared_msg": "Prepared File: ",
        "db_current_latest": "✨ The file contains the latest data state."
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
        "run_ai": "AI 가동 및 솔루션 탐색",
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
        "btn_optimize": "조건 최적화",
        "opt_converged": "수렴 완료",
        "opt_failed": "최적화 실패",
        "dash_title": "AI 지능형 대시보드",
        "opt_success_msg": "AI 추천 조건 도출 완료",
        "btn_download": "최적 파라미터 다운로드 (.csv)",
        "db_save_empty": "저장할 데이터가 없습니다. 먼저 데이터 업로드 후 AI 가동을 완료해 주세요.",
        "db_pc_download": "📥 내보낸 DB 파일 PC로 직접 다운로드",
        "db_export_title": "💾 데이터베이스 외부 내보내기",
        "db_prepare_btn": "⚙️ DB 스냅샷 생성 및 서버 저장",
        "db_prepared_msg": "준비된 파일: ",
        "db_current_latest": "✨ 최신 데이터 상태가 파일에 이미 반영되어 있습니다."
    }
}

# 0. High-Performance UI Theme & Custom Styling
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
            color: #FFFFFF !important; font-weight: 700 !important; border: 1px solid #1976d2 !important;
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

# Define 10 Injection Defect Master Variables
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

# Initialize Session State
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
        'data_changed_since_save': False
    })

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Noto+Sans+KR:wght@300;400;700&display=swap');
    .stApp { background-color: #0b0c10 !important; color: #e1e1e1 !important; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #12141d !important; border-right: 1px solid #1f222e; }
    [data-testid="stSidebar"] label { color: #FFFFFF !important; font-weight: 400 !important; }
    .stSlider label, .stNumberInput label, [data-testid="stWidgetLabel"] p { color: #FFFFFF !important; font-weight: 400 !important; font-size: 1.05rem !important; }
    .metric-container { background-color: #1a1c24; border: 1px solid #2d3142; border-radius: 10px; padding: 15px; text-align: center; }
    .metric-label { color: #94a3b8; font-size: 0.85rem; margin-bottom: 5px; }
    .metric-value { color: #ffffff; font-size: 1.2rem; font-weight: 700; }
    .section-title { display: flex; align-items: center; color: #00e5ff !important; font-weight: 600 !important; font-size: 1.3rem; margin-bottom: 1rem; margin-top: 1.5rem; }
    .square-icon { width: 18px; height: 18px; background-color: #00e5ff; margin-right: 14px; display: inline-block; flex-shrink: 0; }
    .optimized-table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 0.9rem; background-color: #1a1c24; border-radius: 8px; overflow: hidden; }
    .optimized-table th { background-color: #2d3142; color: #FFFFFF !important; font-weight: 700; padding: 12px; border: 1px solid #3f445e; }
    .optimized-table td { color: #FFFFFF !important; padding: 12px; text-align: center; border: 1px solid #3f445e; font-weight: 500; }
    .stButton>button { width: 100%; border-radius: 6px; background: linear-gradient(180deg, #1e88e5 0%, #1565c0 100%); color: white !important; font-weight: 700; border: 1px solid #1976d2; padding: 0.7rem; transition: all 0.3s ease; }
    .stDownloadButton>button { background: linear-gradient(180deg, #2e7d32 0%, #1b5e20 100%) !important; border: 1px solid #2e7d32 !important; }
    h1 { color: #ffffff !important; font-weight: 800 !important; letter-spacing: -0.04em; }
    .custom-progress-container { width: 100%; background-color: #1f222e; border-radius: 20px; margin: 10px 0; height: 22px; position: relative; overflow: hidden; border: 1px solid #2d3142; }
    .custom-progress-bar { height: 100%; border-radius: 20px; transition: width 0.8s ease-in-out; display: flex; align-items: center; justify-content: center; color: white; font-weight: 800; font-size: 0.85rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); min-width: 2.5rem; }
    div[data-testid="stCheckbox"] label p { color: #00e5ff !important; font-weight: 600 !important; }
    </style>
""", unsafe_allow_html=True)

# 1. Sidebar (Data Management)
with st.sidebar:
    st.markdown(f"<h2 style='color:#FFFFFF; font-size:1.5rem;'>{L['data_mgmt']}</h2>", unsafe_allow_html=True)
    
    with st.sidebar.form(key='data_upload_form'):
        u1 = st.file_uploader(L['upload_1'], type=['csv','xlsx'])
        u2 = st.file_uploader(L['upload_2'], type=['csv','xlsx','db'])
        u3 = st.file_uploader(L['upload_3'], type=['csv','xlsx'])
        sub_btn = st.form_submit_button(L['run_ai'])

    if sub_btn:
        def load_data(f):
            if not f: return None
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
            if df_v is not None:
                df_v = df_v.rename(columns=OLD_TO_NEW_MAP)
                df_v = df_v.loc[:, ~df_v.columns.duplicated()]
            if df_r is not None:
                df_r = df_r.rename(columns=OLD_TO_NEW_MAP)
                df_r = df_r.loc[:, ~df_r.columns.duplicated()]
            if df_i is not None:
                df_i = df_i.rename(columns=OLD_TO_NEW_MAP)
                df_i = df_i.loc[:, ~df_i.columns.duplicated()]

            # [수정 포인트 1]: 기존에는 df_comb에 df_v와 df_r만 합쳤으나, 이제 df_i(현재 최적 조건 새로운 데이터)도 함께 병합합니다.
            # 또한 drop_duplicates()를 추가하여 완전히 중복되는 동일 데이터는 한 번만 남기고 새로운 데이터만 추가되도록 처리합니다.
            df_comb = pd.concat([df for df in [df_i, df_v, df_r] if df is not None], ignore_index=True)
            df_comb = df_comb.drop_duplicates(ignore_index=True)
            df_comb = df_comb.loc[:, ~df_comb.columns.duplicated()]
            
            available_targets = [t for t in TARGET_VARS.keys() if t in df_comb.columns]
            
            if len(available_targets) == 0:
                st.sidebar.error(L['err_vars'])
            else:
                df_comb = df_comb.dropna(subset=available_targets)
                vars_list = [c for c in df_comb.columns if c not in TARGET_VARS.keys() and c != 'vars']
                models_dict, scalers_dict = {}, {}
                
                for target in available_targets:
                    df_target = df_comb.copy()
                    t_series = df_target[target]
                    if isinstance(t_series, pd.DataFrame):
                        t_series = t_series.iloc[:, 0]
                        
                    df_target[target] = np.where(t_series >= DEFECT_THRESHOLD, 1, 0)
                    
                    if vars_list and (int(t_series.nunique()) >= 2):
                        scaler = MinMaxScaler().fit(df_target[vars_list])
                        model = LogisticRegression(max_iter=1000).fit(scaler.transform(df_target[vars_list]), df_target[target])
                        models_dict[target], scalers_dict[target] = model, scaler

                bounds_dict = {
                    v: (int(np.floor(df_comb[v].min())), int(np.ceil(df_comb[v].max())) if int(np.floor(df_comb[v].min())) != int(np.ceil(df_comb[v].max())) else int(np.floor(df_comb[v].min())) + 1) for v in vars_list
                }

                st.session_state.update({
                    'models': models_dict, 'scalers': scalers_dict, 'df_injection': df_comb, 
                    'global_process_vars': vars_list, 'global_bounds': bounds_dict,
                    'ui_display_vars': [c for c in df_i.columns if c not in TARGET_VARS.keys() and c != 'vars'],
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
    st.sidebar.markdown(f"<h3 style='color:#00e5ff; font-size:1.1rem;'>{L['db_export_title']}</h3>", unsafe_allow_html=True)
    
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
                
                # [수정 포인트 2]: DB 스냅샷 저장 시에도 drop_duplicates()를 적용하여 중복 저장을 완벽히 차단합니다.
                if not existing_df.empty:
                    df_to_save = pd.concat([existing_df, df_to_save], ignore_index=True).drop_duplicates(ignore_index=True)
                else:
                    df_to_save = df_to_save.drop_duplicates(ignore_index=True)

                conn = sqlite3.connect(final_filename)
                df_to_save['vars'] = df_to_save.apply(lambda row: json.dumps(row.to_dict()), axis=1)
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
                        st.sidebar.markdown(f"<span style='color:#a3e635; font-size:0.85rem;'>{L['db_current_latest']}</span>", unsafe_allow_html=True)
                    
                    st.sidebar.markdown(f"✅ {L['db_prepared_msg']} `{target_file}`")
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

# 2. Main Screen
col_title, col_lang_switch = st.columns([8.5, 1.5])
with col_title:
    st.markdown(f"<h1 style='text-align: left; margin-bottom: 0;'>{L['main_title_1']}<span style='color:#00e5ff;'>{L['main_title_2']}</span></h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#64748b; margin-bottom: 1rem;'>{L['main_desc']}</p>", unsafe_allow_html=True)
with col_lang_switch:
    if st.button("🌐 KO / EN", key="lang_btn_main"):
        st.session_state.lang = "ko" if st.session_state.lang == "en" else "en"
        st.rerun()

is_active = len(st.session_state.get('models', {})) > 0
status_text = L['status_active'] if is_active else L['status_standby']
dot_color = "#00e5ff" if is_active else "#64748b"
var_count = len(st.session_state.get('ui_display_vars', []))
exp_weight = int(st.session_state.get('expert_reliability', 0.0) * 100)

opt_status = st.session_state.get('optimization_success', "N/A")
algo_info = st.session_state.get('selected_algorithm', "N/A")

if opt_status == "Converged": opt_display = f"{L['opt_converged']} ({algo_info})"
elif opt_status == "Failed": opt_display = f"{L['opt_failed']}"
else: opt_display = "N/A"

m1, m2, m3, m4 = st.columns(4)
m1.markdown(f'''<div class="metric-container"><div class="metric-label">{L['m_status']}</div><div class="metric-value"><span style="color:{dot_color}; margin-right:5px;">●</span><span style="color:#FFFFFF;">{status_text}</span></div></div>''', unsafe_allow_html=True)
m2.markdown(f'''<div class="metric-container"><div class="metric-label">{L['m_vars']}</div><div class="metric-value">{var_count} EA</div></div>''', unsafe_allow_html=True)
m3.markdown(f'''<div class="metric-container"><div class="metric-label">{L['m_reliability']}</div><div class="metric-value">{exp_weight}%</div></div>''', unsafe_allow_html=True)
m4.markdown(f'''<div class="metric-container"><div class="metric-label">{L['m_opt']}</div><div class="metric-value" style="font-size: 1.05rem;">{opt_display}</div></div>''', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if not is_active:
    st.info(L['info_standby'])

if is_active:
    t1, t2 = st.tabs([L['tab_diag'], L['tab_master']])

    with t1:
        st.markdown(f'<div class="section-title"><span class="square-icon"></span>{L["sec_a"]}</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, var in enumerate(st.session_state['ui_display_vars']):
            with cols[i % 3]:
                curr_val = st.session_state['current_inputs'].get(var, 0)
                st.session_state['current_inputs'][var] = st.slider(
                    f"{var}", 0, int(curr_val * 2) if curr_val > 0 else 100, int(curr_val), 
                    step=1, key=f"sl_{var}_{st.session_state['ver']}"
                )

        st.divider()

        st.markdown(f'<div class="section-title"><span class="square-icon"></span>{L["sec_c"]}</div>', unsafe_allow_html=True)
        active_targets = list(st.session_state['models'].keys())
        w_cols = st.columns(3)
        for idx, target_key in enumerate(active_targets):
            with w_cols[idx % 3]:
                is_on = st.checkbox(f"{TARGET_VARS[target_key]}", value=st.session_state['defect_switches'].get(target_key, True), key=f"onoff_{target_key}")
                st.session_state['defect_switches'][target_key] = is_on
                st.session_state['defect_weights'][target_key] = st.slider("", 0.0, 5.0, float(st.session_state['defect_weights'].get(target_key, 1.0)), step=0.5, disabled=not is_on, key=f"weight_{target_key}")
                st.markdown("<br>", unsafe_allow_html=True)
        
        st.write(L['sec_c_sub2'])
        selected_expert_vars = st.multiselect(L['lbl_constant'], options=st.session_state['ui_display_vars'], default=list(st.session_state['expert_constraints'].keys()))
        if selected_expert_vars:
            cols_b = st.columns(3) 
            for i, v_name in enumerate(selected_expert_vars):
                with cols_b[i % 3]:
                    st.session_state['expert_constraints'].setdefault(v_name, {'limit': st.session_state['current_inputs'].get(v_name, 0)})
                    st.session_state['expert_constraints'][v_name]['limit'] = st.number_input(f"{v_name}{L['lbl_target']}", value=int(st.session_state['expert_constraints'][v_name]['limit']), step=1)
        st.session_state['expert_reliability'] = st.slider(L['lbl_expert_rel'], 0, 100, int(st.session_state['expert_reliability']*100)) / 100.0
        st.divider()

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
            penalty = sum(abs(input_vals_list[list(all_v).index(v)] - c['limit']) / (c['limit'] + 1e-9) for v, c in st.session_state['expert_constraints'].items() if v in all_v)
            return min(1.0, avg_defect_risk + (penalty * st.session_state['expert_reliability']))

        def get_individual_risks(input_vals_list):
            all_v = st.session_state['global_process_vars']
            df_input = pd.DataFrame([input_vals_list], columns=all_v)
            risks = {}
            for target_key, model in st.session_state['models'].items():
                scaler = st.session_state['scalers'][target_key]
                risks[target_key] = model.predict_proba(scaler.transform(df_input))[0, 1]
            return risks

        st.markdown(f'<div class="section-title"><span class="square-icon"></span>{L["sec_d"]}</div>', unsafe_allow_html=True)
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
                
                new_row = {v: st.session_state['current_inputs'].get(v, 0.0) for v in all_v}
                for target_key, r_val in st.session_state['last_defect_risks'].items():
                    new_row[target_key] = r_val
                
                new_df = pd.DataFrame([new_row])
                st.session_state['df_injection'] = pd.concat([st.session_state['df_injection'], new_df], ignore_index=True)
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
                
                for algo in algorithms:
                    try:
                        res_temp = minimize(calculate_total_risk, x0, method=algo, bounds=bnds, options={'maxiter': 500})
                        if res_temp.success and res_temp.fun < best_fun:
                            best_fun = res_temp.fun
                            best_res = res_temp
                            chosen_algo = algo
                    except: continue
                
                try:
                    random_x0 = [np.random.uniform(b[0], b[1]) for b in bnds]
                    res_global = minimize(calculate_total_risk, random_x0, method='L-BFGS-B', bounds=bnds)
                    if res_global.success and res_global.fun < best_fun:
                        best_fun = res_global.fun
                        best_res = res_global
                        chosen_algo = "Hybrid Multi-Start (L-BFGS-B)"
                except: pass

                if best_res is not None:
                    final_x = [np.clip(val, bnds[i][0], bnds[i][1]) for i, val in enumerate(best_res.x)]
                    opt_dict = {v: int(round(val)) for v, val in zip(all_v, final_x)}
                    
                    st.session_state['last_res_val'] = calculate_total_risk(final_x)
                    st.session_state['last_defect_risks'] = get_individual_risks(final_x)
                    st.session_state['last_opt_df'] = pd.DataFrame([{v: opt_dict.get(v, 0) for v in st.session_state['ui_display_vars']}])
                    st.session_state['optimization_success'] = "Converged"
                    st.session_state['selected_algorithm'] = chosen_algo
                    
                    new_row = {v: opt_dict.get(v, 0) for v in all_v}
                    for target_key, r_val in st.session_state['last_defect_risks'].items():
                        new_row[target_key] = r_val
                    
                    new_df = pd.DataFrame([new_row])
                    st.session_state['df_injection'] = pd.concat([st.session_state['df_injection'], new_df], ignore_index=True)
                    st.session_state['data_changed_since_save'] = True
                    st.rerun()
                else:
                    st.session_state['optimization_success'] = "Failed"
                    st.session_state['selected_algorithm'] = "N/A"

        if st.session_state['last_res_val'] is not None:
            st.divider()
            val = st.session_state['last_res_val']
            total_risk_percent = int(round(val * 100))
            total_color = "#00e5ff" if total_risk_percent < 30 else "#ffab00" if total_risk_percent < 70 else "#ff5252"
            st.markdown(f"""<div style='background-color:#12141d; padding:25px; border-radius:10px; border:1px solid {total_color}44;'>
                    <h4 style='margin-top:0; color:#94a3b8;'>{L['dash_title']}</h4>
                    <h2 style='color:{total_color}; font-size:3rem; margin:0;'>{total_risk_percent}<span style='font-size:1.2rem;'>%</span></h2>
                </div>""", unsafe_allow_html=True)
            for target_key, full_name in TARGET_VARS.items():
                if target_key in st.session_state['last_defect_risks']:
                    r_val = st.session_state['last_defect_risks'][target_key]
                    r_perc = int(round(r_val * 100))
                    is_active_target = st.session_state['defect_switches'].get(target_key, True)
                    bar_color = "#00e5ff" if r_perc < 30 else "#ffab00" if r_perc < 70 else "#ff5252"
                    opacity_style = "opacity: 1.0;" if is_active_target else "opacity: 0.25;"
                    st.markdown(f"""<div style="margin-bottom: 12px; {opacity_style}"><span style="font-size:0.95rem; font-weight:600; color:#ffffff;">{full_name}</span><div class="custom-progress-container"><div class="custom-progress-bar" style="width: {r_perc}%; background: {bar_color};">{r_perc}%</div></div></div>""", unsafe_allow_html=True)
            
            if st.session_state['last_opt_df'] is not None:
                st.success(L['opt_success_msg'])
                df = st.session_state['last_opt_df'].astype(int)
                headers = "".join([f"<th>{c}</th>" for c in df.columns])
                rows = "".join([f"<td>{v}</td>" for v in df.values[0]])
                st.markdown(f"""<div style="overflow-x: auto;"><table class="optimized-table"><thead><tr>{headers}</tr></thead><tbody><tr>{rows}</tr></tbody></table></div>""", unsafe_allow_html=True)
                csv = st.session_state['last_opt_df'].to_csv(index=False).encode('utf-8-sig')
                st.download_button(label=L['btn_download'], data=csv, file_name='total_optimized_params.csv', mime='text/csv')
    
    with t2:
        if not st.session_state['df_injection'].empty:
            st.dataframe(st.session_state['df_injection'], use_container_width=True)
