from datetime import datetime, timezone
import base64, json, random, re, time
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from streamlit_drawable_canvas import st_canvas
from modules.config import (
    APP_NAME,
    APP_VERSION,
    CONSENT_TEXT,
    DIAGNOSTIC_DISCLAIMER,
    PRIVACY_NOTICE,
    STORAGE_ARCHITECTURE_NOTE,
)
from modules.audio_analysis import analyze_audio_bytes
from modules.motion_analysis import analyze_motion, analyze_gait
from modules.session import (
    assessment_completion,
    assessment_draft_payload,
    has_assessment_progress,
    initialize_session,
    reset_session,
    restore_assessment_draft,
)
from modules.reporting import (
    build_report,
    completed_module_explanations,
    humanize_feature_code,
    key_measurement_rows,
    report_json,
)
from modules.accounts import (
    authenticate,
    caregiver_sessions,
    delete_draft,
    delete_user_sessions,
    init_db,
    is_duplicate_email_error,
    load_draft,
    register,
    save_draft,
    save_session,
    storage_backend,
    update_consent_flag,
    update_profile,
    user_sessions,
    admin_overview,
)
from modules.pdf_report import make_pdf
from modules.cognitive_tasks import reaction_summary
from modules.clock_drawing import extract_clock_drawing_features, assess_darwin_feature_compatibility
from modules.model_registry import get_registry          # ← NEW
from modules.alzheimers_model import get_alzheimers_status, load_darwin_sample_profile  # ← NEW
from modules.parkinson_model import train_parkinson_model
from modules.praat_features import run_symptom_pattern_analysis

st.set_page_config(page_title='NeuroNexus | Digital Biomarker Research', page_icon='🧠', layout='wide', initial_sidebar_state='expanded')
components.html(
    """
    <!-- Google Tag Manager -->
    <script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
    new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
    j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
    'https://www.googletagmanager.com/gtm.js?id='+i+dl;
    })(window,document,'script','dataLayer','GTM-PFBLC9ZQ');</script>
    <!-- End Google Tag Manager -->
    """,
    height=0,
)
init_db(); initialize_session(st.session_state)
for k,v in {'user':None,'reaction_times':[],'reaction_started':None,'reaction_ready':False,'clock_result':None}.items():
    if k not in st.session_state: st.session_state[k]=v
if (
    st.session_state.get("user")
    and has_assessment_progress(st.session_state)
    and not st.session_state.get("draft_finalized", False)
):
    save_draft(
        st.session_state.user["id"],
        assessment_draft_payload(st.session_state),
    )

def navigate_workspace(page, directory_role=None):
    st.session_state['workspace_page']=page
    if directory_role is not None:
        st.session_state['directory_role']=directory_role


def mark_assessment_changed():
    st.session_state.draft_finalized=False


st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root{--brand:#1E565C;--sage:#3A7D6B;--bg:#F8FAF9;--line:#D8E2E3;--ink:#1A2426;--muted:#526260}
html,body,[class*="css"]{font-family:'Atkinson Hyperlegible',sans-serif;color:var(--ink);font-size:18px;line-height:1.6}
h1,h2,h3,h4,.stButton button,.stDownloadButton button{font-family:'Plus Jakarta Sans',sans-serif}
h1,h2,h3,h4{color:var(--ink);letter-spacing:-.025em}
.stApp{background-color:var(--bg);background-image:radial-gradient(ellipse at 92% 8%,rgba(58,125,107,.07),transparent 30%),url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 180'%3E%3Cpath d='M0 90h205l25-2 22-42 28 88 34-70 30 26h174l24-2 24-36 24 68 28-34h582' fill='none' stroke='%233A7D6B' stroke-opacity='.12' stroke-width='5'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:top right,bottom center;background-size:auto, min(100%, 1200px) 180px}
[data-testid="stMain"]{background:transparent}
.block-container{max-width:1360px;padding-top:1.7rem;padding-bottom:3rem}
[data-testid="stHeadingWithActionElements"]:has(> h1):not(.nn-hero [data-testid="stHeadingWithActionElements"]){margin:12px 0 18px;padding:12px 18px;border-left:5px solid var(--sage);border-radius:0 14px 14px 0;background:linear-gradient(90deg,rgba(255,255,255,.92),rgba(255,255,255,.58));box-shadow:0 4px 16px rgba(30,86,92,.035)}
[data-testid="stHeadingWithActionElements"] h1{margin:0;color:var(--brand);font-size:clamp(1.7rem,2.4vw,2.25rem);line-height:1.25}
[data-testid="stHeadingWithActionElements"] h2,[data-testid="stHeadingWithActionElements"] h3{color:var(--brand)}
[data-testid="stHeadingWithActionElements"]:has(> h2),[data-testid="stHeadingWithActionElements"]:has(> h3){margin:18px 0 10px}
[data-testid="stSidebar"]{background:#EFF4F2;border-right:1px solid var(--line)}
[data-testid="stSidebar"] [data-testid="stImage"] img{filter:drop-shadow(0 5px 12px rgba(30,86,92,.1))}
[data-testid="stSidebar"] [role="radiogroup"] label{min-height:48px;padding:5px 10px;border-radius:10px;transition:background-color .18s ease,color .18s ease}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:#E1EEE9}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:#DCEAE5;color:#1E565C;font-weight:700}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:var(--muted)}
[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);padding:20px;border-radius:16px;box-shadow:0 6px 20px rgba(30,86,92,.04)}
[data-testid="stMetricLabel"]{font-size:18px}
[data-testid="stMetricValue"]{color:var(--brand)}
[data-testid="stMetric"] [data-testid="stMetricDelta"]{font-size:15px}
[data-testid="stDataFrame"],[data-testid="stTable"]{overflow:hidden;border:1px solid #C6D8D2;border-top:4px solid var(--brand);border-radius:16px;background:#fff;box-shadow:0 12px 30px rgba(30,86,92,.09);transition:box-shadow .2s ease,border-color .2s ease}
[data-testid="stDataFrame"]:hover,[data-testid="stTable"]:hover{border-color:#A9C6BD;box-shadow:0 16px 36px rgba(30,86,92,.13)}
[data-testid="stDataFrame"] iframe{border-radius:12px}
[data-testid="stDataFrame"] [data-testid="stElementContainer"]{background:#fff}
[data-testid="stDataFrame"] [data-testid="stToolbar"]{background:#F4F8F6;border-bottom:1px solid #E1EBE7;padding:5px 10px}
[data-testid="stDataFrame"] [data-testid="stToolbar"] button{min-width:40px;min-height:40px;border-radius:10px;color:var(--brand)}
[data-testid="stDataFrame"] [data-testid="stToolbar"] button:hover{background:#E1EEE9}
[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"]{background:#fff}
[data-testid="stMarkdownContainer"] table{width:100%;border-collapse:separate;border-spacing:0;overflow:hidden;border:1px solid var(--line);border-radius:12px;background:#fff}
[data-testid="stMarkdownContainer"] th{background:#EAF2EF;color:var(--brand);font-weight:700}
[data-testid="stMarkdownContainer"] th,[data-testid="stMarkdownContainer"] td{padding:12px 14px;border-bottom:1px solid #E8EFED;text-align:left}
[data-testid="stMarkdownContainer"] tr:last-child td{border-bottom:0}
[data-testid="stMarkdownContainer"] tr:hover td{background:#F5F8F7}
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button{min-height:52px;padding:0 1.15rem;border-radius:12px;border:1px solid var(--line);font-size:18px;font-weight:700;transition:background-color .18s ease,border-color .18s ease,box-shadow .18s ease,transform .18s ease}
.stButton>button:hover,.stDownloadButton>button:hover,[data-testid="stFormSubmitButton"]>button:hover{border-color:var(--sage);box-shadow:0 5px 14px rgba(58,125,107,.14);transform:translateY(-1px)}
button[data-testid="stBaseButton-primary"]{background:var(--sage)!important;border-color:var(--sage)!important;color:#fff!important}
button[data-testid="stBaseButton-primary"]:hover{background:#2F6B5B!important;border-color:#2F6B5B!important}
button[data-testid="stBaseButton-secondary"]{border-color:var(--line)!important;color:var(--brand)!important;background:#fff!important}
button[data-testid="stBaseButton-secondary"]:hover{background:#EAF2EF!important}
button:focus-visible,input:focus-visible,textarea:focus-visible,[role="tab"]:focus-visible{outline:3px solid #78A99A!important;outline-offset:3px}
input:not([type="checkbox"]):not([type="radio"]),textarea,[data-baseweb="select"]>div{min-height:48px;border-radius:10px!important;border-color:var(--line)!important;font-size:18px}
[data-testid="stWidgetLabel"]{font-size:18px}
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea{background:#FBFCFB}
[data-testid="stFileUploader"],[data-testid="stAudioInput"],[data-testid="stCameraInput"]{border:1px dashed #AFC7C0;border-radius:14px;background:#F7FAF8;padding:8px}
[data-testid="stRadio"] label,[data-testid="stCheckbox"] label{min-height:48px;align-items:center}
[data-testid="stTabs"] [role="tab"]{min-height:48px;padding:0 1rem;font-size:18px;font-weight:700;color:var(--muted);transition:color .18s ease,border-color .18s ease}
[data-testid="stTabs"] [aria-selected="true"]{color:var(--brand)!important;border-bottom-color:var(--sage)!important}
[data-testid="stProgressBar"]>div>div{transition:width .4s ease;background-color:var(--sage)}
[data-testid="stExpander"],[data-testid="stForm"]{border:1px solid var(--line);border-radius:14px;background:#fff}
[data-testid="stExpander"]{box-shadow:0 4px 14px rgba(30,86,92,.035);transition:border-color .18s ease,box-shadow .18s ease}
[data-testid="stExpander"]:hover{border-color:#B8CFCA;box-shadow:0 8px 18px rgba(30,86,92,.07)}
[data-testid="stExpander"] summary{min-height:52px}
[data-testid="stForm"]{padding:18px}
[data-testid="stAlert"]{border-radius:14px;border:1px solid rgba(30,86,92,.12);box-shadow:0 4px 14px rgba(30,86,92,.035)}
[data-testid="stAlert"][kind="info"]{background:#EDF5F2;color:#234B49}
[data-testid="stAlert"][kind="success"]{background:#EAF4EF;color:#28584D}
[data-testid="stAlert"][kind="warning"]{background:#FBF5E9;color:#654E20}
[data-testid="stAlert"][kind="error"]{background:#FAEEEE;color:#6A3333}
[data-testid="stPlotlyChart"]{overflow:hidden;border:1px solid var(--line);border-radius:16px;background:#fff;padding:10px;box-shadow:0 8px 20px rgba(30,86,92,.05)}
[data-testid="stJson"]{box-shadow:0 6px 18px rgba(30,86,92,.04)}
.stMarkdown hr{border-color:var(--line)}
.nn-hero{position:relative;isolation:isolate;overflow:hidden;background:linear-gradient(120deg,#1E565C,#244F52);color:#fff;padding:32px 38px;border-radius:16px;margin-bottom:22px;box-shadow:0 12px 30px rgba(30,86,92,.12)}
.nn-hero:after{content:"";position:absolute;z-index:-1;right:4%;top:10%;width:126px;height:126px;border-radius:32px;background:linear-gradient(rgba(255,255,255,.07),rgba(255,255,255,.07)),linear-gradient(90deg,transparent 42%,rgba(255,255,255,.09) 42% 58%,transparent 58%),linear-gradient(transparent 42%,rgba(255,255,255,.09) 42% 58%,transparent 58%);transform:rotate(12deg)}
.nn-hero h1{color:#fff;font-size:clamp(2rem,4vw,3rem);line-height:1.15}
.nn-hero-title{display:flex;align-items:center;gap:16px;margin:0 0 12px}
.nn-hero-logo{width:72px;height:72px;flex:none}
.nn-hero-title h1{margin:0}
.nn-hero p{position:relative;color:#E3EFEC;font-size:18px;max-width:720px}
.nn-card{background:#fff;border:1px solid var(--line);padding:22px;border-radius:16px;transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease}
.nn-card:hover{transform:translateY(-2px);border-color:#B8CFCA;box-shadow:0 10px 24px rgba(30,86,92,.08)}
.nn-muted{color:var(--muted);font-size:18px;line-height:1.6}
.nn-pill{display:inline-flex;align-items:center;min-height:36px;background:#E1EEE9;color:#28584D;border-radius:99px;padding:5px 13px;font-size:15px;font-weight:700;letter-spacing:.04em}
[data-testid="stJson"]{border:1px solid var(--line);border-radius:14px;background:#fff;padding:12px}
@media(max-width:700px){.block-container{padding-top:1rem;padding-bottom:2rem}.nn-hero{padding:24px 22px}.nn-hero:after{right:-32px;top:auto;bottom:-45px;width:100px;height:100px}.stButton>button,.stDownloadButton>button{width:100%}[data-testid="stHeadingWithActionElements"]:has(> h1):not(.nn-hero [data-testid="stHeadingWithActionElements"]){padding:10px 14px;margin:10px 0 14px}[data-testid="stDataFrame"],[data-testid="stTable"]{border-radius:12px}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}
@page{size:A4;margin:16mm}
@media print{
  html,body,.stApp{background:#fff!important;color:#111!important;font-size:12pt!important}
  [data-testid="stSidebar"],[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"],[data-testid="stTabs"] [role="tablist"],.stButton,.stDownloadButton,footer{display:none!important}
  .block-container{max-width:none!important;padding:0!important;margin:0!important}
  [data-testid="stMain"]{background:#fff!important}
  [data-testid="stJson"]{border:1px solid #cbd5d3!important;box-shadow:none!important;break-inside:avoid}
  [data-testid="stAlert"]{background:#fff!important;box-shadow:none!important}
  .nn-hero{background:#1E565C!important;print-color-adjust:exact;-webkit-print-color-adjust:exact;box-shadow:none!important}
  a{color:#111!important;text-decoration:none!important}
}
</style>''',unsafe_allow_html=True)

# Auth entry
if not st.session_state.user:
    logo_path=Path(__file__).resolve().parent/'assets'/'neuronexus-logo.svg'
    logo_svg=logo_path.read_text(encoding='utf-8').split('<text',1)[0].replace(
        'viewBox="0 0 520 120"',
        'viewBox="0 0 120 120"',
    )+'</svg>'
    logo_data=base64.b64encode(logo_svg.encode('utf-8')).decode('ascii')
    st.markdown(
        f'<div class="nn-hero"><div class="nn-hero-title">'
        f'<img class="nn-hero-logo" src="data:image/svg+xml;base64,{logo_data}" alt="">'
        '<h1>NeuroNexus</h1></div>'
        '<p>A calmer way to explore voice, cognition and movement measurements.</p>'
        '<span class="nn-pill">RESEARCH PROTOTYPE</span></div>',
        unsafe_allow_html=True,
    )
    left,right=st.columns([1.05,1])
    with left:
        st.subheader('Welcome')
        st.write('Sign in to access your personal research dashboard, assessments and reports.')
        st.info('NeuroNexus is experimental software. It cannot diagnose or rule out Alzheimer’s or Parkinson’s disease.')
        with st.expander('Privacy and data handling'):
            st.markdown(PRIVACY_NOTICE)
        st.caption('Demo administrator: admin@neuronexus.local · Password: ChangeMe123!')
    with right:
        login_tab,reg_tab=st.tabs(['Sign in','Create account'])
        with login_tab:
            with st.form('login_form'):
                email=st.text_input('Email address'); password=st.text_input('Password',type='password')
                submitted=st.form_submit_button('Sign in',type='primary',use_container_width=True)
            if submitted:
                user=authenticate(email,password)
                if user: st.session_state.user=user; st.rerun()
                else: st.error('Email or password is incorrect.')
        with reg_tab:
            with st.form('register_form'):
                name=st.text_input('Full name'); email2=st.text_input('Email address'); pw=st.text_input('Create password',type='password'); pw2=st.text_input('Confirm password',type='password')
                account_role=st.selectbox('Account type',['Individual','Caregiver'])
                caregiver=st.text_input('Caregiver email (optional; for individual accounts)')
                ok=st.checkbox('I understand this is a research prototype, not a medical diagnostic service.')
                create=st.form_submit_button('Create individual account',type='primary',use_container_width=True)
            if create:
                if not name.strip() or '@' not in email2 or len(pw)<10 or pw!=pw2 or not ok: st.error('Please complete all fields, use a password of at least 10 characters, and accept the notice.')
                else:
                    try:
                        register(
                            name,
                            email2,
                            pw,
                            'caregiver' if account_role=='Caregiver' else 'individual',
                            caregiver,
                            consent_flags={"registration_acknowledged": True},
                        )
                        st.success('Account created. Please sign in.')
                    except Exception as exc:
                        if is_duplicate_email_error(exc):
                            st.error('That email may already be registered.')
                        else:
                            raise
    st.stop()

user=st.session_state.user
with st.sidebar:
    st.image('assets/neuronexus-logo.svg',width=245)
    st.caption('Digital Biomarker Research OS')
    st.write(f"Signed in as **{user['name']}**")
    if user['role']=='admin': workspace_pages=['Admin overview','User directory','Research assessments','Model registry','My profile','Documentation']
    elif user['role']=='caregiver': workspace_pages=['Caregiver overview','My profile','Documentation']
    else: workspace_pages=['My dashboard','Screening assessments','My reports','My profile','Documentation']
    if st.session_state.get('workspace_page') not in workspace_pages:
        st.session_state['workspace_page']=workspace_pages[0]
    page=st.radio('Workspace',workspace_pages,key='workspace_page')
    st.divider(); st.caption('Experimental research software · v2.0 prototype')
    if st.button('Sign out',use_container_width=True):
        reset_session(st.session_state)
        st.session_state.reaction_times=[]
        st.session_state.reaction_started=None
        st.session_state.reaction_ready=False
        st.session_state.user=None
        st.rerun()

# Shared top banner
st.markdown('<div class="nn-hero"><h1>NeuroNexus</h1><p>Understand your measurements. Track your sessions. Know when to seek professional advice.</p><span class="nn-pill">VOICE · COGNITION · MOVEMENT</span></div>',unsafe_allow_html=True)
st.warning(DIAGNOSTIC_DISCLAIMER)

if page=='Admin overview':
    data=admin_overview(); st.title('Administration overview'); st.caption('Live account and session activity from the local demonstration database.')
    people_df=pd.DataFrame(data['people'])
    recent_df=pd.DataFrame(data['recent'])
    role_counts=people_df['role'].value_counts() if not people_df.empty else pd.Series(dtype='int64')
    caregiver_count=int(role_counts.get('caregiver',0))
    admin_count=int(role_counts.get('admin',0))
    if not recent_df.empty:
        recent_df['created_at']=pd.to_datetime(recent_df['created_at'],errors='coerce')
        recent_df=recent_df.dropna(subset=['created_at'])
        today=pd.Timestamp.now().normalize()
        sessions_today=int((recent_df['created_at'].dt.normalize()==today).sum())
        activity_by_day=(recent_df.assign(day=recent_df['created_at'].dt.strftime('%b %d'))
                         .groupby('day',sort=False).size().rename('Sessions').reset_index()
                         .rename(columns={'day':'Date'}))
    else:
        sessions_today=0
        activity_by_day=pd.DataFrame(columns=['Date','Sessions'])
    st.markdown('#### At a glance')
    metric_cols=st.columns(5)
    with metric_cols[0]:
        st.metric('Total accounts',data['users'])
        st.button('View all accounts',key='admin_all_accounts',use_container_width=True,
                  on_click=navigate_workspace,args=('User directory','All users'))
    with metric_cols[1]:
        st.metric('Individuals',data['individuals'])
        st.button('View individuals',key='admin_individual_accounts',use_container_width=True,
                  on_click=navigate_workspace,args=('User directory','Individuals'))
    with metric_cols[2]:
        st.metric('Caregivers',caregiver_count)
        st.button('View caregivers',key='admin_caregiver_accounts',use_container_width=True,
                  on_click=navigate_workspace,args=('User directory','Caregivers'))
    metric_cols[3].metric('Saved sessions',data['sessions'])
    metric_cols[4].metric('Sessions today · latest 30',sessions_today,help='Sessions dated today among the 30 most recent saved sessions.')
    chart_col,roles_col=st.columns([1.5,1])
    with chart_col:
        with st.container(border=True):
            st.subheader('Recent session activity')
            st.caption('Daily sessions across the latest 30 recorded activities.')
            if activity_by_day.empty:
                st.info('Session activity will appear here after users save sessions.')
            else:
                st.bar_chart(activity_by_day,x='Date',y='Sessions',alt='Daily session activity across the latest 30 recorded sessions')
    with roles_col:
        with st.container(border=True):
            st.subheader('Accounts by role')
            role_chart=pd.DataFrame({
                'Role':['Individual','Caregiver','Admin'],
                'Accounts':[int(role_counts.get('individual',0)),caregiver_count,admin_count],
            })
            st.bar_chart(role_chart,x='Role',y='Accounts',horizontal=True,alt='Account counts for individuals, caregivers, and admins')
    st.subheader('Latest session activity')
    st.caption('The newest 30 saved sessions, refreshed whenever the dashboard loads.')
    if recent_df.empty:
        st.info('No saved sessions yet.')
    else:
        recent_table=recent_df.rename(columns={'id':'Session ID','created_at':'Date and time','name':'Account name','email':'Email'})
        recent_table['Date and time']=recent_table['Date and time'].dt.strftime('%b %d, %Y · %I:%M %p')
        st.dataframe(recent_table,hide_index=True,alt='Latest saved session activity')
    st.info('This is a prototype admin view. Before real deployment, add server-side authorization, audit logs, role management, secure hosting and a formal privacy/security review.')
elif page=='User directory':
    data=admin_overview(); st.title('User directory')
    people_df=pd.DataFrame(data['people'])
    role_counts=people_df['role'].value_counts() if not people_df.empty else pd.Series(dtype='int64')
    caregiver_count=int(role_counts.get('caregiver',0))
    admin_count=int(role_counts.get('admin',0))
    st.caption('Search accounts and view individuals, caregivers, and administrators separately.')
    search_col,role_col,sort_col=st.columns([1.5,1.1,1.2])
    with search_col:
        directory_search=st.text_input('Search accounts',placeholder='Name, email, or account ID',key='directory_search')
    with role_col:
        selected_role=st.segmented_control('Account type',['All users','Individuals','Caregivers','Admins'],default='All users',key='directory_role')
    with sort_col:
        sort_order=st.selectbox('Sort users by',['Newest first','Oldest first','Name A–Z','Name Z–A','ID ascending','ID descending'],key='directory_sort')
    if selected_role!='All users':
        role_value={'Individuals':'individual','Caregivers':'caregiver','Admins':'admin'}[selected_role]
        people_df=people_df[people_df['role'].eq(role_value)]
    if directory_search.strip():
        query=directory_search.strip().casefold()
        search_match=(
            people_df['name'].fillna('').astype(str).str.casefold().str.contains(query,regex=False)
            | people_df['email'].fillna('').astype(str).str.casefold().str.contains(query,regex=False)
            | people_df['id'].astype(str).str.contains(query,regex=False)
        )
        people_df=people_df[search_match]
    sort_options={
        'Newest first':('created_at',False),
        'Oldest first':('created_at',True),
        'Name A–Z':('name',True),
        'Name Z–A':('name',False),
        'ID ascending':('id',True),
        'ID descending':('id',False),
    }
    sort_column,ascending=sort_options[sort_order]
    if not people_df.empty:
        people_df=people_df.sort_values(sort_column,ascending=ascending,kind='stable')
    result_col,role_summary_col=st.columns([1,1])
    with result_col:
        st.metric('Accounts shown',len(people_df),help='Updates as the search, role, and sort controls change.')
    with role_summary_col:
        st.caption(f"Directory total: {len(data['people'])} accounts · {caregiver_count} caregivers · {int(role_counts.get('individual',0))} individuals · {admin_count} admins")
    if people_df.empty:
        st.info('No accounts match your search and account-type filters.')
    else:
        directory_table=people_df.rename(columns={'id':'Account ID','name':'Name','email':'Email','role':'Account type','created_at':'Joined'})
        st.dataframe(directory_table,hide_index=True,alt='Searchable user directory with account types and join dates')
elif page=='Model registry':
    st.title('Research model registry')
    st.caption('Model versions, dataset sources, grouped-CV metrics and validation stage per disease pathway. Research administration only — never a clinical claim.')
    reg=get_registry()
    for disease,info in reg.items():
        st.subheader('Parkinson’s pathway' if disease=='parkinsons' else 'Alzheimer’s pathway')
        if info['available']:
            m=info['subject_level']
            cols=st.columns(4)
            cols[0].metric('Subject-level balanced acc',f"{m['balanced_accuracy']:.3f}")
            cols[1].metric('Sensitivity',f"{m['sensitivity']:.3f}")
            cols[2].metric('Specificity',f"{m['specificity']:.3f}")
            cols[3].metric('Validation stage',"2 of 4 (held-out participants)")
            st.caption(f"Model: {info['model_name']} v{info.get('version')} · Selected: {info['selected_model']} · Participants: {info['participants']} · Dataset: {info['dataset_source']}")
            if info.get('models_compared'):
                with st.expander('Models compared (balanced accuracy)'):
                    st.json(info['models_compared'])
            confidence_interval=info.get('subject_level_95ci')
            if confidence_interval and len(confidence_interval)==2:
                st.caption(f"95% CI (subject-level): {confidence_interval[0]:.3f} – {confidence_interval[1]:.3f} · Trained: {info.get('trained_on')}")
            else:
                st.caption(f"Trained: {info.get('trained_on') or 'Not reported'}")
            for lim in info['limitations']: st.warning(lim)
        else:
            st.info(info['message'])
    st.divider()
    st.markdown('**Validation ladder:** 1 = model trained · 2 = evaluated on held-out participants · 3 = externally validated · 4 = clinically validated. NeuroNexus models are at most stage 2.')
elif page=='Research assessments':
    st.title('Research model laboratory')
    st.write('Develop and evaluate disease-specific research models using appropriately labelled datasets. This workspace is for research administration, not patient diagnosis.')
    st.subheader('Parkinson’s voice baseline — first research model')
    st.markdown('Upload the **UCI Oxford Parkinson’s voice CSV** containing `name`, numeric voice-feature columns, and `status` (0 = control, 1 = Parkinson’s-labelled). The training pipeline keeps recordings from the same participant together during cross-validation.')
    st.info('Important: this dataset contains precomputed biomedical voice features. It is not compatible with the app’s current MFCC summary, phone accelerometer or gait features. Training this baseline does not enable live voice or multimodal patient predictions.')
    dataset=st.file_uploader('Upload labelled Parkinson’s research CSV',type=['csv'],key='pd_research_dataset')
    if st.button('Train and evaluate research baseline',type='primary',disabled=dataset is None):
        try:
            result=train_parkinson_voice_model(dataset,output_dir='models')
            st.session_state['pd_model_metadata']=result
            st.success('Research baseline trained and evaluated. Review limitations before interpreting any metric.')
        except Exception as exc:
            st.error(str(exc))
    result=st.session_state.get('pd_model_metadata')
    if result:
        st.subheader('Participant-grouped evaluation')
        st.caption(f"{result['dataset_rows']} recordings · {result['participant_count']} participants · {result['feature_count']} voice features")
        metrics=result['metrics']
        cols=st.columns(4)
        for col,(label,key) in zip(cols,[('Balanced accuracy','balanced_accuracy'),('Sensitivity','sensitivity'),('Specificity','specificity'),('F1 score','f1')]):
            value=metrics.get(key); col.metric(label, f"{value:.3f}" if value is not None else 'N/A')
        st.json(metrics['confusion_matrix'])
        st.warning('These are internal cross-validation results on a small, retrospective voice dataset—not clinical performance and not evidence that this model can identify disease in an individual. Independent external validation is required.')
        st.download_button('Download model evaluation metadata',json.dumps(result,indent=2),file_name='parkinson_voice_research_metadata.json',mime='application/json')
elif page=='Caregiver overview':
    st.title('Caregiver portal'); st.write('View session summaries only for participants who entered your email as their caregiver contact.')
    rows=caregiver_sessions(user['email'])
    if rows:
        for r in rows:
            with st.expander(f"{r['participant_name']} · {r['created_at']}"):
                st.json(json.loads(r['summary']))
    else: st.info('No linked participant sessions yet. Ask the participant to add this email in My profile. No automatic disease alerts are generated.')
elif page=='My profile':
    st.title('My profile')
    role_options=['Individual (Participant)','Caregiver','Admin / Clinician Triage']
    actual_preview={'individual':role_options[0],'caregiver':role_options[1],'admin':role_options[2]}
    role_preview=st.radio(
        'Preview role',
        role_options,
        index=role_options.index(actual_preview.get(user['role'],role_options[0])),
        horizontal=True,
        key=f"profile_role_preview_{user['id']}",
    )
    st.caption('Preview only: this switcher does not change your signed-in role, permissions, or access to another person’s data.')
    profile_generation=int(st.session_state.get('profile_widget_generation',0))
    profile_notice=st.session_state.pop('profile_notice',None)
    if profile_notice:
        st.success(profile_notice)
    if not st.session_state.get('consent_granted'):
        has_optional_context=bool(
            st.session_state.get('baseline_context')
            or st.session_state.get('caregiver_observation_notes')
            or st.session_state.get('caregiver_notes_consent')
        )
        st.session_state.baseline_context=None
        st.session_state.caregiver_observation_notes=[]
        st.session_state.caregiver_notes_consent=False
        if has_optional_context:
            profile_generation+=1
            st.session_state.profile_widget_generation=profile_generation

    if role_preview==role_options[0]:
        st.subheader('Baseline calibration')
        baseline=st.session_state.get('baseline_context') or {}
        if not st.session_state.get('consent_granted'):
            st.info('Grant consent on Screening assessments before saving optional profile context.')
        age_options=['Not provided','18–39','40–59','60–69','70+']
        sex_options=['Not provided','Female','Male','Intersex','Prefer not to say']
        hand_options=['Not provided','Right','Left','Ambidextrous','Prefer not to say']
        device_options=['Not provided','Touchscreen / Stylus','Mouse / Trackpad']
        def baseline_index(field,options):
            return options.index(baseline[field]) if baseline.get(field) in options else 0
        with st.form(f'baseline_context_form_{profile_generation}'):
            age=st.selectbox('Age bracket',age_options,index=baseline_index('age_bracket',age_options))
            biological_sex=st.selectbox('Biological sex',sex_options,index=baseline_index('biological_sex',sex_options))
            dominant_hand=st.selectbox('Dominant hand',hand_options,index=baseline_index('dominant_hand',hand_options))
            input_device=st.selectbox('Primary input device',device_options,index=baseline_index('primary_input_device',device_options))
            save_baseline=st.form_submit_button(
                'Save baseline context',
                type='primary',
                disabled=not st.session_state.get('consent_granted'),
            )
        if save_baseline:
            st.session_state.baseline_context={
                'age_bracket':age,
                'biological_sex':biological_sex,
                'dominant_hand':dominant_hand,
                'primary_input_device':input_device,
            }
            st.success('Optional context saved in this session. It appears in reports generated from this session.')

        st.subheader('Privacy & data governance')
        if st.session_state.get('consent_granted'):
            st.success('Consent status: Granted for the current session')
        else:
            st.info('Consent status: Not granted for the current session')
        history_key=f'local_history_enabled_{profile_generation}'
        if history_key not in st.session_state:
            st.session_state[history_key]=bool(st.session_state.get('local_history_enabled',False))
        history_enabled=st.toggle('Enable saved session history',key=history_key)
        st.session_state.local_history_enabled=history_enabled
        st.caption(
            'When enabled, you may save derived reports to the configured persistent '
            'store from the Report tab. Turning this off does not delete saved reports; '
            'use Purge My Session Data to delete them.'
        )
        if st.button('Purge My Session Data',type='secondary'):
            deleted_sessions=delete_user_sessions(user['id'])
            reset_session(st.session_state)
            st.session_state.reaction_times=[]
            st.session_state.reaction_started=None
            st.session_state.reaction_ready=False
            st.session_state.profile_notice=(
                f'Purged current session data and {deleted_sessions} saved session '
                f'{"summary" if deleted_sessions == 1 else "summaries"}. Downloaded files remain.'
            )
            st.rerun()
        st.caption('Purge removes this account’s finalized reports, local draft, and current in-memory assessment/context data. Downloaded files, account details, and system logs are outside this prototype control.')
    elif role_preview==role_options[1]:
        st.subheader('Linked participant')
        with st.container(border=True):
            st.caption('Sample card · not linked to an actual participant')
            linked_col,relationship_col,age_col=st.columns(3)
            linked_col.metric('Participant ID','SAMPLE-P-1042')
            relationship_col.metric('Relationship','Adult child')
            age_col.metric('Age bracket','70+ · illustrative')
        st.caption('This preview does not reveal or authorize access to another person’s account.')
        assisted_key=f'assisted_session_mode_{profile_generation}'
        if assisted_key not in st.session_state:
            st.session_state[assisted_key]=bool(st.session_state.get('assisted_session_mode',False))
        assisted=st.toggle('Assisted Session Mode',key=assisted_key)
        st.session_state.assisted_session_mode=assisted
        st.subheader('Caregiver observation notes')
        if not st.session_state.get('consent_granted'):
            st.info('Participant session consent is required before caregiver notes can be added to a report.')
        notes_consent_key=f'caregiver_notes_consent_{profile_generation}'
        if notes_consent_key not in st.session_state:
            st.session_state[notes_consent_key]=bool(st.session_state.get('caregiver_notes_consent',False))
        notes_consent=st.checkbox(
            'Participant agrees these caregiver observations may be included in this session report',
            key=notes_consent_key,
            disabled=not st.session_state.get('consent_granted'),
        )
        prior_notes_consent=bool(st.session_state.get('caregiver_notes_consent',False))
        st.session_state.caregiver_notes_consent=bool(notes_consent and st.session_state.get('consent_granted'))
        if (
            st.session_state.caregiver_notes_consent
            and not (user.get("consent_flags") or {}).get("caregiver_notes", False)
        ):
            updated_flags=update_consent_flag(user["id"],"caregiver_notes",True)
            user={**user,"consent_flags":updated_flags}
            st.session_state.user=user
        if prior_notes_consent and not st.session_state.caregiver_notes_consent:
            st.session_state.caregiver_observation_notes=[]
            st.session_state.profile_widget_generation=profile_generation+1
            profile_generation=int(st.session_state.profile_widget_generation)
        note_options=[
            ('tremor','Observed tremor'),
            ('memory','Observed memory lapses'),
            ('gait','Observed gait changes'),
        ]
        if st.session_state.caregiver_notes_consent:
            selected_notes=[]
            for note_key,label in note_options:
                widget_key=f'caregiver_note_{note_key}_{profile_generation}'
                checked=st.checkbox(label,key=widget_key)
                if checked:
                    selected_notes.append(label)
            st.session_state.caregiver_observation_notes=selected_notes
        else:
            st.session_state.caregiver_observation_notes=[]
            st.caption('Selected notes, if any, are user-reported observations—not clinical findings—and are added to the PDF report.')
    else:
        st.subheader('De-identified triage table preview')
        st.caption('Illustrative template only; no patient records or triage decisions are shown.')
        triage_preview=pd.DataFrame([
            {'Session ID':'DEMO-S-001','Risk flag':'Not generated','Flagged biomarkers':'No validated individual flags','Referral status':'Not assigned'},
            {'Session ID':'DEMO-S-002','Risk flag':'Not generated','Flagged biomarkers':'No validated individual flags','Referral status':'Not assigned'},
            {'Session ID':'DEMO-S-003','Risk flag':'Not generated','Flagged biomarkers':'No validated individual flags','Referral status':'Not assigned'},
        ])
        st.dataframe(triage_preview,hide_index=True,alt='Synthetic triage table template with no patient risk classifications')
        st.caption('This prototype does not calculate Low, Moderate, or Elevated clinical risk levels and does not generate referrals.')

        st.subheader('Model governance & fairness')
        registry=get_registry()
        park=registry['parkinsons']
        def format_metric(metrics,key):
            value=metrics.get(key)
            return f'{value:.3f}' if isinstance(value,(int,float)) else 'Not reported'
        if park.get('available'):
            subject=park.get('subject_level') or {}
            recording=park.get('recording_level') or {}
            park_validation=(
                f"Grouped CV; subject-level balanced accuracy {format_metric(subject,'balanced_accuracy')}, "
                f"sensitivity {format_metric(subject,'sensitivity')}, specificity {format_metric(subject,'specificity')}; "
                f"recording-level balanced accuracy {format_metric(recording,'balanced_accuracy')}. Research evaluation only."
            )
            park_source=f"{park.get('dataset_source','Unknown')} · {park.get('participants','Unknown')} participants; upstream license/citation not verified."
            park_version=park.get('version') or 'Not recorded'
            park_name=park.get('model_name') or 'Parkinson voice research model'
        else:
            park_validation=park.get('message','No validation metrics available.')
            park_source='multiple_sound attribution not verified'
            park_version='Unavailable'
            park_name='Parkinson voice research model'
        alzheimer=registry['alzheimers']
        if alzheimer.get('available'):
            alzheimer_meta=alzheimer.get('metadata') or {}
            alzheimer_dataset=alzheimer_meta.get('dataset') or {}
            alzheimer_metrics=alzheimer.get('subject_level') or {}
            alzheimer_validation=(
                f"Internal grouped participant CV; accuracy {format_metric(alzheimer_metrics,'accuracy')}, "
                f"balanced accuracy {format_metric(alzheimer_metrics,'balanced_accuracy')}, "
                f"ROC-AUC {format_metric(alzheimer_metrics,'roc_auc')}, "
                f"sensitivity {format_metric(alzheimer_metrics,'sensitivity')}, "
                f"specificity {format_metric(alzheimer_metrics,'specificity')}. Not clinical validation."
            )
            alzheimer_version=alzheimer.get('version') or 'Not recorded'
            alzheimer_source=(
                f"UCI DARWIN · {alzheimer_dataset.get('license','license not recorded')} · "
                f"DOI {alzheimer_dataset.get('doi','not recorded')}"
            )
        else:
            alzheimer_validation=alzheimer.get('message','No research model is installed.')
            alzheimer_version='Unavailable'
            alzheimer_source='UCI DARWIN; model not trained'
        governance=pd.DataFrame([
            {'Pathway':park_name,'Version':park_version,'Dataset attribution':park_source,'Validation':park_validation},
            {'Pathway':"Alzheimer's kinematics (DARWIN)",'Version':alzheimer_version,'Dataset attribution':alzheimer_source,'Validation':alzheimer_validation},
            {'Pathway':"Alzheimer's speech",'Version':'No trained model','Dataset attribution':'Not applicable','Validation':'Descriptive features only; no individual inference'},
        ])
        st.dataframe(governance,hide_index=True,alt='Current model versions, dataset attributions, and research validation status')
        st.subheader('Age-group fairness audit preview · not computed')
        fairness_preview=pd.DataFrame([
            {'Age group':'18–39','Sensitivity':'Not evaluated','Specificity':'Not evaluated','Audit status':'No subgroup validation evidence'},
            {'Age group':'40–59','Sensitivity':'Not evaluated','Specificity':'Not evaluated','Audit status':'No subgroup validation evidence'},
            {'Age group':'60+','Sensitivity':'Not evaluated','Specificity':'Not evaluated','Audit status':'No subgroup validation evidence'},
        ])
        st.dataframe(fairness_preview,hide_index=True,alt='Age-group fairness audit template; subgroup metrics have not been evaluated')
        st.caption('The age-group rows are an audit template, not simulated performance results. Demographic subgroup data and fairness metrics are not available.')

    st.divider()
    profile_col,account_col=st.columns([1.25,1])
    with profile_col:
        st.subheader('Personal details')
        with st.form('profile'):
            nm=st.text_input('Full name',value=user['name'])
            cg=st.text_input('Caregiver email',value=user.get('caregiver_email') or '')
            save=st.form_submit_button('Save profile',type='primary')
    with account_col:
        st.subheader('Account overview')
        role_label={'individual':'Individual','caregiver':'Caregiver','admin':'Administrator'}.get(user['role'],user['role'].title())
        joined=(user.get('created_at') or 'Unknown')[:10]
        st.metric('Account type',role_label)
        st.text_input('Email address',value=user['email'],disabled=True,key=f"profile_email_{user['id']}")
        st.caption(f'Member since {joined}')
    if save:
        update_profile(user['id'],nm,cg); st.session_state.user=authenticate(user['email'], '') or {**user,'name':nm,'caregiver_email':cg}; st.success('Profile updated.')
elif page=='Documentation':
    st.title('Technical & clinical reference')
    st.caption('Implementation facts, research provenance, safety boundaries, and a proposed path to evaluation. This prototype is not approved for clinical use.')
    st.info(STORAGE_ARCHITECTURE_NOTE,title='Privacy-by-design storage')
    st.caption(f'Persistent account/report backend: {storage_backend()}. Local assessment drafts remain on this app instance.')
    with st.expander('1. System architecture & multimodal pipeline',expanded=True):
        st.markdown("""
**Design principle — “To Sense is to Predict” is an aspiration, not a current clinical claim.** NeuroNexus brings several optional signal tasks into one consent-gated Streamlit session and report. It does not combine them in a validated multimodal disease-prediction model.

1. **Voice capture and standardization.** Record/uploaded audio is decoded and converted to mono 16 kHz WAV. The general voice path reports MFCC summaries, RMS, silence ratio, zero-crossing rate, spectral centroid, median pitch, and technical quality. It does **not** calculate vocal jitter, shimmer, or HNR in this general path.
2. **Separate sustained-vowel research path.** A sustained vowel can be processed with Praat-compatible settings for jitter, shimmer, HNR, pitch, and voice-break features. Those 26 features are passed to the existing Parkinson’s research artifact for an experimental pattern comparison. This result is task-specific, internally evaluated, and not a diagnosis or disease probability.
3. **Clock drawing.** The current mouse/touch canvas records path geometry such as stroke count, approximate path length, direction changes, and bounding-box area. It has no calibrated pressure or reliable pointer timestamps, so DARWIN-style pen velocity, air/pause time, and pressure cannot be calculated. A DARWIN kinematic model is trained for research, but this canvas schema is incompatible and is not scored by that model.
4. **Other behavioral/movement tasks.** Resting motion derives an acceleration-magnitude spectrum from a supplied CSV; gait reports candidate peaks and estimated cadence using a preliminary heuristic; reaction time is a browser/device response-time demonstration. None is a validated disease classifier.
5. **Session and report.** Derived measurements are assembled into a JSON/PDF report. Finalized reports are saved only when a user explicitly saves them. Incomplete module results stay in a local SQLite draft and are removed on reset, sign-out, or successful report save; raw audio and sensor uploads are not copied into that draft.

The speech-feature tab also computes descriptive transcript features (for example word count and type-token ratio). No trained Alzheimer’s speech classifier is installed.
""")
    with st.expander('2. Training datasets & legal governance'):
        st.markdown("""
### Parkinson’s voice artifact currently in `models/`

The checked-in metadata identifies the training source as **`multiple_sound`**: 1,040 recordings, 40 participants, 26 acoustic features, with 20 participants in each label group. It records a soft-voting **Logistic Regression + Gradient Boosting** model. The candidate pipelines use median imputation and `SelectKBest(f_classif, k=8)` feature selection; Logistic Regression and SVM pipelines standardize selected features. Evaluation uses 10-fold stratified grouped cross-validation, keeping participant recordings together.

- Participant-level grouped-CV mean: balanced accuracy **0.617**, sensitivity **0.600**, specificity **0.633**; reported 95% CI for balanced accuracy **0.443–0.790**.
- Recording-level out-of-fold metrics: ROC-AUC **0.573**, balanced accuracy **0.552**, sensitivity **0.546**, specificity **0.558**.
- These are internal retrospective research metrics, not external or clinical validation. The artifact has no recorded training date. The result must not be presented as a personal risk estimate.

The training utility also has a loader for the **UCI Oxford Parkinson’s voice CSV**, but that is not the dataset recorded in the checked-in artifact metadata. Source: [UCI Oxford Parkinson’s Disease Detection Dataset](https://archive.ics.uci.edu/dataset/174/parkinsons), DOI [10.24432/C59C74](https://doi.org/10.24432/C59C74).

### DARWIN handwriting dataset

The supplied UCI **DARWIN** table has 174 participants, an `ID`, 450 task features, and a `class` label. It is licensed **CC BY 4.0**. Attribution: Fontanella, F. (2022), *DARWIN* [Dataset], UCI Machine Learning Repository, DOI [10.24432/C55D0K](https://doi.org/10.24432/C55D0K); see the [UCI dataset record](https://archive.ics.uci.edu/dataset/732/darwin). NeuroNexus trains a calibrated logistic-regression research artifact with `SelectKBest(f_classif, k=20)` and participant-grouped cross-validation. Its metrics are internal research results, not external or clinical validation. The app does not apply the artifact to live web-canvas data.

### Attribution status and compliance

The checked-in Parkinson model metadata and local integration notes do not preserve an upstream URL, DOI, or license for the `multiple_sound` training files. The code describes them as Multiple Sound Recording data, but open-access status and redistribution rights have **not** been verified here. Do not claim complete licensing compliance or redistribute those data until the exact source and terms are recorded. Keep a dataset manifest with source, version, license, citation, access date, permitted uses, and transformations for every future training run.
""")
    with st.expander('3. Clinical scope, assumptions & limitations'):
        st.markdown("""
**Current scope:** experimental signal capture and descriptive research measurements only. NeuroNexus is **not currently a validated screening, triage, or diagnostic aid** and must not determine referral priority, disease status, treatment, or reassurance that disease is absent.

- The Parkinson’s artifact is trained on a small retrospective dataset and evaluated only with internal grouped cross-validation. It is not externally validated, calibrated for clinical risk, or validated for this app’s ordinary voice/MFCC input.
- A DARWIN kinematic research artifact is trained and registered, but it is not deployed for live canvas scoring or individual prediction. DARWIN’s tablet-derived 25-task features are not reconstructable from this mouse/touch canvas.
- Voice measures vary with language, accent, age, sex, task, microphone, room acoustics, and unrelated health conditions. Current representation and subgroup performance have not been comprehensively assessed.
- Phone motion depends on sensor calibration, placement, orientation, sampling rate, and timestamp quality; absent timestamps are assumed to be 50 Hz. The gait peak method may miss or double-count steps.
- Canvas geometry varies with mouse, touchscreen, stylus, browser, viewport, and drawing surface. This canvas has no calibrated pressure or capture-time data.
- Browser/device scheduling affects reaction-time estimates. Technical signal-quality scores measure usability of a recording, not health.

Hospital use would require a defined intended use, prospective ethics-approved studies, representative multi-site/device/language cohorts, locked analysis protocols, external validation, calibration, clinical oversight, regulatory assessment, and monitored post-deployment performance.
""")
    with st.expander('4. Privacy, security & responsible AI'):
        st.markdown("""
### Implemented safeguards and current behavior

- The consent checkbox gates access to assessment controls. It is not a validated e-consent workflow and no consent receipt is stored.
- The UI tells participants not to say names or addresses. There is **no automatic PII detection or redaction**; de-identify transcripts and avoid identifiers in audio and uploads.
- Audio features are computed in the active app session. Normalized voice bytes can remain in Streamlit session memory until the user clears the session, signs out, or the runtime session ends. Optional speech-audio extraction writes a temporary WAV under the OS temporary directory; the directory is removed when processing exits.
- Incomplete assessment module results are kept in a local SQLite `assessment_drafts` record to support resume. Clear-session, sign-out, and successful report-save remove that draft. Raw audio and sensor uploads are not stored in it.
- Account names, email addresses, caregiver links, roles, consent flags, password salts/hashes, and explicitly saved derived reports use Supabase PostgreSQL when `DATABASE_URL` is configured, or local SQLite otherwise. Passwords are PBKDF2-hashed. Database encryption, access policies, backups, logs, and retention need separate review; this authentication and role setup is not production security.

### Responsible AI status

Displayed feature values and technical-quality checks are **not** model explanations. SHAP/LIME, feature-attribution reporting, validated reference ranges, and subgroup fairness analysis are not implemented. No validated healthy baseline or automated “elevated” flag is available. Model metrics shown here are study-level validation summaries, not individual predictions.

Before any real participant data: complete ethics/privacy review, lawful basis and consent design, data minimization, access controls, encryption/key management, retention/deletion policy, audit logging, incident response, vendor/hosting review, and jurisdiction-specific legal review.
""")
    with st.expander('5. Real-world clinical deployment pathway'):
        st.markdown("""
**A three-minute pre-visit workflow is a hypothesis to test, not a measured capability or deployment claim.** Abu Dhabi primary-care clinics and community screening networks are possible settings for co-design; no clinical partner, approval, or deployment is currently confirmed.

1. **Prototype governance:** confirm dataset licenses and citations, intended use, ethics pathway, privacy impact, security architecture, languages, accessibility, and stakeholder requirements.
2. **Prospective silent-mode study:** recruit representative participants across sites and devices; collect reference assessments under an approved protocol; keep outputs hidden from care decisions while evaluating completion time, data quality, subgroup performance, calibration, and false-positive/false-negative impact.
3. **Clinician-supervised feasibility pilot:** only after independent evidence and required approvals, test workflow usability with trained staff. A qualified clinician remains responsible; no automated referral denial, diagnosis, or treatment decision.
4. **Controlled integration:** if evidence and regulation support it, add secure identity/access management, encrypted transport/storage, auditable consent, minimum-necessary EHR integration, retention controls, training, incident handling, drift monitoring, and a rollback plan.
5. **Ongoing oversight:** review performance across demographic/language/device groups, collect clinician and participant feedback, monitor harms and operational burden, and revalidate material model or workflow changes.

The prototype does not connect to an EHR, generate clinic referrals, or establish that a three-minute screen improves outcomes. Those claims require measured prospective evidence.
""")
elif page=='My dashboard':
    st.title(f"Good day, {user['name'].split()[0]}")
    sessions=user_sessions(user['id'])
    a,b,c=st.columns(3); a.metric('Saved sessions',len(sessions)); b.metric('Assessments available',6); c.metric('Latest session',sessions[0]['created_at'][:10] if sessions else 'Not yet')
    st.subheader('Your assessment journey')
    cols=st.columns(3)
    for col,(title,desc,ico) in zip(cols,[('Voice','Speech and acoustic measurements','🎙️'),('Cognition','Clock drawing and reaction time','🧩'),('Movement','Resting motion and gait','🚶')]):
        with col:
            st.markdown(f'<div class="nn-card"><h3>{ico} {title}</h3><p class="nn-muted">{desc}</p></div>',unsafe_allow_html=True)
    st.write('')
    st.button('Start a screening session',type='primary',on_click=navigate_workspace,args=('Screening assessments',))
    st.subheader('Saved assessment activity')
    st.caption('Counts show which activities appear in your saved summaries; they are not measures of health or change.')
    assessment_counts={
        'Voice':0,
        'Resting motion':0,
        'Gait':0,
        'Clock drawing':0,
        'Reaction time':0,
        'Speech features':0,
    }
    for session in sessions:
        summary=json.loads(session['summary'])
        for key,label in [('voice','Voice'),('resting_motion','Resting motion'),('gait','Gait')]:
            if summary.get(key):
                assessment_counts[label]+=1
        cognitive=summary.get('cognitive_tasks') or {}
        if (cognitive.get('clock_drawing') or {}).get('completed'):
            assessment_counts['Clock drawing']+=1
        if (cognitive.get('reaction_time') or {}).get('trials',0):
            assessment_counts['Reaction time']+=1
        if summary.get('alzheimers_speech_features'):
            assessment_counts['Speech features']+=1
    activity_df=pd.DataFrame({
        'Assessment':list(assessment_counts),
        'Saved sessions':list(assessment_counts.values()),
    })
    if any(assessment_counts.values()):
        st.bar_chart(activity_df,x='Assessment',y='Saved sessions',horizontal=True,
                     alt='Number of saved sessions containing each assessment activity')
    else:
        st.info('Assessment activity will appear here after you save a session with completed activities.')
    st.subheader('Recent sessions')
    if sessions:
        for s in sessions[:5]:
            with st.expander(s['created_at']): st.json(json.loads(s['summary']))
    else: st.info('Your saved session summaries will appear here after you complete assessments and choose Save session.')
elif page=='My reports':
    st.title('My reports'); sessions=user_sessions(user['id'])
    session_reports=[(session,json.loads(session['summary'])) for session in sessions]
    saved_modules={}
    for _,saved_report in session_reports:
        saved_modules.update(completed_module_explanations(saved_report))
    with st.container(border=True):
        st.subheader('How to read your results')
        if saved_modules:
            for label,description in saved_modules.items():
                st.markdown(f'- **{label}:** {description}')
        else:
            st.write('No completed modules are available in your saved reports yet.')
        st.caption('These are descriptive measurements, not diagnoses or disease-risk estimates.')
    registry=get_registry()
    st.subheader('Research model status')
    park=registry['parkinsons']
    if park['available']:
        metrics=park.get('subject_level') or {}
        metric_text=' · '.join(
            f"{label} {metrics[key]:.3f}"
            for label,key in [
                ('Balanced accuracy','balanced_accuracy'),
                ('Sensitivity','sensitivity'),
                ('Specificity','specificity'),
            ]
            if metrics.get(key) is not None
        )
        park_status=f"Research artifact installed · internal grouped CV: {metric_text or 'metrics unavailable'} · not clinically validated."
    else:
        park_status=park.get('message','No research model is installed.')
    alzheimer_status=registry['alzheimers']
    if alzheimer_status.get('available'):
        metrics=alzheimer_status.get('subject_level') or {}
        metric_text=' · '.join(
            f"{label} {metrics[key]:.3f}"
            for label,key in [
                ('Accuracy','accuracy'),
                ('Balanced accuracy','balanced_accuracy'),
                ('Sensitivity','sensitivity'),
                ('Specificity','specificity'),
                ('ROC-AUC','roc_auc'),
            ]
            if metrics.get(key) is not None
        )
        alzheimer_model_status=(
            f"Research artifact installed · internal participant-grouped CV: "
            f"{metric_text or 'metrics unavailable'} · not clinically validated; "
            "not used for live canvas scoring."
        )
    else:
        alzheimer_model_status=alzheimer_status.get('message','No DARWIN research model is installed.')
    model_rows=[
        {'Pathway':'Parkinson’s voice','Status':park_status},
        {'Pathway':'Alzheimer’s kinematics (DARWIN)','Status':alzheimer_model_status},
        {'Pathway':'Alzheimer’s speech features','Status':'Descriptive transcript features only; no speech model is installed.'},
    ]
    st.dataframe(pd.DataFrame(model_rows),hide_index=True,alt='Research model availability and validation status; not a personal risk result')
    st.caption('Model metrics describe research validation only. They are not patient-specific predictions. The current web canvas does not supply the DARWIN 25-task tablet feature schema.')
    if not sessions: st.info('No saved reports yet. Complete assessments and save a session first.')
    for s,report in session_reports:
        with st.expander(f"Session · {s['created_at']}"):
            st.json(report)
            st.subheader('Key measurement breakdown')
            st.dataframe(
                pd.DataFrame(key_measurement_rows(report)),
                hide_index=True,
                alt='Recorded measurements with no clinical reference ranges or risk interpretation',
            )
            st.download_button('Download JSON',s['summary'],file_name=f"neuronexus_session_{s['id']}.json",mime='application/json',key=f"json{s['id']}")
            st.download_button('Download PDF',make_pdf(user['name'],report,registry),file_name=f"neuronexus_session_{s['id']}.pdf",mime='application/pdf',key=f"pdf{s['id']}")
elif page=='Screening assessments':
    st.title('Your screening assessments'); st.write('Complete only the activities you feel comfortable doing. You may stop at any time.')
    with st.expander('Before you begin: consent and safety',expanded=not st.session_state.consent_granted):
        st.markdown(PRIVACY_NOTICE); st.checkbox(CONSENT_TEXT,key='consent_granted')
    if not st.session_state.consent_granted: st.info('Please provide consent before running analysis.'); st.stop()
    if not (user.get("consent_flags") or {}).get("assessment", False):
        updated_flags=update_consent_flag(user["id"],"assessment",True)
        user={**user,"consent_flags":updated_flags}
        st.session_state.user=user
    if (
        not st.session_state.get("draft_resumed", False)
        and not has_assessment_progress(st.session_state)
    ):
        saved_draft=load_draft(user["id"])
        if saved_draft:
            completed_modules=sum(assessment_completion(saved_draft).values())
            with st.container(border=True):
                st.subheader('Unfinished assessment draft')
                st.info(
                    f'A local draft is available with {completed_modules} completed '
                    'module(s). Raw audio and sensor uploads are not stored in the draft.'
                )
                resume_col,begin_col=st.columns(2)
                if resume_col.button(
                    'Resume Where You Left Off',
                    type='primary',
                    width='stretch',
                ):
                    restore_assessment_draft(st.session_state,saved_draft)
                    st.session_state.draft_resumed=True
                    st.session_state.draft_finalized=False
                    st.rerun()
                if begin_col.button(
                    'Start From Beginning',
                    type='secondary',
                    width='stretch',
                ):
                    reset_session(st.session_state, reset_consent=False)
                    st.session_state.reaction_times=[]
                    st.session_state.reaction_started=None
                    st.session_state.reaction_ready=False
                    st.rerun()
    completion=assessment_completion(st.session_state)
    completed_count=sum(completion.values())
    st.progress(completed_count/len(completion))
    st.caption(f'Session Progress: {completed_count}/{len(completion)} modules completed')
    labels={
        'voice':'🎙️ Voice',
        'resting_motion':'📳 Resting motion',
        'gait':'🚶 Gait',
        'clock_drawing':'🕒 Clock drawing',
        'reaction_time':'⚡ Reaction time',
        'speech_features':'🧠 Speech features',
    }
    tabs=st.tabs([
        f"{'✓ ' if completion[key] else ''}{labels[key]}"
        for key in ('voice','resting_motion','gait','clock_drawing','reaction_time','speech_features')
    ]+['📄 Report'])
    voice_tab,motion_tab,gait_tab,clock_tab,reaction_tab,alz_tab,report_tab=tabs

    with voice_tab:
        st.subheader('Voice assessment'); st.write('Read a short neutral passage or sustain a vowel for several seconds. Use the same task when repeating sessions.')
        st.caption('Avoid saying names, addresses or other identifying information.')
        st.caption('WAV, MP3, M4A, OGG or FLAC all work — the file is converted to 16 kHz mono WAV automatically when you press Analyze.')
        rec=st.audio_input('Record a voice sample',key=f"rec{st.session_state.upload_generation}"); up=st.file_uploader('Or upload an audio file',type=['wav','mp3','m4a','ogg','flac'],key=f"wav{st.session_state.upload_generation}"); chosen=rec or up
        if st.button('Analyze voice',disabled=chosen is None):
            try:
                from modules.audio_analysis import to_standard_wav
                wav_bytes,conv=to_standard_wav(chosen.getvalue()); st.session_state.voice_wav_bytes=wav_bytes
                st.session_state.audio_result=analyze_audio_bytes(wav_bytes)
                st.session_state.pattern_result=None
                mark_assessment_changed()
                st.success('Audio converted to WAV (16 kHz mono) — acoustic features extracted.' if conv['converted'] else 'Acoustic features extracted.')
                st.rerun()
            except Exception as e:
                st.session_state.audio_result=None
                mark_assessment_changed()
                st.error(str(e))
        r=st.session_state.audio_result
        if r:
            q=r['quality']; st.metric('Recording quality',f"{q['score']}/100 · {q['label']}"); st.caption('This is technical recording quality, not a disease score.')
            x=st.columns(4)
            for col,label,key,fmt in zip(x,['Duration','RMS level','Silence','Pitch estimate'],['duration_seconds','rms','silence_ratio','pitch_median_hz'],['{:.1f} s','{:.4f}','{:.1%}','{:.1f} Hz']):
                val=r.get(key); col.metric(label,fmt.format(val) if val is not None else 'Unavailable')
            st.plotly_chart(go.Figure(go.Bar(x=list(range(1,14)),y=r['mfcc_means'])).update_layout(title='Exploratory acoustic coefficients',xaxis_title='Coefficient',yaxis_title='Mean value'),use_container_width=True)
            st.caption('Speech features are influenced by language, accent, age, task, device and environment. They are not diagnostic alone.')

        # New feature section starts here (properly indented with 8 spaces)
        st.divider()
        st.subheader('Experimental symptom-pattern analysis')
        st.write("Sustain a steady vowel such as 'ahhh' for at least 5 seconds, then run the research pattern comparison.")

        if chosen is None:
            st.caption('Record or upload a voice sample above to enable pattern analysis.')
        if st.button('Run experimental pattern analysis', disabled=chosen is None):
            try:
                from modules.praat_features import run_symptom_pattern_analysis
                from modules.audio_analysis import to_standard_wav
                src = st.session_state.get('voice_wav_bytes') or to_standard_wav(chosen.getvalue())[0]
                st.session_state.pattern_result = run_symptom_pattern_analysis(src)
                mark_assessment_changed()
            except Exception as e:
                st.session_state.pattern_result = None
                mark_assessment_changed()
                st.error(f'Pattern analysis not run: {e}')

        pr = st.session_state.get('pattern_result')
        if pr:
            if pr['recording_quality']['score'] < 60:
                st.warning('Recording quality is low; the pattern output below is unreliable. Re-record in a quiet room.')
            st.metric('Research pattern result', pr['experimental_output']['pattern_label'])
            st.caption(f"Model context: subject-level balanced accuracy {pr['model_context']['subject_level_balanced_accuracy']:.2f} "
                       f"(95% CI {pr['model_context']['ci95'][0]:.2f}–{pr['model_context']['ci95'][1]:.2f}) · internal CV only")
            with st.expander('Measurements extracted'):
                st.json({k: round(v, 5) if isinstance(v, float) else v for k, v in pr['compatible_features'].items()})
            st.error(pr['required_disclaimer'])
    with motion_tab:
        st.subheader('Resting motion'); st.write('Record a short resting sensor sample using the companion phone logger, then upload its CSV.')
        mf=st.file_uploader('Upload motion CSV',type=['csv'],key=f"motion{st.session_state.upload_generation}")
        if st.button('Analyze resting motion',disabled=mf is None):
            try:
                st.session_state.motion_result=analyze_motion(pd.read_csv(mf))
                mark_assessment_changed()
                st.success('Motion features extracted.')
                st.rerun()
            except Exception as e:
                st.session_state.motion_result=None
                mark_assessment_changed()
                st.error(str(e))
        r=st.session_state.motion_result
        if r:
            st.metric('Signal quality',f"{r['quality']['score']}/100"); a,b,c=st.columns(3); a.metric('Duration',f"{r['duration_seconds']:.1f}s"); b.metric('Sampling rate',f"{r['sample_rate_hz']:.1f}Hz"); c.metric('Dominant frequency',f"{r['dominant_frequency_hz']:.2f}Hz" if r['dominant_frequency_hz'] else 'Unavailable')
            st.plotly_chart(go.Figure(go.Scatter(x=r['frequencies_hz'],y=r['spectrum'])).update_layout(title='Exploratory motion spectrum'),use_container_width=True)
            st.caption('Frequency peaks are nonspecific and do not diagnose Parkinson’s disease.')
    with gait_tab:
        st.subheader('Gait assessment'); st.write('Only perform this if walking is safe for you. Use a level, clear path and stop if dizzy or unsteady.')
        gf=st.file_uploader('Upload gait CSV',type=['csv'],key=f"gait{st.session_state.upload_generation}")
        if st.button('Analyze gait',disabled=gf is None):
            try:
                st.session_state.gait_result=analyze_gait(pd.read_csv(gf))
                mark_assessment_changed()
                st.success('Exploratory gait features extracted.')
                st.rerun()
            except Exception as e:
                st.session_state.gait_result=None
                mark_assessment_changed()
                st.error(str(e))
        r=st.session_state.gait_result
        if r:
            a,b,c=st.columns(3); a.metric('Signal quality',f"{r['quality']['score']}/100"); b.metric('Candidate step events',r['candidate_step_events']); c.metric('Estimated cadence',f"{r['cadence_steps_per_min']:.1f}/min" if r['cadence_steps_per_min'] is not None else 'Unavailable'); st.caption(r['method_note'])
    with clock_tab:
        st.subheader('Clock-drawing activity')
        st.write('Draw a clock face, add the numbers, and set the hands to **10 past 11**. This task records coarse drawing geometry only; it does not calculate Alzheimer’s risk or provide a diagnosis.')
        st.caption('Mouse/touch canvas data has no calibrated pressure or reliable pointer timestamps. DARWIN tablet kinematics cannot be inferred from this drawing.')
        darwin_status=get_alzheimers_status()
        if darwin_status.get('available'):
            st.metric('Experimental Kinematic Similarity Indicator','Not calculated')
            st.caption('The DARWIN model is available for dataset-level research evaluation only. Its 450 tablet features do not match this single-task browser canvas, so no user-level similarity or prediction is produced.')
        else:
            st.metric('Experimental Kinematic Similarity Indicator','Unavailable')
            st.caption(darwin_status.get('message','DARWIN research model is not installed.'))
        if st.button(
            'Load Sample DARWIN Test Profile',
            type='secondary',
            disabled=not darwin_status.get('available'),
            key='load_darwin_sample_profile',
        ):
            try:
                st.session_state.darwin_sample_profile=load_darwin_sample_profile()
                st.rerun()
            except (RuntimeError, KeyError) as exc:
                st.error(f'Could not load the DARWIN cohort example: {exc}')
        sample_profile=st.session_state.get('darwin_sample_profile')
        if sample_profile:
            st.info(sample_profile['notice'])
            st.dataframe(
                pd.DataFrame([
                    {'Feature':humanize_feature_code(name),'Cohort median':value}
                    for name,value in sample_profile['features'].items()
                ]),
                hide_index=True,
                alt='Dataset-wide median values for selected DARWIN research features',
            )
        canvas=st_canvas(fill_color='rgba(13,148,136,0.12)',stroke_width=3,stroke_color='#0d192b',background_color='#ffffff',height=340,width=520,drawing_mode='freedraw',key=f"clock_canvas_{st.session_state.upload_generation}")
        drawing_json=getattr(canvas,'json_data',None)
        has_strokes=isinstance(drawing_json,dict) and bool(drawing_json.get('objects'))
        if st.button('Analyze drawing',disabled=not has_strokes,key='analyze_clock_drawing'):
            try:
                drawing_features=extract_clock_drawing_features(drawing_json)
                selected_darwin_features=(
                    (darwin_status.get('metadata') or {})
                    .get('feature_selection',{})
                    .get('selected_features',[])
                )
                darwin_compatibility=assess_darwin_feature_compatibility(
                    drawing_features,
                    selected_darwin_features,
                )
                st.session_state.clock_result={
                    'completed':True,
                    'task':'Draw a clock showing 11:10',
                    'saved_at':datetime.now(timezone.utc).isoformat(),
                    'scoring':'No clinical or kinematic similarity score is produced.',
                    'features':drawing_features,
                    'darwin_model_compatibility':darwin_compatibility,
                }
                mark_assessment_changed()
                st.success('Drawing geometry recorded for this session.')
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if st.session_state.clock_result:
            clock_features=st.session_state.clock_result.get('features')
            if clock_features:
                st.subheader('Measured drawing descriptors')
                metric_cols=st.columns(4)
                metric_cols[0].metric('Strokes',clock_features['stroke_count'])
                metric_cols[1].metric('Approx. path length',f"{clock_features['approx_path_length_px']:.1f} px")
                metric_cols[2].metric('Canvas area used',f"{clock_features['bounding_box_area_fraction']:.1%}")
                turn=clock_features['mean_direction_change_degrees']
                metric_cols[3].metric('Mean direction change',f'{turn:.1f}°' if turn is not None else 'Unavailable')
                st.subheader('Measured feature breakdown')
                spatial_footprint=pd.DataFrame(
                    {'Canvas fraction used':[clock_features['bounding_box_area_fraction']]},
                    index=['Bounding-box spatial footprint'],
                )
                st.bar_chart(
                    spatial_footprint,
                    horizontal=True,
                    alt='Bounding-box spatial footprint as the fraction of canvas used',
                )
                timing_features=pd.DataFrame([
                    {'Feature':'Drawing duration','Value':clock_features['drawing_duration_seconds']},
                    {'Feature':'Air / pause time','Value':clock_features['pause_or_air_time_seconds']},
                    {'Feature':'Average stroke speed','Value':clock_features['average_stroke_speed_px_per_second']},
                    {'Feature':'Pressure','Value':clock_features['pressure']},
                    {'Feature':'Jerk','Value':clock_features['jerk']},
                ])
                timing_features['Value']=timing_features['Value'].map(
                    lambda value:'Unavailable from this canvas' if value is None else value
                )
                st.dataframe(timing_features,hide_index=True,alt='Clock-drawing timing and pressure features unavailable from the canvas export')
                compatibility=st.session_state.clock_result.get('darwin_model_compatibility') or {}
                st.metric('Experimental Kinematic Similarity Indicator','Not calculated')
                st.caption(compatibility.get('message','No compatible DARWIN feature schema is available.'))
                st.info(clock_features['timing_limitation'])
            else:
                st.json(st.session_state.clock_result)
    with reaction_tab:
        st.subheader('Visual reaction-time activity'); st.write('Press Start, wait for the green signal, then tap as quickly as you comfortably can. Complete five trials.')
        if st.button('Start / reset reaction trial',type='primary'):
            st.session_state.reaction_started=time.perf_counter(); st.session_state.reaction_ready=False; st.session_state.reaction_delay=random.uniform(1.5,3.5)
        if st.session_state.reaction_started is not None:
            elapsed=time.perf_counter()-st.session_state.reaction_started
            if elapsed>=st.session_state.get('reaction_delay',2): st.session_state.reaction_ready=True
            if st.session_state.reaction_ready:
                st.success('🟢 TAP NOW')
                if st.button('I saw it — tap!',key=f"react{len(st.session_state.reaction_times)}"):
                    ms=(time.perf_counter()-st.session_state.reaction_started)*1000-st.session_state.get('reaction_delay',2)*1000
                    st.session_state.reaction_times.append(max(0,round(ms,1))); mark_assessment_changed(); st.session_state.reaction_started=None; st.session_state.reaction_ready=False; st.rerun()
            else: st.info('Wait for the green signal…'); time.sleep(.25); st.rerun()
        if st.session_state.reaction_times:
            summary=reaction_summary(st.session_state.reaction_times); a,b,c=st.columns(3); a.metric('Trials',summary['trials']); b.metric('Median response',f"{summary['median_ms']} ms"); c.metric('Response variability',f"{summary['variability_ms']} ms"); st.caption('Timing is affected by browser and device latency; this is not a clinical cognitive test.')
            if st.button('Clear reaction trials'): st.session_state.reaction_times=[]; mark_assessment_changed(); st.rerun()
    with alz_tab:
        st.subheader("Alzheimer's speech features — research only")
        st.write('Describe your morning or a recent event in a few sentences (at least 10 words), or paste what you said in the voice task above.')
        st.caption('No trained model exists for this pathway yet: you receive measurements only, never a score. Features require a transcript; audio adds timing features.')
        transcript = st.text_area('Transcript', height=120, key=f"alz_transcript_{st.session_state.upload_generation}",
            placeholder='For example: I woke up at seven, made coffee, and walked the dog around the block before breakfast...')
        transcript_word_count=len(re.findall(r"[a-z'\-]+",transcript.lower()))
        if transcript_word_count < 10:
            st.info(f'Enter at least 10 words to enable extraction ({transcript_word_count}/10 entered).')
        alz_audio = st.file_uploader('Optional: matching audio file (any format — auto-converted to WAV)', type=['wav','mp3','m4a','ogg','flac'], key=f"alz_audio_{st.session_state.upload_generation}")
        if st.button('Extract speech features', disabled=transcript_word_count < 10):
            try:
                from modules.alzheimers_model import run_alzheimers_feature_report
                from modules.audio_analysis import to_standard_wav
                if alz_audio is not None:
                    wb, _ = to_standard_wav(alz_audio.getvalue())
                    import tempfile
                    with tempfile.TemporaryDirectory(prefix='neuronexus-speech-') as tmp_dir:
                        audio_path=Path(tmp_dir)/'matching.wav'
                        audio_path.write_bytes(wb)
                        st.session_state.alz_result = run_alzheimers_feature_report(transcript, str(audio_path))
                else:
                    st.session_state.alz_result = run_alzheimers_feature_report(transcript)
                mark_assessment_changed()
                st.rerun()
            except Exception as e:
                st.session_state.alz_result = None; mark_assessment_changed(); st.error(str(e))
        ar = st.session_state.get('alz_result')
        if ar:
            feats = ar['features']
            c = st.columns(4)
            for col, (label, key, fmt) in zip(c, [('Words', 'n_words', '{:.0f}'), ('Type-token ratio', 'type_token_ratio', '{:.3f}'),
                                                  ('Pronoun ratio', 'pronoun_ratio', '{:.3f}'), ('Idea density', 'idea_density', '{:.3f}')]):
                col.metric(label, fmt.format(feats[key]))
            if ar['audio_used']:
                c2 = st.columns(3)
                for col, (label, key, fmt) in zip(c2, [('Duration', 'duration_seconds', '{:.1f} s'),
                                                       ('Words per minute', 'words_per_minute', '{:.1f}'), ('Pause ratio', 'pause_ratio', '{:.1%}')]):
                    if key in feats: col.metric(label, fmt.format(feats[key]))
            with st.expander('All extracted features'):
                st.json({k: (round(v, 5) if isinstance(v, float) else v) for k, v in feats.items()})
            st.info(ar['note'])

    with report_tab:
        st.subheader('Session summary'); report=build_report(audio_result=st.session_state.audio_result,motion_result=st.session_state.motion_result,gait_result=st.session_state.gait_result,mode='Individual')
        report['cognitive_tasks']={'clock_drawing':st.session_state.clock_result,'reaction_time':reaction_summary(st.session_state.reaction_times)}
        if st.session_state.clock_result and st.session_state.clock_result.get('features'):
            report['clock_drawing_measurements']=st.session_state.clock_result['features']
        _ar=st.session_state.get('alz_result')
        if _ar: report['alzheimers_speech_features']=_ar['features']
        pattern_result=st.session_state.get('pattern_result')
        if pattern_result:
            report['voice_pattern_features']=pattern_result.get('compatible_features') or {}
        if st.session_state.get('darwin_sample_profile'):
            report['darwin_sample_profile']=st.session_state.darwin_sample_profile
        if st.session_state.get('consent_granted'):
            report['consent_flags']={
                'assessment':True,
                'optional_profile_context':bool(st.session_state.get('baseline_context')),
                'caregiver_observations':bool(st.session_state.get('caregiver_notes_consent')),
            }
            if st.session_state.get('baseline_context'):
                report['participant_context']=st.session_state.baseline_context
            if st.session_state.get('caregiver_notes_consent'):
                notes=st.session_state.get('caregiver_observation_notes') or []
                if notes:
                    report['caregiver_observations']=notes
            report['assisted_session_mode']=bool(st.session_state.get('assisted_session_mode'))
        st.info('This report contains measurements and task completion information, not a diagnosis or disease-risk estimate.')
        st.json(report)
        st.download_button('Download PDF report',make_pdf(user['name'],report,get_registry()),file_name='neuronexus_report.pdf',mime='application/pdf')
        st.download_button('Download JSON report',json.dumps(report,indent=2),file_name='neuronexus_report.json',mime='application/json')
        if st.session_state.get('local_history_enabled'):
            if st.button('Save session to my dashboard',type='primary'):
                save_session(user['id'],report)
                delete_draft(user['id'])
                st.session_state.draft_finalized=True
                st.success('Finalized session report saved to your account.')
        else:
            st.button('Save session to my dashboard',type='primary',disabled=True)
            st.caption('Enable saved session history in My profile to save this finalized report.')
    st.divider()
    if st.button('Clear current screening session'):
        reset_session(st.session_state, reset_consent=False); st.session_state.reaction_started=None; st.session_state.reaction_ready=False; st.rerun()

st.divider(); st.caption('NeuroNexus 2.0 prototype · Not a medical device · Experimental measurements only · Seek a qualified clinician for health concerns.')
