import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_selection import VarianceThreshold

# ============ 설정 ============ #
SAMPLE_SIZE = 2000  # 샘플 크기
DELIMS = ['_', '-', '.']  # 그룹핑 구분자
# ============================== #

# 세션 상태 초기화
if 'selected_cols' not in st.session_state:
    st.session_state.selected_cols = []
if 'show_eda' not in st.session_state:
    st.session_state.show_eda = False

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# 클릭 콜백
def on_column_click(col):
    st.session_state.selected_cols = [col]
    st.session_state.show_eda = True

# 뒤로가기 콜백
def to_list():
    st.session_state.show_eda = False
    st.session_state.selected_cols = []

# 그룹핑 함수
def get_prefix(col):
    for d in DELIMS:
        if d in col:
            return col.split(d)[0]
    return col

# ---------------- App ---------------- #
st.title("📊 CSV EDA Dashboard with Feature Filters")

# --- 사이드바: 데이터 업로드 & 필터 설정 ---
st.sidebar.header("⚙️ Settings")
uploaded = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])
if not uploaded:
    st.info("먼저 CSV 파일을 업로드하세요.")
    st.stop()

df = load_data(uploaded)

miss_th = st.sidebar.slider("결측 비율 임계치", 0.0, 1.0, 0.3, step=0.05)
var_th  = st.sidebar.slider("분산 임계치 (숫자형)", 0.0, 1.0, 0.01, step=0.01)
corr_th = st.sidebar.slider("상관 계수 임계치", 0.0, 1.0, 0.8, step=0.05)

# 컬럼 타입 분류
d_numeric     = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
d_categorical = [c for c in df.columns if c not in d_numeric]

# --- 필터 계산 ---
# 1. 결측률 기준
miss_ratio = df.isna().mean()
high_missing = list(miss_ratio[miss_ratio >= miss_th].index)

# 2. 분산 기준 (숫자형)
selector = VarianceThreshold(threshold=var_th)
selector.fit(df[d_numeric].fillna(0))
low_variance = [c for c, v in zip(d_numeric, selector.variances_) if v <= var_th]

# 3. 상관 기준 (숫자형)
corr = df[d_numeric].corr().abs()
high_corr_cols = set()
for i in range(len(d_numeric)):
    for j in range(i+1, len(d_numeric)):
        if corr.iloc[i,j] >= corr_th:
            high_corr_cols.add(d_numeric[i])
            high_corr_cols.add(d_numeric[j])
high_corr_cols = list(high_corr_cols)

# --- 목록 및 EDA 분기 ---
if not st.session_state.show_eda:
    st.subheader("🔍 Feature Filters")
    tab1, tab2, tab3 = st.tabs(["Missing", "Low Variance", "High Correlation"])
    # Missing
    with tab1:
        st.markdown(f"**결측 비율 ≥ {miss_th:.0%}**")
        for c in high_missing or ["없음"]:
            if c != "없음":
                st.button(c, key=f"miss_{c}", on_click=on_column_click, args=(c,))
            else:
                st.write(c)
    # Variance
    with tab2:
        st.markdown(f"**분산 ≤ {var_th}**")
        for c in low_variance or ["없음"]:
            if c != "없음":
                st.button(c, key=f"var_{c}", on_click=on_column_click, args=(c,))
            else:
                st.write(c)
    # Correlation
    with tab3:
        st.markdown(f"**절대 상관 ≥ {corr_th}**")
        for c in high_corr_cols or ["없음"]:
            if c != "없음":
                st.button(c, key=f"corr_{c}", on_click=on_column_click, args=(c,))
            else:
                st.write(c)

else:
    st.subheader("📈 EDA View")
    st.button("🔙 Back to Filters", on_click=to_list)
    for col in st.session_state.selected_cols:
        s = df[col]
        miss = s.isna().mean()
        uniq = s.nunique(dropna=True)
        miss_color = 'red' if miss >= miss_th else 'black'
        uniq_color = 'red' if uniq == 1 else 'black'
        st.markdown(f"## {col}")
        st.markdown(
            f"- Missing: <span style='color:{miss_color}'>{miss:.2%}</span>  "
            f"- Unique: <span style='color:{uniq_color}'>{uniq}</span>", unsafe_allow_html=True
        )
        if show_graphs and pd.api.types.is_numeric_dtype(s):
            sample = s.dropna().sample(n=min(len(s.dropna()), SAMPLE_SIZE), random_state=1)
            fig = px.histogram(sample, nbins=20, title="Distribution")
            st.plotly_chart(fig, use_container_width=True)
        if show_stats:
            st.write(s.describe())
