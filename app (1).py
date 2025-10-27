import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -------------------------
# CONFIG / CREDENTIALS
# -------------------------
CSV_PATH = "hospital_processed_with_forecasts.csv"  # make sure this file is in same folder

# Demo credentials (change these for deployment)
CREDENTIALS = {
    "admin": "admin123",   # full access
    "client": "client123"  # limited view if needed
}

# -------------------------
# HELPERS
# -------------------------
def load_data(path):
    df = pd.read_csv(path)
    # Normalize date columns
    if 'Admission_Date' in df.columns:
        df['Admission_Date'] = pd.to_datetime(df['Admission_Date'], errors='coerce')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Some datasets use 'Length_of_Stay_Days' or 'Stay_Duration'
    if 'Length_of_Stay_Days' in df.columns:
        df['Length_of_Stay_Days'] = pd.to_numeric(df['Length_of_Stay_Days'], errors='coerce')
    if 'Stay_Duration' in df.columns:
        df['Stay_Duration'] = pd.to_numeric(df['Stay_Duration'], errors='coerce')
    # Ensure occupancy rate numeric
    if 'Bed_Occupancy_Rate' in df.columns:
        df['Bed_Occupancy_Rate'] = pd.to_numeric(df['Bed_Occupancy_Rate'], errors='coerce')
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df

def kpi_card(col, title, value, delta=None, help_text=None):
    with col:
        st.markdown(f"{title}")
        st.markdown(f"<h2 style='margin:5px'>{value}</h2>", unsafe_allow_html=True)
        if delta is not None:
            st.caption(delta)
        if help_text:
            st.info(help_text)

# -------------------------
# STYLING (Login Page + App)
# -------------------------
st.set_page_config(page_title="Hospital Admissions & Bed Occupancy", layout="wide", initial_sidebar_state="expanded")

LOGIN_CSS = """
<style>
/* background */
[data-testid="stAppViewContainer"]{
  background: linear-gradient(90deg, #f7f9fc 0%, #eef6ff 100%);
}

/* center card */
.login-card {
  background: white;
  padding: 32px;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
}

/* inputs */
.stTextInput>div>div>input {
  border-radius: 8px;
}

/* buttons */
.stButton>button {
  background: linear-gradient(90deg,#0f62fe,#3ddc97);
  color: white;
  border: none;
  padding: 8px 14px;
  border-radius: 8px;
}
</style>
"""

st.markdown(LOGIN_CSS, unsafe_allow_html=True)

# -------------------------
# LOGIN / AUTH
# -------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None

def do_logout():
    st.session_state.logged_in = False
    st.session_state.user = None
    st.rerun()

def show_login():
    # Centered login card
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)
        st.image(
            "https://raw.githubusercontent.com/microsoft/PowerBI-visuals/master/resources/powerbi_icon.png",
            width=72,
        )
        st.markdown("<h2 style='margin-bottom:2px'>Hospital Dashboard Login</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#6b7280;margin-top:0'>Sign in to access hospital analytics and forecasts</p>", unsafe_allow_html=True)

        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        remember = st.checkbox("Remember me")
        if st.button("Sign in"):
            if username in CREDENTIALS and CREDENTIALS[username] == password:
                st.session_state.logged_in = True
                st.session_state.user = username
                st.success(f"Welcome back, {username}!")
                st.rerun()
            else:
                st.error("Invalid credentials. If this is a demo, try 'admin/admin123' or 'client/client123'.")

        st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state.logged_in:
    show_login()
    st.stop()

# -------------------------
# APP (After login)
# -------------------------
df = load_data(CSV_PATH)

# Sidebar: Logout + navigation
with st.sidebar:
    st.markdown("### ⚕ Hospital Dashboard")
    st.write(f"*User:* {st.session_state.user}")
    st.button("Logout", on_click=do_logout)
    st.markdown("---")
    page = st.radio("Navigate", ["Overview", "Department Analysis", "Patient Demographics", "Forecasts", "Raw Data"])
    st.markdown("---")
    st.markdown("Built with 💙 • Streamlit + Plotly")

# -------------------------
# PREPARE AGGREGATIONS
# -------------------------
# Primary date for actuals
if 'Admission_Date' in df.columns:
    main_date_col = 'Admission_Date'
elif 'Date' in df.columns:
    main_date_col = 'Date'
else:
    main_date_col = None

if main_date_col:
    df = df.dropna(subset=[main_date_col])  # drop rows without date for time-series
    df[main_date_col] = pd.to_datetime(df[main_date_col], errors='coerce')
    df['YearMonth'] = df[main_date_col].dt.to_period('M').astype(str)
else:
    df['YearMonth'] = df['Year'].astype(str) + "-" + df['Month'].astype(str).str.zfill(2)

# Admissions actual
admissions_ts = df.groupby(main_date_col).agg(Admissions=("Patient_ID", "count")).reset_index().sort_values(main_date_col)

# Forecasts: group by Date column if Admissions_Forecast exists
has_forecast = 'Admissions_Forecast' in df.columns and df['Admissions_Forecast'].notna().any()
if has_forecast:
    forecast_df = df.dropna(subset=['Admissions_Forecast', 'Date']).copy()
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'], errors='coerce')
    forecast_ts = forecast_df.groupby('Date').agg(
        Admissions_Forecast=('Admissions_Forecast', 'sum'),
        Admissions_Forecast_Lower=('Admissions_Forecast_Lower', 'sum'),
        Admissions_Forecast_Upper=('Admissions_Forecast_Upper', 'sum'),
        Occupancy_Forecast=('Occupancy_Forecast', 'sum')
    ).reset_index().sort_values('Date')
else:
    forecast_ts = pd.DataFrame(columns=['Date','Admissions_Forecast'])

# Occupancy actual
if 'Bed_Occupancy_Rate' in df.columns:
    occupancy_ts = df.groupby(main_date_col).agg(Occupancy_Rate=('Bed_Occupancy_Rate','mean')).reset_index().sort_values(main_date_col)
else:
    occupancy_ts = pd.DataFrame()

# Useful aggregations
total_admissions = df['Patient_ID'].nunique() if df['Patient_ID'].dtype == object else df['Patient_ID'].count()
avg_bed_occ = df['Bed_Occupancy_Rate'].mean() if 'Bed_Occupancy_Rate' in df.columns else np.nan
avg_stay = df['Length_of_Stay_Days'].mean() if 'Length_of_Stay_Days' in df.columns else (df['Stay_Duration'].mean() if 'Stay_Duration' in df.columns else np.nan)

# forecast next months sum (Date > max actual)
next_forecast_sum = None
if has_forecast and not forecast_ts.empty:
    last_actual = admissions_ts[main_date_col].max()
    future_forecasts = forecast_ts[forecast_ts['Date'] > last_actual]
    next_forecast_sum = future_forecasts['Admissions_Forecast'].sum()

# -------------------------
# PAGES
# -------------------------
if page == "Overview":
    st.markdown("<h1 style='margin-bottom:8px'>🏥 Executive Summary</h1>", unsafe_allow_html=True)
    st.markdown("Quick snapshot of hospital activity and capacity.")

    # KPI row
    k1, k2, k3, k4 = st.columns([1.2,1.2,1.2,1.2])
    kpi_card(k1, "Total Admissions", f"{int(total_admissions):,}", help_text="Unique patient entries in dataset")
    kpi_card(k2, "Avg Bed Occupancy %", f"{avg_bed_occ:.2f} %", help_text="Mean bed occupancy across the dataset")
    kpi_card(k3, "Avg Stay Duration (days)", f"{avg_stay:.2f}", help_text="Average length of stay")
    if next_forecast_sum is not None:
        kpi_card(k4, "Forecasted Admissions (upcoming)", f"{next_forecast_sum:,.0f}", help_text="Sum of forecasted admissions for future dates")
    else:
        kpi_card(k4, "Forecasted Admissions (sample)", "N/A", help_text="No forecast rows available")

    st.markdown("---")

    # Admissions trend (actual vs forecast)
    st.subheader("Admissions Trend (Actual vs Forecast)")
    fig = go.Figure()
    if not admissions_ts.empty:
        fig.add_trace(go.Scatter(x=admissions_ts[main_date_col], y=admissions_ts['Admissions'],
                                 mode='lines+markers', name='Actual Admissions'))
    if has_forecast and not forecast_ts.empty:
        fig.add_trace(go.Scatter(x=forecast_ts['Date'], y=forecast_ts['Admissions_Forecast'],
                                 mode='lines+markers', name='Forecast Admissions', line=dict(dash='dash')))
        # confidence band if available
        if 'Admissions_Forecast_Lower' in forecast_ts.columns and 'Admissions_Forecast_Upper' in forecast_ts.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_ts['Date'], forecast_ts['Date'][::-1]]),
                y=pd.concat([forecast_ts['Admissions_Forecast_Upper'], forecast_ts['Admissions_Forecast_Lower'][::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name='Forecast range'
            ))
    fig.update_layout(margin=dict(t=8,b=8,l=8,r=8), xaxis_title="Date", yaxis_title="Admissions")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # Occupancy trend
    st.subheader("Bed Occupancy Trend (Actual vs Forecast)")
    fig2 = go.Figure()
    if not occupancy_ts.empty:
        fig2.add_trace(go.Scatter(x=occupancy_ts[main_date_col], y=occupancy_ts['Occupancy_Rate'], mode='lines+markers', name='Actual Occupancy %'))
    if has_forecast and not forecast_ts.empty and 'Occupancy_Forecast' in forecast_ts.columns:
        fig2.add_trace(go.Scatter(x=forecast_ts['Date'], y=forecast_ts['Occupancy_Forecast'], mode='lines+markers', name='Forecast Occupancy'))
    fig2.update_layout(margin=dict(t=8,b=8,l=8,r=8), xaxis_title="Date", yaxis_title="Occupancy (%)")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Admissions by Department")
    dept_counts = df.groupby('Department').agg(Admissions=('Patient_ID','count')).reset_index().sort_values('Admissions', ascending=False)
    fig3 = px.bar(dept_counts, x='Department', y='Admissions', text='Admissions')
    fig3.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig3, use_container_width=True)

elif page == "Department Analysis":
    st.markdown("<h1>📊 Department Analysis</h1>", unsafe_allow_html=True)
    # department selector
    depts = df['Department'].dropna().unique().tolist()
    selected_depts = st.multiselect("Select Departments", options=depts, default=depts)

    df_dept = df[df['Department'].isin(selected_depts)].copy()

    # Admissions by Department & Type (stacked)
    st.subheader("Admissions by Department and Type")
    dept_type = df_dept.groupby(['Department','Admission_Type']).agg(Count=('Patient_ID','count')).reset_index()
    fig = px.bar(dept_type, x='Department', y='Count', color='Admission_Type', barmode='group', text='Count')
    st.plotly_chart(fig, use_container_width=True)

    # Admission Type by Department (stacked normalized)
    st.subheader("Admission Type Distribution by Department")
    dept_pivot = dept_type.pivot(index='Department', columns='Admission_Type', values='Count').fillna(0)
    fig2 = px.bar(dept_pivot.reset_index(), x='Department', y=dept_pivot.columns.tolist(), title="Admission Type by Department")
    st.plotly_chart(fig2, use_container_width=True)

    # Avg stay duration by dept & type
    st.subheader("Average Stay Duration by Department & Type")
    if 'Stay_Duration' in df.columns:
        stay_table = df_dept.groupby(['Department','Admission_Type']).agg(Avg_Stay=('Stay_Duration','mean')).reset_index()
    else:
        stay_table = df_dept.groupby(['Department','Admission_Type']).agg(Avg_Stay=('Length_of_Stay_Days','mean')).reset_index()
    st.dataframe(stay_table.style.format({"Avg_Stay":"{:.2f}"}))

    # Occupancy Trend by Department
    st.subheader("Occupancy Trend by Department")
    if 'Bed_Occupancy_Rate' in df.columns:
        occ_dept = df_dept.groupby([main_date_col,'Department']).agg(Occupancy=('Bed_Occupancy_Rate','mean')).reset_index()
        fig_occ = px.line(occ_dept, x=main_date_col, y='Occupancy', color='Department', markers=True)
        st.plotly_chart(fig_occ, use_container_width=True)
    else:
        st.info("No Bed_Occupancy_Rate column found.")

elif page == "Patient Demographics":
    st.markdown("<h1>👥 Patient Demographics & Trends</h1>", unsafe_allow_html=True)
    st.subheader("Gender Distribution")
    if 'Gender' in df.columns:
        gender_counts = df['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender','Count']
        figg = px.pie(gender_counts, names='Gender', values='Count', hole=0.45)
        st.plotly_chart(figg, use_container_width=True)
    else:
        st.info("No Gender column found.")

    st.subheader("Patients by Age Group")
    if 'Age_Group' in df.columns:
        age_counts = df['Age_Group'].value_counts().reset_index()
        age_counts.columns = ['Age_Group','Count']
        fig_age = px.bar(age_counts, x='Age_Group', y='Count', text='Count')
        st.plotly_chart(fig_age, use_container_width=True)
    else:
        st.info("No Age_Group column found.")

    st.subheader("Avg Stay Duration by Age Group")
    if 'Age_Group' in df.columns and ('Stay_Duration' in df.columns or 'Length_of_Stay_Days' in df.columns):
        col_stay = 'Stay_Duration' if 'Stay_Duration' in df.columns else 'Length_of_Stay_Days'
        avg_stay_age = df.groupby('Age_Group').agg(Avg_Stay=(col_stay,'mean')).reset_index()
        fig_as = px.bar(avg_stay_age, x='Age_Group', y='Avg_Stay', text=avg_stay_age['Avg_Stay'].round(2))
        st.plotly_chart(fig_as, use_container_width=True)
    else:
        st.info("No stay duration column found.")

    st.subheader("Admission Type by Gender")
    if 'Admission_Type' in df.columns and 'Gender' in df.columns:
        g_pivot = df.groupby(['Gender','Admission_Type']).agg(Count=('Patient_ID','count')).reset_index()
        fig_gt = px.bar(g_pivot, x='Gender', y='Count', color='Admission_Type', barmode='group', text='Count')
        st.plotly_chart(fig_gt, use_container_width=True)
    else:
        st.info("Columns required not found.")

elif page == "Forecasts":
    st.markdown("<h1>🔮 Forecasting Dashboard</h1>", unsafe_allow_html=True)
    st.subheader("Forecast Summary")
    if has_forecast and not forecast_ts.empty:
        total_forecast = forecast_ts['Admissions_Forecast'].sum()
        st.metric("Sum of Forecasted Admissions (available rows)", f"{total_forecast:,.0f}")
        # Forecast by department if exists
        if 'Department' in df.columns and 'Admissions_Forecast' in df.columns:
            fdept = df.dropna(subset=['Admissions_Forecast']).groupby('Department').agg(Forecasted_Admissions=('Admissions_Forecast','sum')).reset_index()
            figf = px.bar(fdept, x='Department', y='Forecasted_Admissions', text='Forecasted_Admissions')
            st.plotly_chart(figf, use_container_width=True)
        # Actual vs Forecast by Year-Month
        st.subheader("Actual vs Forecast by Month")
        actual_month = df.groupby('YearMonth').agg(Actual_Admissions=('Patient_ID','count')).reset_index()
        # build forecast month aggregation (from forecast_ts)
        if not forecast_ts.empty:
            forecast_month = forecast_ts.copy()
            forecast_month['YearMonth'] = forecast_month['Date'].dt.to_period('M').astype(str)
            forecast_month_agg = forecast_month.groupby('YearMonth').agg(Forecast_Admissions=('Admissions_Forecast','sum')).reset_index()
            merged = pd.merge(actual_month, forecast_month_agg, on='YearMonth', how='outer').sort_values('YearMonth')
            figm = go.Figure()
            figm.add_trace(go.Bar(x=merged['YearMonth'], y=merged['Actual_Admissions'], name='Actual'))
            if 'Forecast_Admissions' in merged.columns:
                figm.add_trace(go.Bar(x=merged['YearMonth'], y=merged['Forecast_Admissions'], name='Forecast'))
            figm.update_layout(barmode='group', xaxis_tickangle=-45)
            st.plotly_chart(figm, use_container_width=True)
    else:
        st.info("No forecast rows present in dataset. Ensure 'Admissions_Forecast' and 'Date' columns exist and have future dates.")

elif page == "Raw Data":
    st.markdown("<h1>📂 Raw Dataset</h1>", unsafe_allow_html=True)
    st.dataframe(df.head(200))

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("© Project: Hospital Admissions & Bed Occupancy - Generated with Streamlit. Place CSV in the same folder as app.py.")
