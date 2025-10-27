#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##--------- Hospital Admission Trends & Bed Occupancy Analysis----------


# In[56]:


##-----library Installation----
# get_ipython().system('pip install prophet')


# In[57]:


##----Import Libraries----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet


# In[58]:


##----Load Dataset----

df = pd.read_csv('hospital_admissions_synthetic.csv')

df['Admission_Date'] = pd.to_datetime(df['Admission_Date'])
df['Discharge_Date'] = pd.to_datetime(df['Discharge_Date'])
df.head()


# In[59]:


df.info()


# #### As we can see that there is total 2000 rows and 12 columns in this dataset
# #### But the datatype of Admission Date and Discharge date should be in date_time
# #### Also there is no any null values

# In[60]:


##----Data Cleaning & Preprocessing----


# In[61]:


# to check duplicate values
df.duplicated().sum()


# In[62]:


# Check for missing values
print(df.isnull().sum())

# Create additional time features
df['Month'] = df['Admission_Date'].dt.month
df['Year'] = df['Admission_Date'].dt.year


# In[63]:


df.describe()#to check the max,min,avrg of numerical columns


# In[64]:


df['Gender'].value_counts() # we can see the number of male and female patients


# In[65]:


##----Exploratory Data Analysis (EDA)----


# In[66]:


##----Patients demographics-----


# In[67]:


# Age distribution
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()


# In[68]:


# Gender distribution
sns.countplot(x='Gender', data=df)
plt.title("Gender Distribution")
plt.show()


# In[69]:


# Admissions by department
plt.figure(figsize=(8,5))
sns.countplot(x='Department', data=df)
plt.title("Admissions by Department")
plt.show()


# In[70]:


sns.countplot(x='Admission_Type', data=df)
plt.title("Admissions by Type")
plt.show()


# In[71]:


# Bed Occupancy Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Bed_Occupancy_Rate'], bins=20, kde=True)
plt.title("Bed Occupancy Rate Distribution")
plt.show()


# In[72]:


# Emergency vs Scheduled
plt.figure(figsize=(6,4))
sns.countplot(x='Admission_Type', data=df)
plt.title("Emergency vs Scheduled Admissions")
plt.show()


# In[73]:


# Average length of stay by department
df.groupby('Department')['Length_of_Stay_Days'].mean().plot(kind='bar')
plt.title("Average Length of Stay by Department")
plt.show()


# In[74]:


##----Feature Engineering----


# In[75]:


# Stay duration in days
df['Stay_Duration'] = (df['Discharge_Date'] - df['Admission_Date']).dt.days

print(df[['Admission_Date', 'Discharge_Date', 'Stay_Duration']].head())


# In[76]:


# Occupancy Flag (High/Medium/Low)
def  occupancy_category(rate):
    if rate > 80:
        return "High"
    elif rate>=50:
        return "Medium"
    else:
        return "Low"

df['Occupancy_Level']=df['Bed_Occupancy_Rate'].apply(occupancy_category)
print(df[['Bed_Occupancy_Rate', 'Occupancy_Level']].head())


# In[77]:


# Age Groups
def age_group(age):
    if age < 18:
        return "Child"
    elif age < 60:
        return "Adult"
    else:
        return "Senior"

# Only if 'Age' column exists
if 'Age' in df.columns:
    df['Age_Group'] = df['Age'].apply(age_group)
    print(df[['Age', 'Age_Group']].head())


# In[78]:


## Statistical Analysis


# In[79]:


# Descriptive Statistics

print("Descriptive Statistics for Bed Occupancy Rate: ")
print(df['Bed_Occupancy_Rate'].describe())


# In[80]:


# Group wise Analysis

# Average Bed Occupancy Rate by Department
print("\nAverage Bed Occupancy Rate by Department:")
print(df.groupby('Department')['Bed_Occupancy_Rate'].mean())


# In[81]:


# Average Stay Duration by Admission Type (if Stay_Duration exists)
if 'Stay_Duration' in df.columns:
    print("\nAverage Stay Duration by Admission Type:")
    print(df.groupby('Admission_Type')['Stay_Duration'].mean())


# In[82]:


# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.select_dtypes(include=['float64','int64']).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# In[83]:


# Monthly Admission Trends

monthly = df.groupby(['Year','Month']).size()
monthly.plot(kind='line', marker='o', figsize=(8,5))
plt.title("Monthly Admissions Trend")
plt.xlabel("Year, Month")
plt.ylabel("Number of Admissions")
plt.show()


# In[84]:


##----Advance Statistical Analysis----


# In[85]:


# Chi-Square Test: Admission Type vs Department

from scipy.stats import chi2_contingency

contingency = pd.crosstab(df['Admission_Type'], df['Department'])
chi2, p, dof, expected = chi2_contingency(contingency)

print("Chi-Square Test Results:")
print("Chi2 Value:", chi2)
print("p-value:", p)

if p < 0.05:
    print("✅ Significant relationship between Admission Type and Department")
else:
    print("❌ No significant relationship found")


# In[86]:


# ANOVA: Bed Occupancy Rate across Departments

from scipy.stats import f_oneway

groups = [group['Bed_Occupancy_Rate'].values for name, group in df.groupby("Department")]
f_stat, p = f_oneway(*groups)

print("ANOVA Results:")
print("F-statistic:", f_stat)
print("p-value:", p)

if p < 0.05:
    print("✅ Significant difference in occupancy rates between departments")
else:
    print("❌ No significant difference found")


# In[87]:


df.to_csv("hospital_processed.csv", index=False)
print("New DataFrame is Saved") 


# In[88]:


##----Predictive Modeling (Forecasting)----


# In[89]:


# Admissions per month

admissions= df.groupby('Admission_Date').size().reset_index(name='Admissions')
admissions= admissions.set_index('Admission_Date').resample('ME').sum().reset_index()

# Average Bed Ocuupancy Rate per month

occupancy= df.groupby('Admission_Date')['Bed_Occupancy_Rate'].mean().reset_index()
occupancy= occupancy.set_index('Admission_Date').resample('ME').mean().reset_index()

# Print samples

print("\nSample of Monthly Admissions:")
print(admissions.head())
print("\nSample of Monthly Bed Occupancy Rate:")
print(occupancy.head())


# In[90]:


# ARIMA - Statistical Time Series Model (Auto-Regressive Integrated Moving Average)


# In[91]:


# Admissions time series

ts_adm= admissions.set_index('Admission_Date')['Admissions']

# ARIMA model
model_adm= ARIMA(ts_adm, order=(2,1,2))
fit_adm= model_adm.fit()

# Forecast next 6 Months
forecast_adm= fit_adm.forecast(steps=6)
print("\nARIMA Forecast - Admissions:")
print(forecast_adm)

# Plot
plt.figure(figsize=(10,5))
plt.plot(ts_adm, label="Actual Admissions")
plt.plot(pd.date_range(ts_adm.index[-1], periods=6, freq='ME'), forecast_adm, label="Forecast", color="red")
plt.title("Hospital Admissions Forecast (ARIMA)")
plt.legend()
plt.show()



# In[92]:


# ARIMA was trained on monthly admissions.
# Forecast shows admissions will slightly increase over the next 6 months.
# This helps hospitals plan resources (e.g., more staff during higher demand months).


# In[93]:


# Occupancy time series

ts_occ= occupancy.set_index('Admission_Date')['Bed_Occupancy_Rate']

# ARIMA model
model_occ= ARIMA(ts_occ, order=(2,1,2))
fit_occ= model_occ.fit()

# Forecast next 6 Months
forecast_occ = fit_occ.forecast(steps=6)
print("\nARIMA Forecast - Bed Occupancy Rate:")
print(forecast_occ)

# Plot
plt.figure(figsize=(10,5))
plt.plot(ts_occ, label="Actual Bed Occupancy Rate")
plt.plot(pd.date_range(ts_occ.index[-1], periods=6, freq='ME'), forecast_occ, label="Forecast", color="red")
plt.title("Bed Occupancy Rate Forecast (ARIMA)")
plt.legend()
plt.show()


# In[94]:


# ARIMA predicted bed occupancy to remain stable around ~74–75% in the next 6 months.
# This indicates that hospital beds are consistently utilized at a high level, requiring efficient resource planning.


# In[95]:


# Prophet - Machine Learning Model


# In[96]:


# Prepare admissions data

prophet_adm = admissions.rename(columns={"Admission_Date":"ds", "Admissions":"y"})

# Train Prophet model
m_adm = Prophet()
m_adm.fit(prophet_adm)

# Predict next 6 months
future_adm = m_adm.make_future_dataframe(periods=6, freq='M')
forecast_adm = m_adm.predict(future_adm)

print("\nProphet Forecast - Admissions:")
print(forecast_adm[['ds','yhat','yhat_lower','yhat_upper']].tail(6))

# Plot forecast
m_adm.plot(forecast_adm)
plt.title("Hospital Admissions Forecast (Prophet)")
plt.show()

# Trend/seasonality
m_adm.plot_components(forecast_adm)
plt.show()


# In[97]:


# Prophet forecast shows admissions will increase gradually over the next 6 months.
# Unlike ARIMA, Prophet also provides seasonality insights, making it more interpretable.


# In[98]:


# Prepare occupancy data

prophet_occ = occupancy.rename(columns={"Admission_Date":"ds", "Bed_Occupancy_Rate":"y"})

# Train Prophet model
m_occ = Prophet()
m_occ.fit(prophet_occ)

# Predict next 6 months
future_occ = m_occ.make_future_dataframe(periods=6, freq='M')
forecast_occ = m_occ.predict(future_occ)

print("\nProphet Forecast - Bed Occupancy Rate:")
print(forecast_occ[['ds','yhat','yhat_lower','yhat_upper']].tail(6))

# Plot forecast
m_occ.plot(forecast_occ)
plt.title("Bed Occupancy Rate Forecast (Prophet)")
plt.show()

# Trend/seasonality
m_occ.plot_components(forecast_occ)
plt.show()


# In[99]:


# Prophet forecast shows occupancy will remain stable (~74–75%) with minor seasonal fluctuations.
# This suggests hospital bed utilization is consistent across time.


# In[100]:


#----Model Evaluation (ARIMA VS Prophet)---


# In[101]:


# import libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# In[102]:


# 1. Admissions Evaluation

# Actual last 6 months admissions
y_true_adm = ts_adm[-6:]

# ARIMA in-sample predictions for last 6 months
y_pred_arima_adm = fit_adm.predict(start=len(ts_adm)-6, end=len(ts_adm)-1)

# Prophet predictions on same actual dates
prophet_adm_full = m_adm.predict(prophet_adm[['ds']])   # predictions on training dates
y_pred_prophet_adm = prophet_adm_full.set_index('ds').loc[y_true_adm.index, 'yhat']

# Metrics
print("\n Admissions Forecast Evaluation:")
print("ARIMA MAE:", mean_absolute_error(y_true_adm, y_pred_arima_adm))
print("ARIMA RMSE:", np.sqrt(mean_squared_error(y_true_adm, y_pred_arima_adm)))
print("Prophet MAE:", mean_absolute_error(y_true_adm, y_pred_prophet_adm))
print("Prophet RMSE:", np.sqrt(mean_squared_error(y_true_adm, y_pred_prophet_adm)))


# In[103]:


# 2. Occupancy Evaluation

# Actual last 6 months
y_true_occ = ts_occ[-6:]

# ARIMA in-sample predictions for last 6 months
y_pred_arima_occ = fit_occ.predict(start=len(ts_occ)-6, end=len(ts_occ)-1)

# Prophet predictions on same actual dates
prophet_occ_full = m_occ.predict(prophet_occ[['ds']])   # predictions on training dates
y_pred_prophet_occ = prophet_occ_full.set_index('ds').loc[y_true_occ.index, 'yhat']

# Metrics
print("\n Bed Occupancy Forecast Evaluation:")
print("ARIMA MAE:", mean_absolute_error(y_true_occ, y_pred_arima_occ))
print("ARIMA RMSE:", np.sqrt(mean_squared_error(y_true_occ, y_pred_arima_occ)))
print("Prophet MAE:", mean_absolute_error(y_true_occ, y_pred_prophet_occ))
print("Prophet RMSE:", np.sqrt(mean_squared_error(y_true_occ, y_pred_prophet_occ)))


# In[104]:


# For both metrics (Admissions and Bed Occupancy Rate), Prophet gave lower errors than ARIMA.
# Therefore, Prophet is the best-performing model in your project.


# In[105]:


# --- Export Admissions Forecast ---
admissions_forecast = forecast_adm[['ds','yhat','yhat_lower','yhat_upper']].copy()
admissions_forecast.columns = ['Date','Admissions_Forecast','Admissions_Forecast_Lower','Admissions_Forecast_Upper']

# --- Export Occupancy Forecast ---
occupancy_forecast = forecast_occ[['ds','yhat','yhat_lower','yhat_upper']].copy()
occupancy_forecast.columns = ['Date','Occupancy_Forecast','Occupancy_Forecast_Lower','Occupancy_Forecast_Upper']

# --- Merge forecasts into one frame ---
forecast_merged = pd.merge(admissions_forecast, occupancy_forecast, on="Date", how="outer")

# --- Merge with historical processed dataset ---
df_export = pd.merge(df, forecast_merged, left_on="Admission_Date", right_on="Date", how="left")

# --- Save to CSV ---
df_export.to_csv("hospital_processed_with_forecasts.csv", index=False)
print("Final dataset with features + forecasts exported!")


# In[ ]:


# get_ipython().system('jupyter nbconvert --to script "Minorproject (3).ipynb"')


# In[106]:


# get_ipython().system('mv "Minorproject (3).py" app.py')


# In[107]:
import streamlit as st

# --- SIMPLE LOGIN + LOGOUT SYSTEM ---
USERNAME = "admin"     # change this
PASSWORD = "1234"      # change this

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Logout function
def logout():
    st.session_state.logged_in = False
    st.success("You have been logged out.")
    st.rerun()

# Login form
if not st.session_state.logged_in:
    st.title("🔐 Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()  # Stop rest of the app until login
else:
    # --- Show Dashboard after Login ---
    st.sidebar.success(f"Logged in as: {USERNAME}")
    st.sidebar.button("Logout", on_click=logout)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Hospital Admission Trends And Bed Occupancy Analysis")
st.sidebar.header("Dashboard Controls")

# Example: load dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data", df.head())

    st.write("### Summary Statistics")
    st.write(df.describe())

    st.write("### Data Visualization")
    column = st.selectbox("Select column for distribution plot:", df.columns)
    fig, ax = plt.subplots()
    sns.histplot(df[column], kde=True, ax=ax)
    st.pyplot(fig)


# In[ ]:






