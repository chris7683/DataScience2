import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@st.cache_data
def load_data_and_create_figure():
    df1 = pd.read_csv('ds1new.csv')
    df2 = pd.read_csv('ds2new.csv')

    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=(
            'Age and yearly salary distribution',
            'Job role and monthly rate',
            'Monthly hours and satisfaction level',
            'Time spent at company and salary',
            'DF1 Correlation Matrix',
            'DF2 Correlation Matrix'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )

    fig.add_trace(
        go.Scatter(
            x=df1['Age'],
            y=df1['Income_per_Year'],
            mode='markers',
            marker=dict(color='orange')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=df1['JobRole'],
            y=df1['MonthlyRate'],
            name='rate compared to job role',
            marker=dict(color='blue')
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=df2['average_montly_hours'],
            y=df2['satisfaction_level'],
            mode='markers',
            marker=dict(color='green')
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=df2['salary'],
            y=df2['time_spend_company'],
            marker=dict(color='purple')
        ),
        row=2, col=2
    )


 
    fig.update_layout(
        height=1300,
        width=1300,
        title_text="Univariate and Multivariate Analysis",
        title_x=0.5,
        showlegend=False,
        barmode='group',
        margin=dict(l=150, r=50, t=100, b=80)
    )

    fig.update_xaxes(tickangle=45, row=3, col=1)
    fig.update_xaxes(tickangle=45, row=3, col=2)

    return (df1, df2, fig)

# ---- Load Models and Data ----
df1, df2, fig = load_data_and_create_figure()
model1 = joblib.load('ds1cls.pkl')  # DF1 - Classification
model2 = joblib.load('ds1reg.pkl')  # DF1 - Regression
model3 = joblib.load('ds2cls.pkl')  # DF2 - Classification
model4 = joblib.load('ds2reg.pkl')  # DF2 - Regression

# ---- App UI ----
st.title('HR Predictions')
st.subheader('Exploratory Data Analysis')
st.plotly_chart(fig, use_container_width=True)

# ---- Prediction Section ----
st.subheader('Model Predictions')
col1, col2 = st.columns(2)

with col1:
    st.markdown("### DF1: Predict Attrition")
    age1 = st.slider("Age", 18, 60, key="age1")
    job_level1 = st.selectbox("Job Level", [1, 2, 3, 4, 5], key="job_level1")
    monthly_income1 = st.number_input("Monthly Income", 1000, 20000, key="income1")
    total_working_years1 = st.slider("Total Working Years", 0, 40, key="work_years1")
    overtime1 = st.selectbox("OverTime", ["No", "Yes"], key="overtime1")
    overtime1 = 1 if overtime1 == "Yes" else 0

with col2:
    st.markdown("### DF1: Predict Monthly Income")
    age2 = st.slider("Age", 18, 60, 30, key="age2")
    distance2 = st.slider("Distance From Home", 0, 50, 10, key="dist2")
    education2 = st.selectbox("Education", [1, 2, 3, 4, 5], key="edu2")
    job_level2 = st.selectbox("Job Level", [1, 2, 3, 4, 5], key="job_level2")
    years_at_company2 = st.slider("Years at Company", 0, 40, key="years2")

col3, col4 = st.columns(2)

with col3:
    st.markdown("### DF2: Predict Leaving")
    satisfaction3 = st.slider("Satisfaction Level", 0.0, 1.0, 0.5, key="sat3")
    evaluation3 = st.slider("Last Evaluation", 0.0, 1.0, 0.5, key="eval3")
    hours3 = st.slider("Avg Monthly Hours", 50, 350, key="hours3")
    time_spend3 = st.slider("Time at Company", 0, 20, key="time3")
    salary3 = st.selectbox("Salary Level", [0, 1, 2], key="sal3")

with col4:
    st.markdown("### DF2: Predict Workload Intensity")
    satisfaction4 = st.slider("Satisfaction Level", 0.0, 1.0, 0.5, key="sat4")
    evaluation4 = st.slider("Last Evaluation", 0.0, 1.0, 0.5, key="eval4")
    time_spend4 = st.slider("Time at Company", 0, 20, 3, key="time4")
    salary4 = st.selectbox("Salary Level", [0, 1, 2], key="sal4")

if st.button("Predict All"):
    # Attrition Prediction
    input1 = np.array([[age1, 1, 10, 3, 3, 1, 3, job_level1, 4, 1,
                        monthly_income1, 2, overtime1, 15, 3, 3, 1,
                        total_working_years1, 2, 3, 5, 3, 1, 4]])
    input1_scaled = model1.named_steps['scaler'].transform(input1) if hasattr(model1, 'named_steps') else input1
    pred1 = model1.predict(input1_scaled)[0]
    st.success(f"DF1 - Attrition Prediction: {'Yes' if pred1 else 'No'}")

    # Monthly Income Prediction
    input2 = np.array([[age2, distance2, education2, 3, job_level2, 2,
                        10, years_at_company2, 5, 1, 2, 1, 1, 3]])
    input2_scaled = model2.named_steps['scaler'].transform(input2) if hasattr(model2, 'named_steps') else input2
    pred2 = model2.predict(input2_scaled)[0]
    st.success(f"DF1 - Predicted Monthly Income: ${round(pred2, 2)}")

    # Leaving Prediction
    input3 = np.array([[satisfaction3, evaluation3, 3, hours3, time_spend3, 0,
                        satisfaction3 * evaluation3,
                        [1.0, 1.5, 2.0][salary3],
                        time_spend3, 2, salary3]])
    input3_scaled = model3.named_steps['scaler'].transform(input3) if hasattr(model3, 'named_steps') else input3
    pred3 = model3.predict(input3_scaled)[0]
    st.success(f"DF2 - Leaving Prediction: {'Yes' if pred3 else 'No'}")

    # Avg Monthly Hours Prediction
    input4 = np.array([[satisfaction4, evaluation4, 4, 0, time_spend4, 0,
                        satisfaction4 * evaluation4,
                        [1.0, 1.5, 2.0][salary4],
                        2, salary4]])
    input4_scaled = model4.named_steps['scaler'].transform(input4) if hasattr(model4, 'named_steps') else input4
    pred4 = model4.predict(input4_scaled)[0]
    st.success(f"DF2 - Predicted Avg Monthly Hours: {round(pred4, 2)}")
