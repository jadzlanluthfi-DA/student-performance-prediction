


import streamlit as st

import pandas as pd

import joblib



# =========================

# CONFIG

# =========================



st.set_page_config(

    page_title="Student Performance Prediction",

    page_icon="🎓",

    layout="wide"

)



# =========================

# LOAD MODEL

# =========================



model = joblib.load("model.pkl")

scaler = joblib.load("scaler.pkl")



# =========================

# HEADER

# =========================



st.title("🎓 Student Performance Prediction")



st.markdown("""

### Academic Performance Dashboard



Aplikasi ini memprediksi kemungkinan seorang siswa **PASS** atau **FAIL**

berdasarkan data akademik dan aktivitas belajar menggunakan

**Naive Bayes Classifier**.

""")



st.divider()



# =========================

# INPUT

# =========================



col1, col2 = st.columns(2)



with col1:



    st.subheader("👤 Informasi Siswa")



    age = st.selectbox(

        "Age",

        range(14, 20)

    )



    gender_text = st.selectbox(

        "Gender",

        ["Female", "Male"]

    )



    gender = 0 if gender_text == "Female" else 1



    student_class = st.selectbox(

        "Class",

        [9, 10, 11, 12]

    )



    parent_text = st.selectbox(

        "Parental Education",

        ["High School", "Bachelor", "Master"]

    )



    mapping_parent = {

        "High School": 0,

        "Bachelor": 1,

        "Master": 2

    }



    parental_education = mapping_parent[parent_text]



    internet_text = st.selectbox(

        "Internet Access",

        ["No", "Yes"]

    )



    internet_access = 1 if internet_text == "Yes" else 0



    extra_text = st.selectbox(

        "Extracurricular Activities",

        ["No", "Yes"]

    )



    extracurricular = 1 if extra_text == "Yes" else 0



with col2:



    st.subheader("📚 Data Akademik")



    study_hours = st.slider(

        "Study Hours Per Day",

        min_value=0.0,

        max_value=12.0,

        value=3.0,

        step=0.5

    )



    attendance = st.slider(

        "Attendance Percentage",

        min_value=0,

        max_value=100,

        value=75

    )



    math_score = st.slider(

        "Math Score",

        0,

        100,

        70

    )



    science_score = st.slider(

        "Science Score",

        0,

        100,

        70

    )



    english_score = st.slider(

        "English Score",

        0,

        100,

        70

    )



    previous_year_score = st.slider(

        "Previous Year Score",

        0,

        100,

        70

    )



    final_percentage = st.slider(

        "Final Percentage",

        0.0,

        100.0,

        70.0

    )



st.divider()



# =========================

# PREDICT

# =========================



if st.button("🔍 Predict Performance", use_container_width=True):



    data = pd.DataFrame({

        'Age':[age],

        'Gender':[gender],

        'Class':[student_class],

        'Study_Hours_Per_Day':[study_hours],

        'Attendance_Percentage':[attendance],

        'Parental_Education':[parental_education],

        'Internet_Access':[internet_access],

        'Extracurricular_Activities':[extracurricular],

        'Math_Score':[math_score],

        'Science_Score':[science_score],

        'English_Score':[english_score],

        'Previous_Year_Score':[previous_year_score],

        'Final_Percentage':[final_percentage]

    })



    data_scaled = scaler.transform(data)



    prediction = model.predict(data_scaled)[0]



    probabilities = model.predict_proba(data_scaled)



    fail_prob = probabilities[0][0] * 100

    pass_prob = probabilities[0][1] * 100



    st.divider()



    st.subheader("📊 Prediction Result")



    if prediction == 1:



        st.success("🎓 Student is Predicted to PASS")



        st.metric(

            "PASS Probability",

            f"{pass_prob:.2f}%"

        )



        st.progress(int(pass_prob))



        st.balloons()



    else:



        st.error("❌ Student is Predicted to FAIL")



        st.metric(

            "FAIL Probability",

            f"{fail_prob:.2f}%"

        )



        st.progress(int(fail_prob))



        st.snow()



    # =========================

    # SUMMARY

    # =========================



    st.divider()



    st.subheader("📋 Student Academic Summary")



    average_score = (

        math_score +

        science_score +

        english_score

    ) / 3



    m1, m2, m3 = st.columns(3)



    m1.metric(

        "Average Score",

        f"{average_score:.1f}"

    )



    m2.metric(

        "Attendance",

        f"{attendance}%"

    )



    m3.metric(

        "Study Hours",

        f"{study_hours} hrs/day"

    )



    st.write("### 🔎 Insights")



    # Study Hours



    if study_hours < 2:

        st.warning("📖 Study duration is relatively low.")

    elif study_hours < 5:

        st.info("📖 Study duration is adequate.")

    else:

        st.success("📖 Study duration is high.")



    # Attendance



    if attendance < 60:

        st.warning("📅 Attendance level is low.")

    elif attendance < 80:

        st.info("📅 Attendance level is moderate.")

    else:

        st.success("📅 Attendance level is high.")



    # Average Score



    if average_score < 60:

        st.warning("📝 Academic performance is below average.")

    elif average_score < 80:

        st.info("📝 Academic performance is good.")

    else:

        st.success("📝 Academic performance is excellent.")



    # Internet



    if internet_access == 1:

        st.success("🌐 Student has internet access.")

    else:

        st.warning("🌐 Student does not have internet access.")



    # Extracurricular



    if extracurricular == 1:

        st.success("🏆 Student participates in extracurricular activities.")

    else:

        st.info("🏆 Student does not participate in extracurricular activities.")



    st.divider()



    st.subheader("📄 Input Data")



    st.dataframe(

        data,

        use_container_width=True

    )



st.caption(

    "Built using Naive Bayes Classifier for Student Performance Prediction."

)
