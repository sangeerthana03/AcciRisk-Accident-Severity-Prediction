


import streamlit as st
import pickle
import pandas as pd


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AcciRisk | Road Safety Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. LOAD TRAINED MODEL
# =====================================================

with open('models/rf_model.pkl', 'rb') as file:
    saved_data = pickle.load(file)

model = saved_data['model']
columns = saved_data['columns']


# =====================================================
# 3. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* Main application */

.stApp {
    background-color: #0B1120;
    color: #E2E8F0;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #111C30;
    border-right: 1px solid #26354D;
}

[data-testid="stSidebar"] * {
    color: #CBD5E1;
}

/* Blue sidebar headings */
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h3 span,
[data-testid="stSidebar"] h3 p {
    color: #3B82F6 !important;
}

/* Headings */

h1 {
    color: #F8FAFC !important;
    font-size: 32px !important;
    font-weight: 700 !important;
}

h2 {
    color: #F1F5F9 !important;
    font-size: 23px !important;
}

h3 {
    color: #E2E8F0 !important;
    font-size: 18px !important;
}

p, label {
    color: #CBD5E1 !important;
}

/* Header */

.main-subtitle {
    color: #94A3B8;
    font-size: 15px;
    margin-top: -12px;
    margin-bottom: 25px;
}

/* Section headings */

.section-heading {
    color: #60A5FA;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 16px;
}

/* Input fields */

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background-color: #17243A;
    border: 1px solid #33445F;
    border-radius: 7px;
}

div[data-baseweb="input"] input {
    color: #F8FAFC !important;
}

div[data-baseweb="select"] span {
    color: #F8FAFC !important;
}

div[data-baseweb="select"] svg {
    fill: #94A3B8;
}

/* Buttons */

div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background-color: #2563EB;
    color: white;
    border: none;
    border-radius: 8px;
    height: 48px;
    font-size: 15px;
    font-weight: 600;
    width: 100%;
    transition: 0.2s;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #1D4ED8;
    color: white;
    border: none;
}

/* Horizontal divider */

hr {
    border-color: #26354D !important;
}

/* Information cards */

.info-card {
    background-color: #142138;
    border: 1px solid #263B59;
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 12px;
}

.info-label {
    color: #94A3B8;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.info-value {
    color: #F8FAFC;
    font-size: 19px;
    font-weight: 600;
    margin-top: 7px;
}

/* Prediction panel */

.prediction-panel {
    background-color: #142138;
    border: 1px solid #33445F;
    border-radius: 12px;
    padding: 25px;
    text-align: center;
    margin-top: 12px;
}

.prediction-label {
    color: #94A3B8;
    font-size: 13px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.prediction-value {
    font-size: 32px;
    font-weight: 750;
    margin-top: 10px;
}

/* Probability */

.probability-label {
    color: #E2E8F0;
    font-size: 14px;
    font-weight: 600;
}

/* Footer */

.footer {
    text-align: center;
    color: #64748B;
    font-size: 12px;
    padding-top: 20px;
}

/* Hide default Streamlit menu */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("""
    <h3 style="color:#3B82F6;">Project Overview</h3>
    """, unsafe_allow_html=True)

    st.write(
        "A machine learning application that predicts "
        "road accident severity based on driver, road, "
        "vehicle and environmental conditions."
    )

    st.divider()

    st.markdown("""
    <h3 style="color:#3B82F6;">Model Information</h3>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <div class="info-label">Algorithm</div>
        <div class="info-value">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <div class="info-label">Task</div>
        <div class="info-value">Multi-class Classification</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <div class="info-label">Output Classes</div>
        <div class="info-value">Minor | Moderate | Severe</div>
    </div>
    """, unsafe_allow_html=True)




# =====================================================
# 5. MAIN HEADER
# =====================================================



st.markdown("""
<div style="margin-bottom: 25px;">
    <div style="font-size:46px; font-weight:800; letter-spacing:-1.5px;">
        <span style="color:#F8FAFC;">Acci</span><span style="color:#3B82F6;">Risk</span>
    </div>
    <div style="color:#94A3B8; font-size:16px; margin-top:8px;">
        Machine Learning Based Accident Severity Prediction
    </div>
</div>
""", unsafe_allow_html=True)



st.divider()


# =====================================================
# 6. INPUT FORM
# =====================================================

st.markdown("## Accident Details")

st.write(
    "Provide the following information to generate a prediction."
)

with st.form("accident_prediction_form"):

    # ROAD AND DRIVER DETAILS

    st.markdown(
        '<div class="section-heading">01 &nbsp; Road and Driver Information</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        speed_kmh = st.number_input(
            "Speed (km/h)",
            min_value=0.0,
            max_value=250.0,
            value=40.0
        )

        visibility_m = st.number_input(
            "Visibility (metres)",
            min_value=0.0,
            max_value=10000.0,
            value=1000.0
        )

        driver_age = st.number_input(
            "Driver Age",
            min_value=16,
            max_value=100,
            value=30
        )

        driver_experience_yrs = st.number_input(
            "Driving Experience (years)",
            min_value=0,
            max_value=82,
            value=5
        )

    with col2:

        alcohol_involved = st.selectbox(
            "Alcohol Involvement",
            ["No", "Yes"]
        )

        seatbelt_worn = st.selectbox(
            "Seatbelt Worn",
            ["No", "Yes"]
        )

        road_type = st.selectbox(
            "Road Type",
            [
                "Expressway",
                "Highway",
                "One-way",
                "Rural Road",
                "Urban Road"
            ]
        )

        vehicle_type = st.selectbox(
            "Vehicle Type",
            [
                "Bus",
                "Car",
                "Motorcycle",
                "SUV",
                "Truck",
                "Van"
            ]
        )

    st.divider()

    # ENVIRONMENTAL CONDITIONS

    st.markdown(
        '<div class="section-heading">02 &nbsp; Environmental Conditions</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        weather_condition = st.selectbox(
            "Weather Condition",
            [
                "Clear",
                "Fog",
                "Rain",
                "Snow",
                "Storm",
                "Wind"
            ]
        )

    with col2:

        road_condition = st.selectbox(
            "Road Condition",
            [
                "Dry",
                "Icy",
                "Potholed",
                "Under Construction",
                "Wet"
            ]
        )

    with col3:

        light_condition = st.selectbox(
            "Light Condition",
            [
                "Dawn",
                "Daylight",
                "Dusk",
                "Night - Lit",
                "Night - Unlit"
            ]
        )

    st.divider()

    predict_button = st.form_submit_button(
        "GENERATE PREDICTION"
    )


# =====================================================
# 7. PREDICTION LOGIC
# =====================================================

if predict_button:

    if driver_experience_yrs > driver_age - 16:

        st.error(
            "Invalid driver details. Driving experience "
            "cannot exceed the number of years since age 16."
        )

    else:

        # Convert categorical Yes/No values

        alcohol_involved = (
            1 if alcohol_involved == "Yes" else 0
        )

        seatbelt_worn = (
            1 if seatbelt_worn == "Yes" else 0
        )

        # Create input DataFrame

        input_data = pd.DataFrame([{
            'speed_kmh': speed_kmh,
            'visibility_m': visibility_m,
            'driver_age': driver_age,
            'driver_experience_yrs': driver_experience_yrs,
            'alcohol_involved': alcohol_involved,
            'seatbelt_worn': seatbelt_worn,
            'road_type': road_type,
            'weather_condition': weather_condition,
            'road_condition': road_condition,
            'light_condition': light_condition,
            'vehicle_type': vehicle_type
        }])

        # One-hot encoding

        input_data = pd.get_dummies(input_data)

        # Align columns with training data

        input_data = input_data.reindex(
            columns=columns,
            fill_value=0
        )

        # Model prediction

        prediction = model.predict(
            input_data.to_numpy()
        )

        probabilities = model.predict_proba(
            input_data.to_numpy()
        )[0]

        predicted_class = prediction[0]


        # =================================================
        # 8. DISPLAY PREDICTION RESULT
        # =================================================

        st.divider()

        # st.markdown("## Prediction Analysis")
        #
        # st.write(
        #     "The trained Random Forest model has generated "
        #     "the following prediction."
        # )

        if predicted_class == 'Severe':

            result_color = "#F87171"
            result_icon = "🔴"

        elif predicted_class == 'Minor':

            result_color = "#4ADE80"
            result_icon = "🟢"

        else:

            result_color = "#FBBF24"
            result_icon = "🟠"


        st.markdown(f"""
        <div class="prediction-panel">
            <div class="prediction-label">
                PREDICTED ACCIDENT SEVERITY
            </div>
            <div class="prediction-value" style="color:{result_color};">
                {result_icon} {predicted_class}
            </div>
        </div>
        """, unsafe_allow_html=True)


        # =================================================
        # 9. PROBABILITY DISTRIBUTION
        # =================================================

        st.markdown("### Class Probability Distribution")

        st.write(
            "Estimated probability assigned by the model "
            "to each severity category."
        )

        for severity, probability in zip(
            model.classes_,
            probabilities
        ):

            col1, col2 = st.columns([1, 5])

            with col1:
                st.markdown(
                    f"<div class='probability-label'>{severity}</div>",
                    unsafe_allow_html=True
                )

            with col2:
                st.progress(float(probability))

            st.caption(f"{probability:.2%}")


        st.divider()







