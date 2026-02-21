import streamlit as st
import pandas as pd
import pickle

# =========================
# LOAD MODEL
# =========================
with open("final_model2.sav", "rb") as f3:   # change name if needed
    pipeline = pickle.load(f3)

st.title("🚀 Startup Success Prediction")

st.markdown("Enter startup details below and predict success.")

# =========================
# USER INPUTS
# =========================
def user_input():
    col1, col2 = st.columns(2)

    with col1:
        age_first_funding_year = st.number_input("Age First Funding (years)", 0.0, 10.0, 1.0)
        age_last_funding_year = st.number_input("Age Last Funding (years)", 0.0, 15.0, 2.0)
        age_first_milestone_year = st.number_input("Age First Milestone (years)", 0.0, 10.0, 1.0)
        age_last_milestone_year = st.number_input("Age Last Milestone (years)", 0.0, 15.0, 3.0)
        age_startup_year = st.number_input("Startup Age (years)", 0.0, 20.0, 5.0)

        funding_rounds = st.number_input("Funding Rounds", 0, 10, 2)
        funding_total_usd = st.number_input("Total Funding (USD, millions)", 0.0, 100.0, 10.0)
        milestones = st.number_input("Milestones", 0, 20, 5)
        avg_participants = st.number_input("Avg Participants", 0.0, 10.0, 3.0)

    with col2:
        is_CA = st.checkbox("CA")
        is_NY = st.checkbox("NY")
        is_MA = st.checkbox("MA")
        is_TX = st.checkbox("TX")
        is_otherstate = st.checkbox("Other State")

        is_software = st.checkbox("Software")
        is_web = st.checkbox("Web")
        is_mobile = st.checkbox("Mobile")
        is_enterprise = st.checkbox("Enterprise")
        is_advertising = st.checkbox("Advertising")
        is_gamesvideo = st.checkbox("Games/Video")
        is_ecommerce = st.checkbox("E-commerce")
        is_biotech = st.checkbox("Biotech")
        is_consulting = st.checkbox("Consulting")
        is_othercategory = st.checkbox("Other Category")

        has_VC = st.checkbox("Has VC")
        has_angel = st.checkbox("Has Angel")

        has_roundA = st.checkbox("Round A")
        has_roundB = st.checkbox("Round B")
        has_roundC = st.checkbox("Round C")
        has_roundD = st.checkbox("Round D")

        is_top500 = st.checkbox("Top 500 Startup")

        tier_relationships = st.selectbox("Tier Relationships", [1, 2, 3, 4])

    # Derived features (same as training)
    has_RoundABCD = int(has_roundA or has_roundB or has_roundC or has_roundD)
    has_Investor = int(has_VC or has_angel)
    has_Seed = int(has_RoundABCD == 0 and has_Investor == 1)
    invalid_startup = int(has_RoundABCD == 0 and has_VC == 0 and has_angel == 0)

    data = {
        'age_first_funding_year': age_first_funding_year,
        'age_last_funding_year': age_last_funding_year,
        'age_first_milestone_year': age_first_milestone_year,
        'age_last_milestone_year': age_last_milestone_year,
        'funding_rounds': funding_rounds,
        'funding_total_usd': funding_total_usd,
        'milestones': milestones,

        'is_CA': int(is_CA),
        'is_NY': int(is_NY),
        'is_MA': int(is_MA),
        'is_TX': int(is_TX),
        'is_otherstate': int(is_otherstate),

        'is_software': int(is_software),
        'is_web': int(is_web),
        'is_mobile': int(is_mobile),
        'is_enterprise': int(is_enterprise),
        'is_advertising': int(is_advertising),
        'is_gamesvideo': int(is_gamesvideo),
        'is_ecommerce': int(is_ecommerce),
        'is_biotech': int(is_biotech),
        'is_consulting': int(is_consulting),
        'is_othercategory': int(is_othercategory),

        'has_VC': int(has_VC),
        'has_angel': int(has_angel),
        'has_roundA': int(has_roundA),
        'has_roundB': int(has_roundB),
        'has_roundC': int(has_roundC),
        'has_roundD': int(has_roundD),

        'avg_participants': avg_participants,
        'is_top500': int(is_top500),

        'has_RoundABCD': has_RoundABCD,
        'has_Investor': has_Investor,
        'has_Seed': has_Seed,
        'invalid_startup': invalid_startup,

        'age_startup_year': age_startup_year,
        'tier_relationships': tier_relationships
    }

    return pd.DataFrame([data])


# =========================
# PREDICTION
# =========================
input_df = user_input()

if st.button("Predict"):
    pred = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][1]

    if pred == 1:
        st.success(f"✅ Startup is likely SUCCESSFUL (Confidence: {prob:.2f})")
    else:
        st.error(f"❌ Startup is likely FAILED (Confidence: {1 - prob:.2f})")