import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Happiness Dashboard", layout="wide")

# --- DATA PREPARATION (Using Global Happiness Trends) ---
@st.cache_data
def load_happiness_data():
    # Hum Gapminder data ko Happiness Proxy ke taur par use kar rahe hain
    # Aur us mein Social factors simulate kar rahe hain
    df = px.data.gapminder().query("year == 2007")
    
    # Happiness Score Calculation (Hypothetical AI Model based on GDP & Life Expectancy)
    # Happiness = (Log of GDP * 0.4) + (Life Expectancy * 0.3) + (Random Social Factor)
    df['Happiness_Score'] = (np.log10(df['gdpPercap']) * 1.5) + (df['lifeExp'] * 0.05)
    
    # Adding specific focus for Pakistan
    countries = ['Pakistan', 'India', 'China', 'United States', 'Norway', 'Finland']
    return df[df['country'].isin(countries)]

import numpy as np
df_happy = load_happiness_data()

# --- SIDEBAR ---
st.sidebar.title("😊 Happiness Analysis")
menu = st.sidebar.radio("Menu:", ["Project Overview", "Happiness vs Wealth", "AI Happiness Predictor"])

# --- 1. OVERVIEW ---
if menu == "Project Overview":
    st.title("🌈 World Happiness Report: AI Analysis")
    st.markdown("""
    ### Why Happiness?
    Sirf paisa (GDP) kisi mulk ki kamyabi ka miyar nahi hota. Yeh project analyze karta hai ke:
    *   **Social Support** aur **Life Expectancy** ka Happiness par kya asar hai?
    *   AI kaise predict kar sakta hai ke log kitne khush hain?
    """)
    st.image("https://images.unsplash.com/photo-1516211697506-8360bd773497?auto=format&fit=crop&w=1000&q=80", caption="Happiness is a Global Goal")

# --- 2. VISUALIZATION ---
elif menu == "Happiness vs Wealth":
    st.header("📊 Happiness vs. Economic Strength")
    
    fig = px.scatter(df_happy, x="gdpPercap", y="Happiness_Score", 
                     size="pop", color="country", text="country",
                     log_x=True, size_max=60, template="plotly_dark",
                     title="Global Happiness Mapping")
    
    st.plotly_chart(fig, use_container_width=True)
    st.write("Yeh graph dikhata hai ke Wealth zaroori hai, lekin aik point ke baad Happiness doosre factors par depend karti hai.")

# --- 3. AI PREDICTOR ---
elif menu == "AI Happiness Predictor":
    st.header("🔮 AI Happiness Score Predictor")
    
    # Training a simple model
    X = df_happy[['gdpPercap', 'lifeExp']]
    y = df_happy['Happiness_Score']
    model = LinearRegression().fit(X, y)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Country Stats")
        gdp_input = st.number_input("GDP per Capita ($):", value=2000)
        life_input = st.slider("Average Life Expectancy:", 40, 90, 67)
        
    with col2:
        st.subheader("AI Prediction")
        pred = model.predict([[gdp_input, life_input]])[0]
        st.metric(label="Predicted Happiness Score (1-10)", value=f"{pred:.2f}")
        
        if pred > 7:
            st.balloons()
            st.success("This indicates a very high quality of life!")
        elif pred > 5:
            st.info("Moderate happiness level. Growing potential!")