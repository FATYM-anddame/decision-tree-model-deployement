import streamlit as st
from adult_income import show as adult_income
from heart import show as heart_disease
# Create a dictionary to map page names to their respective show functions
pages = {
    "adult income": adult_income,
    "heart disease": heart_disease,
}

# Create a sidebar with page selection
selected_page = st.sidebar.selectbox("selectionner le jeu de données", list(pages.keys()))

# Render the selected page
page = pages[selected_page]
page()
