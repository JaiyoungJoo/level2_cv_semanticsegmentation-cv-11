import streamlit as st
from multiapp import MultiApp
from apps import home, data_stats,data_eda # import your app modules here

app = MultiApp()

# Add all your application here
st.set_page_config(layout="wide")
app.add_app("Hand Bone Segmentation", home.app)
app.add_app("Error Analysis", data_stats.app)
app.add_app("Data EDA", data_eda.app)

# The main app
app.run()