import streamlit as st

st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Select an option", ("Train Models", "Make Predictions"))

if option == "Train Models":
    import training
    training.train_models()
else:
    import inference
    inference.run_inference()
