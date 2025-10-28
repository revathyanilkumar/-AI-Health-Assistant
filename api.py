import streamlit as st
from model import load_model, generate_response

st.set_page_config(page_title="AI Health Assistant", layout="centered")

st.title("🩺 AI Health Assistant")
st.markdown("Ask me any health-related question! (For educational purposes only.)")

# Load model
@st.cache_resource
def get_model():
    return load_model()

model, tokenizer = get_model()

# Chat UI
user_input = st.text_area("Enter your health question here:", height=100)

if st.button("Get Answer"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            response = generate_response(model, tokenizer, user_input)
        st.markdown(f"**Answer:** {response}")
    else:
        st.warning("Please enter a question.")
