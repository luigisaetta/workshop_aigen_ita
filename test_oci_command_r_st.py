"""
Streamlit client for simple test on OCI Command R
"""

import streamlit as st
from oci_command_r_oo import OCICommandR

st.title("Oracle QA Chatbot")
st.text_input("Ask a question:", key="question")

# LLM
chat = OCICommandR(
    model="cohere.command-r-16k",
    service_endpoint="https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..aaaaaaaaushuwb2evpuf7rcpl4r7ugmqoe7ekmaiik3ra3m7gec3d234eknq",
    max_tokens=1024,
    is_streaming=False,
)

# standalone question
chat_history = []
# here you should put docs returned by vector search
documents = []

if st.button("Answer"):
    question = st.session_state.question

    with st.spinner("Invoking Command-R..."):
        response = chat.invoke(question, chat_history, documents)

    answer = response.data.chat_response.text

    st.write(answer)
