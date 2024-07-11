"""
Streamlit client for simple test on OCI Command R
"""

import streamlit as st
from oci_command_r_oo import OCICommandR


def inserisci_link_multipli_con_tooltip(stringa, intervalli, urls, tooltips):
    """
    to add
    """
    # Offset totale dovuto agli inserimenti
    offset = 0

    for (start, stop), url, tooltip in zip(intervalli, urls, tooltips):
        # Calcola le nuove posizioni tenendo conto dell'offset
        start += offset
        stop += offset

        # Creare il link HTML con il tooltip
        link_start = f'<a href="{url}" target="_blank" title="{tooltip}">'
        link_end = "</a>"

        # Inserisci il tag di apertura del link subito dopo la posizione di start
        stringa = stringa[:start] + link_start + stringa[start:]
        # Aggiorna l'offset dopo l'inserimento del tag di apertura
        offset += len(link_start)

        # Calcola la nuova posizione di stop considerando l'inserimento precedente
        stop += len(link_start)
        # Inserisci il tag di chiusura del link subito dopo la posizione di stop
        stringa = stringa[: stop + 1] + link_end + stringa[stop + 1 :]
        # Aggiorna l'offset dopo l'inserimento del tag di chiusura
        offset += len(link_end)

    return stringa


st.title("Oracle AI Assistant")
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

span_demarks = [(131, 150), (160, 180)]
URL = ["http://www.oracle.com", "http://www.oracle.com"]
TOOLTIP = ["Documento 1, pag. 10", "Documento 2, pag. 11"]

if st.button("Answer"):
    question = st.session_state.question

    with st.spinner("Invoking Command-R..."):
        response = chat.invoke(question, chat_history, documents)

    answer = response.data.chat_response.text

    # insert marker
    new_answer = inserisci_link_multipli_con_tooltip(answer, span_demarks, URL, TOOLTIP)

    st.markdown(new_answer, unsafe_allow_html=True)
