"""
Streamlit client for simple test on OCI Command R
"""

from time import time
import streamlit as st

from factory_for_citations_demo import do_query_and_answer
from utils import get_console_logger


def inserisci_link_multipli_con_tooltip(stringa, intervalli, urls, tooltips):
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

logger = get_console_logger()

span_demarks = [(131, 150), (160, 180)]
url = ["http://www.oracle.com", "http://www.oracle.com"]
tooltip = ["Documento 1, pag. 10", "Documento 2, pag. 11"]

if st.button("Answer"):
    question = st.session_state.question

    with st.spinner("Invoking Command-R..."):
        time_start = time()

        response = do_query_and_answer(question)

        time_elapsed = time() - time_start
        logger.info("Total elapsed time: %s sec.", round(time_elapsed, 1))
        logger.info("")

    answer = response.data.chat_response.text

    # insert marker
    new_answer = inserisci_link_multipli_con_tooltip(answer, span_demarks, url, tooltip)

    st.markdown(new_answer, unsafe_allow_html=True)
