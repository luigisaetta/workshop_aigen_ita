"""
Streamlit client for simple test on OCI Command R
"""

from time import time
import streamlit as st

from factory_for_citations_demo import do_query_and_answer
from utils import get_console_logger
from oci_citations_utils import extract_complete_citations, extract_document_list

# Constant
USER = "user"
ASSISTANT = "assistant"


def highlight_substrings(text, delimiters, doc_ids_list):
    """
    Evidenzia le sottostringhe in un testo e aggiunge le liste di doc_id tra parentesi quadre.

    Args:
    - text (str): il testo originale.
    - delimiters (list of tuples): ogni tupla contiene (start, end).
    - doc_ids_list (list of lists): lista delle liste di doc_id associati ai delimitatori.

    Returns:
    - str: il testo con le sottostringhe evidenziate e le liste di doc_id aggiunte.
    """
    combined = list(zip(delimiters, doc_ids_list))
    combined.sort(key=lambda x: x[0][0], reverse=True)

    for (start, end), doc_ids in combined:
        original_substring = text[start:end]
        highlighted_substring = (
            f'<span style="background-color: green;">{original_substring}</span>'
        )
        doc_id_str = f' [{", ".join(doc_ids)}]'
        text = text[:start] + highlighted_substring + doc_id_str + text[end:]

    return text


st.title("Oracle AI Assistant")
st.text_input("Ask a question:", key="question")

logger = get_console_logger()

# this is a list of tuples (start, end)
span_demarks = []
# this is a list of list [[1,2], [1,3]]
doc_ids = []

if st.button("Answer"):
    question = st.session_state.question

    with st.spinner("Invoking Command-R..."):
        time_start = time()

        # here we call the LLM
        response = do_query_and_answer(question)

        time_elapsed = time() - time_start
        logger.info("Total elapsed time: %s sec.", round(time_elapsed, 1))
        logger.info("")

    answer = response.data.chat_response.text

    # handle citations
    citations = extract_complete_citations(response)

    for citation in citations:
        span_demarks.append(citation["interval"])

        docs_for_this_citation = []
        for doc in citation["documents"]:
            docs_for_this_citation.append(doc["id"])
        doc_ids.append(docs_for_this_citation)

        assert len(doc_ids) == len(span_demarks)

    # insert citation
    highlighted_text = highlight_substrings(answer, span_demarks, doc_ids)

    st.markdown(highlighted_text, unsafe_allow_html=True)

    # show document for citations
    extracted_doc = extract_document_list(response)

    st.markdown("")
    st.markdown("Document list:")

    for doc in extracted_doc:
        st.markdown(f"[{doc['id']}]: {doc['source']}, pag: {doc['page']}")

    print(citations)
    print("")
