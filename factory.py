"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-05-23
Python Version: 3.11
"""

import logging

# Cohere
from langchain_cohere import ChatCohere, CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

# to handle conversational memory
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# LLM
from langchain_community.llms import OCIGenAI

from factory_vector_store import get_vector_store
from oci_cohere_embeddings_utils import OCIGenAIEmbeddingsWithBatch

# prompts
from oracle_chat_prompts import CONTEXT_Q_PROMPT, QA_PROMPT

from utils import print_configuration, check_value_in_list, load_configuration

from config_private import (
    COMPARTMENT_ID,
    COHERE_API_KEY,
)

# configuratin is global
config = load_configuration()

#
# functions
#


def get_embed_model(model_type="OCI"):
    """
    get the Embeddings Model
    """
    check_value_in_list(model_type, ["OCI", "COHERE"])

    if model_type == "OCI":
        embed_model = OCIGenAIEmbeddingsWithBatch(
            auth_type="API_KEY",
            model_id=config["embeddings"]["oci"]["embed_model"],
            service_endpoint=config["embeddings"]["oci"]["embed_endpoint"],
            compartment_id=COMPARTMENT_ID,
        )
    return embed_model


def get_llm(model_type="OCI"):
    """
    Build and return the LLM client
    """
    check_value_in_list(model_type, ["OCI", "COHERE"])

    if model_type == "OCI":
        llm = OCIGenAI(
            auth_type="API_KEY",
            model_id=config["llm"]["oci"]["llm_model"],
            service_endpoint=config["llm"]["oci"]["endpoint"],
            compartment_id=COMPARTMENT_ID,
            model_kwargs={
                "max_tokens": config["llm"]["max_tokens"],
                "temperature": config["llm"]["temperature"],
            },
        )
    if model_type == "COHERE":
        llm = ChatCohere(
            cohere_api_key=COHERE_API_KEY,
            model=config["llm"]["cohere"]["llm_model"],
            max_tokens=config["llm"]["max_tokens"],
            temperature=config["llm"]["temperature"],
        )
    return llm


#
# create the entire RAG chain
#
def build_rag_chain(verbose):
    """
    Build the entire RAG chain
    """
    logger = logging.getLogger("ConsoleLogger")

    # print all the used configuration to the console
    print_configuration(config)

    embed_model = get_embed_model(config["embeddings"]["embed_model_type"])

    v_store = get_vector_store(
        vector_store_type=config["vector_store"]["store_type"], embed_model=embed_model
    )

    # 10/05: I can add a filter here (for ex: to filter by profile_id)
    base_retriever = v_store.as_retriever(k=config["retriever"]["top_k"])

    # add the reranker
    if config["reranker"]["add_reranker"]:
        if verbose:
            logger.info("Adding a reranker...")

        cohere_rerank = CohereRerank(
            cohere_api_key=COHERE_API_KEY,
            top_n=config["retriever"]["top_n"],
            model=config["reranker"]["cohere_reranker_model"],
        )

        retriever = ContextualCompressionRetriever(
            base_compressor=cohere_rerank, base_retriever=base_retriever
        )
    else:
        # no reranker
        retriever = base_retriever

    llm = get_llm(model_type=config["llm"]["model_type"])

    # steps to add chat_history
    # 1. create a retriever using chat history
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONTEXT_Q_PROMPT
    )

    # 2. create the chain for answering
    # we need to use a different prompt from the one used to
    # condense the standalone question
    question_answer_chain = create_stuff_documents_chain(llm, QA_PROMPT)

    # 3, the entire chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # this returns sources and can be streamed
    return rag_chain
