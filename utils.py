"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-04-27
Python Version: 3.11
"""

import logging
import os
import toml

from config_private import LANGSMITH_API_KEY


def remove_path_from_ref(ref_pathname):
    """
    remove the path from source (ref)
    """
    ref = ref_pathname
    # check if / or \ is contained
    if len(ref_pathname.split(os.sep)) > 0:
        ref = ref_pathname.split(os.sep)[-1]

    return ref


def load_configuration():
    """
    read the configuration from config.toml
    """
    config = toml.load("config.toml")

    return config


def enable_tracing(config):
    """
    To enable tracing with LangSmith
    """
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = config["tracing"]["langchain_project"]
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY


def get_console_logger():
    """
    To get a logger to print on console
    """
    logger = logging.getLogger("ConsoleLogger")

    # to avoid duplication of logging
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False

    return logger


def format_docs(docs):
    """
    format docs for LCEL
    """
    return "\n\n".join(doc.page_content for doc in docs)


def print_configuration(config):
    """
    print the current config
    """
    logger = logging.getLogger("ConsoleLogger")

    logger.info("--------------------------------------------------")
    logger.info("Configuration used:")
    logger.info("")

    EMBED_MODEL_TYPE = config["embeddings"]["embed_model_type"]
    logger.info(" Embedding model type: %s", EMBED_MODEL_TYPE)

    if EMBED_MODEL_TYPE == "OCI":
        logger.info(
            " Using %s for Embeddings...", config["embeddings"]["oci"]["embed_model"]
        )

    if config["reranker"]["add_reranker"]:
        logger.info(" Added Cohere Reranker...")
        logger.info(
            " Using %s as reranker...", config["reranker"]["cohere_reranker_model"]
        )

    logger.info(" Using %s as Vector Store...", config["vector_store"]["store_type"])
    logger.info(" Retrieval parameters:")
    logger.info("    TOP_K: %s", config["retriever"]["top_k"])

    if config["reranker"]["add_reranker"]:
        logger.info("    TOP_N: %s", config["retriever"]["top_n"])

    LLM_MODEL_TYPE = config["llm"]["model_type"]
    logger.info(" Using %s as Generative Model type...", LLM_MODEL_TYPE)
    if LLM_MODEL_TYPE == "COHERE":
        logger.info(" Using %s for LLM...", config["llm"]["cohere"]["llm_model"])
    if LLM_MODEL_TYPE == "OCI":
        logger.info(" Using %s for LLM...", config["llm"]["oci"]["llm_model"])

    if config["tracing"]["enable"] == "true":
        logger.info("")
        logger.info(" Enabled Observability with LangSmith...")

    logger.info("--------------------------------------------------")
    logger.info("")


def check_value_in_list(value, values_list):
    """
    to check that we don't enter a not supported value
    """
    if value not in values_list:
        raise ValueError(
            f"Value {value} is not valid: value must be in list {values_list}"
        )


def answer(chain, question):
    """
    method to test answer
    """
    response = chain.invoke({"question": question})

    print(question)
    print("")
    print(response["answer"])
    print("")
