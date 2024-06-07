"""
Factory methods for demo citations
"""

import logging

import oracledb

from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy

# to compute embeddings vectors
from langchain_community.embeddings import OCIGenAIEmbeddings

from oci_command_r_oo import OCICommandR

from utils import load_configuration

# private information
from config_private import COMPARTMENT_ID, DB_USER, DB_PWD, DB_HOST_IP, DB_SERVICE

# some configs
logger = logging.getLogger("ConsoleLogger")

logger.setLevel(logging.INFO)

# load the config in the config.toml file
config = load_configuration()

# embeddings model: we're using OCI GenAI multilingual Cohere
OCI_EMBED_MODEL = config["embeddings"]["oci"]["embed_model"]
EMBED_ENDPOINT = config["embeddings"]["oci"]["embed_endpoint"]

LLM_ENDPOINT = config["llm"]["oci"]["endpoint"]

DSN = f"{DB_HOST_IP}:1521/{DB_SERVICE}"

# number of docs retrieved for each query
# reduced from config to simplify output here
TOP_K = 4


def get_embed_model():
    """ """
    embed_model = OCIGenAIEmbeddings(
        auth_type="API_KEY",
        model_id=OCI_EMBED_MODEL,
        service_endpoint=EMBED_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )
    return embed_model


def get_oracle_vs(embed_model):
    """ 
    create a connection and return oraclevs
    """
    try:
        # we need to provide a connection as input to OracleVS
        connection = oracledb.connect(user=DB_USER, password=DB_PWD, dsn=DSN)
        logger.info("Connection successful!")

        # get an instance of OracleVS
        v_store = OracleVS(
            client=connection,
            table_name="ORACLE_KNOWLEDGE",
            distance_strategy=DistanceStrategy.COSINE,
            embedding_function=embed_model,
        )

    except Exception as e:
        logger.error("Connection failed!")
        logger.error(e)
        v_store = None

    return v_store


def get_retriever(v_store):
    """ 
    get the LangChain retriever
    """
    retriever = v_store.as_retriever(search_kwargs={"k": TOP_K})

    return retriever


def get_chat_model():
    """ 
    build the chat Cohere client
    """
    command_r_params = {
        "model": "cohere.command-r-16k",
        "service_endpoint": LLM_ENDPOINT,
        "compartment_id": COMPARTMENT_ID,
        "max_tokens": 1024,
    }
    # this is a custom class that wraps OCI Python SDK
    chat = OCICommandR(**command_r_params)

    return chat


def do_query_and_answer(query):
    """ 
    build the chain, process the query and return chat answer
    """
    embed_model = get_embed_model()

    v_store = get_oracle_vs(embed_model)

    retriever = get_retriever(v_store)

    logger.info("Doing semantich search...")
    result_docs = retriever.invoke(query)

    # Cohere wants a map
    # take the output from the AI Vector Search
    # and transform in a format suitable for Cohere command-r
    documents_txt = [
        {
            "id": str(i + 1),
            "snippet": doc.page_content,
            "source": doc.metadata["source"],
            "page": str(doc.metadata["page"]),
        }
        for i, doc in enumerate(result_docs)
    ]

    chat_history = []

    chat = get_chat_model()

    logger.info("Invoking chat model...")

    response = chat.invoke(query=query, chat_history=[], documents=documents_txt)

    return response
