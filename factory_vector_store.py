"""
Author: Luigi Saetta
Date created: 2024-05-20
Date last modified: 2024-05-23

Usage:
    This module handles the creation of the Vector Store 
    used in the RAG chain, based on config

Python Version: 3.11
"""

import logging
import oracledb

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy

from utils import check_value_in_list

from config import (
    COLLECTION_NAME,
    # shared params for opensearch
    OPENSEARCH_SHARED_PARAMS,
)
from config_private import (
    OPENSEARCH_USER,
    OPENSEARCH_PWD,
    DB_USER,
    DB_PWD,
    DB_HOST_IP,
    DB_SERVICE,
)


def get_vector_store(vector_store_type, embed_model):
    """
    local_index_dir, books_dir only needed for FAISS
    Faiss: Read or rebuild the index and retur a Vector Store
    """

    check_value_in_list(vector_store_type, ["OPENSEARCH", "23AI"])

    logger = logging.getLogger("ConsoleLogger")

    v_store = None

    if vector_store_type == "OPENSEARCH":
        # this assumes that there is an OpenSearch cluster available
        # or docker, at the specified URL
        v_store = OpenSearchVectorSearch(
            embedding_function=embed_model,
            http_auth=(OPENSEARCH_USER, OPENSEARCH_PWD),
            **OPENSEARCH_SHARED_PARAMS,
        )

    elif vector_store_type == "23AI":
        dsn = f"{DB_HOST_IP}:1521/{DB_SERVICE}"

        try:
            connection = oracledb.connect(user=DB_USER, password=DB_PWD, dsn=dsn)

            v_store = OracleVS(
                client=connection,
                table_name=COLLECTION_NAME,
                distance_strategy=DistanceStrategy.COSINE,
                embedding_function=embed_model,
            )
        except oracledb.Error as e:
            err_msg = "An error occurred in get_vector_store: " + str(e)
            logger.error(err_msg)

    return v_store
