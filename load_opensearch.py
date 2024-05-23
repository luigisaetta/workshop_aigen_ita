"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-04-30
Python Version: 3.11

Usage:
    With this code you can load the initial content inside 
    the OpenSearch based Vector Store
"""

from langchain_community.vectorstores import OpenSearchVectorSearch

from factory import get_embed_model
from chunk_index_utils import load_books_and_split
from utils import get_console_logger, load_configuration

from config_private import OPENSEARCH_USER, OPENSEARCH_PWD

logger = get_console_logger()

config = load_configuration()

# load all the books in BOOKS_DIR
books_dir = config["text_splitting"]["books_dir"]

docs = load_books_and_split(books_dir)

embed_model = get_embed_model(model_type="OCI")

OPENSEARCH_PARAMS = {
    "opensearch_url": config["vector_store"]["opensearch"]["opensearch_url"],
    "use_ssl": config["vector_store"]["opensearch"]["use_ssl"],
    "verify_certs": config["vector_store"]["opensearch"]["verify_certs"],
    "ssl_assert_hostname": config["vector_store"]["opensearch"]["ssl_assert_hostname"],
    "ssl_show_warn": config["vector_store"]["opensearch"]["ssl_show_warn"],
    "bulk_size": int(config["vector_store"]["opensearch"]["bulk_size"]),
    "index_name": config["vector_store"]["opensearch"]["index_name"],
    "engine": config["vector_store"]["opensearch"]["engine"],
}

# load text and embeddings in OpenSearch
docsearch = OpenSearchVectorSearch.from_documents(
    docs,
    embedding=embed_model,
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PWD),
    **OPENSEARCH_PARAMS
)

# Do a test
QUERY = "La metformina può essere usata per curare il diabete di tipo 2 nei pazienti anziani?"
results = docsearch.similarity_search(QUERY, k=4)

print("Test:")
print(QUERY)
print(len(results))
print(results)
