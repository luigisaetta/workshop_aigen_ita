"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-04-30

Usage:
    This program can be used to test OpenSearch
    Vector Store

Python Version: 3.11
"""

from glob import glob
from tqdm.auto import tqdm

from langchain_community.document_loaders import PyPDFLoader

from factory import get_embed_model, get_vector_store
from chunk_index_utils import get_recursive_text_splitter
from utils import get_console_logger, load_configuration


#
# Main
#
config = load_configuration()

logger = get_console_logger()

books_dir = config["text_splitting"]["books_dir"]

logger.info(
    "Loading documents from %s...",
)

text_splitter = get_recursive_text_splitter()

books_list = glob(books_dir + "/*.pdf")

docs = []

for book in tqdm(books_list):
    loader = PyPDFLoader(file_path=book)

    docs += loader.load_and_split(text_splitter=text_splitter)

logger.info("Loaded %s chunks...", len(docs))

embed_model = get_embed_model(model_type="OCI")

# need to replace with from_documents
docsearch = get_vector_store("OPENSEARCH", embed_model)

# test
QUERY = "La metformina può essere usata per curare il diabete di tipo 2 nei pazienti anziani?"
results = docsearch.similarity_search(QUERY, k=4)

print(len(results))

for doc in results:
    print(doc)
    print("")
