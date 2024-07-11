"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-04-27
Python Version: 3.11
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#
# The prompt for the condensed query on the Vector Store
#
CONTEXT_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

CONTEXT_Q_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXT_Q_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

#
# The prompt for the answer from the LLM
#
QA_SYSTEM_PROMPT = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Don't add sentences like: According to the provided context.

{context}"""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

#
# prompt for italian language
#

QA_SYSTEM_PROMPT_IT = """Sei un assistente per task di domanda-risposta. \
Utilizza i frammenti seguenti di testo per rispondere alla domanda. \
Se non conosci la risposta, dici semplicemente che non conosci la risposta \

{context}"""

QA_PROMPT_IT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT_IT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
