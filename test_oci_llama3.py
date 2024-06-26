"""
Test OCILlama3
"""

import sys

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from oci_llama3_oo import OCILlama3

#
# get query from command line
#
if len(sys.argv) > 1:
    query = sys.argv[1]
    print("")
    print(f"Query is: {query}")
else:
    print("")
    print("No query specified !!!")
    print("")
    sys.exit(-1)

IS_STREAMING = True

chat = OCILlama3(
    model="meta.llama-3-70b-instruct",
    service_endpoint="https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..aaaaaaaaushuwb2evpuf7rcpl4r7ugmqoe7ekmaiik3ra3m7gec3d234eknq",
    max_tokens=1024,
    is_streaming=IS_STREAMING,
)

chat_history = [
    HumanMessage(content="Ciao, cosa è l'aspirina?"),
    AIMessage(
        content="E' il farmaco più comunemente utilizzato per trattare la febbre"
    ),
    HumanMessage(content="Può essere utilizzato nei bambini?"),
]

if IS_STREAMING:
    response = chat.stream(chat_history)
else:
    response = chat.invoke(chat_history)

chat.print_response(response)
