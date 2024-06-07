"""
Test OCiCommandR
"""

from oci_llama3_oo import OCILlama3

chat = OCILlama3(
    model="meta.llama-3-70b-instruct",
    service_endpoint="https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..aaaaaaaaushuwb2evpuf7rcpl4r7ugmqoe7ekmaiik3ra3m7gec3d234eknq",
    max_tokens=1024,
    is_streaming=False,
)

QUERY = "Cosa è la Retrieval Augmented Generation?"
chat_history = []
# no grounded
documents = []

response = chat.invoke(QUERY, chat_history, documents)

chat.print_response(response)
