"""
Test OCiCommandR
"""

from oci_command_r_oo import OCICommandR

chat = OCICommandR(
    model="cohere.command-r-16k",
    service_endpoint="https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..aaaaaaaaushuwb2evpuf7rcpl4r7ugmqoe7ekmaiik3ra3m7gec3d234eknq",
    max_tokens=1024,
    is_streaming=True,
)

QUERY = "What is Retrieval Augmented Generation?"
chat_history = []
# no grounded
documents = []

response = chat.invoke(QUERY, chat_history, documents)

chat.print_response(response)
