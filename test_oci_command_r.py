"""
Test OCiCommandR
"""

import sys
from oci_command_r_oo import OCICommandR

if len(sys.argv) > 1:
    query = sys.argv[1]
    print("")
    print(f"Query is: {query}")
else:
    print("")
    print("No query specified !!!")
    print("")
    sys.exit(-1)

#
# This is the format for preamble recommended by
# Cohere (https://docs.cohere.com/docs/preambles#)
#
preamble_it = """

## Task & Context
Tu sei un Assistente AI che fornisce risposte a domande in lingua italiana.
Se non conosci la risposta, rispondi semplicemente: Mi dispiace non lo so.

## Style Guide
Usa sempre la lingua italiana. Inizia ogni risposta con Ciao, ma si cortese ed educato nelle risposte.

"""

chat = OCICommandR(
    model="cohere.command-r-16k",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..aaaaaaaaushuwb2evpuf7rcpl4r7ugmqoe7ekmaiik3ra3m7gec3d234eknq",
    max_tokens=1024,
    is_streaming=True,
    # mostra come possiamo cambiare il prompt
    preamble_override=preamble_it,
)

chat_history = []
# no grounded
documents = []

response = chat.invoke(query, chat_history, documents)

chat.print_response(response)
