"""
Cohere test is_query_only
"""

from oci.generative_ai_inference.models import CohereMessage

from oci_command_r_oo import OCICommandR

chat = OCICommandR(
    model="cohere.command-r-16k",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..aaaaaaaaushuwb2evpuf7rcpl4r7ugmqoe7ekmaiik3ra3m7gec3d234eknq",
    # mostra come possiamo cambiare il prompt
    is_search_queries_only=True,
)

msg1 = CohereMessage(
    role="USER", message="L'aspirina può essere usata per trattare la febbre?"
)
msg2 = CohereMessage(
    role="CHATBOT",
    message="Si, può essere usata, seguendo le indicazioni e sotto controllo medico",
)

chat_history = [msg1, msg2]

# no grounded
documents = []

query = "Quali possono essere gli effetti collaterali nei bambini e negli adolescenti?"

response = chat.invoke(query, chat_history, documents=[])

print("Message history: ", chat_history)
print("")
print("Query originaria ", query)
print("Query modificata:")
print(response.data.chat_response.search_queries[0].text)
