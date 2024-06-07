""""
Experimental client for Llama3 in OCI

last update: 07/06/2024
"""

from typing import Any, Dict, List, Optional
import logging
import json

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import oci
from oci.generative_ai_inference.models import GenericChatRequest, ChatDetails
from oci.generative_ai_inference.models import BaseChatRequest, TextContent, Message
from oci.generative_ai_inference.models import OnDemandServingMode
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.retry import NoneRetryStrategy


logger = logging.getLogger("oci_llama3")

# additional
OCI_CONFIG_DIR = "~/.oci/config"
TIMEOUT = (10, 240)


#
# supporting functions
#
def make_security_token_signer(oci_config):
    """
    to add
    """
    pk = oci.signer.load_private_key_from_file(oci_config.get("key_file"), None)

    with open(oci_config.get("security_token_file")) as f:
        st_string = f.read()

    return oci.auth.signers.SecurityTokenSigner(st_string, pk)


def get_generative_ai_dp_client(endpoint, profile, use_session_token):
    """
    create the client for OCI GenAI
    """
    config = oci.config.from_file(OCI_CONFIG_DIR, profile)

    if use_session_token:
        signer = make_security_token_signer(oci_config=config)

        client = GenerativeAiInferenceClient(
            config=config,
            signer=signer,
            service_endpoint=endpoint,
            retry_strategy=NoneRetryStrategy(),
            timeout=TIMEOUT,
        )
    else:
        client = GenerativeAiInferenceClient(
            config=config,
            service_endpoint=endpoint,
            retry_strategy=NoneRetryStrategy(),
            timeout=TIMEOUT,
        )

    return client


class OCILlama3(BaseChatModel):
    """
    This class wraps the code to use llama3 r in OCI

    Usage:
        chat = OCILlama3(
            model="meta.llama3-3-70b-instruct",
            service_endpoint="endpoint",
            compartment_id="ocid",
            max_tokens=512
        )
    """

    client: Any
    """ the client for OCI genai"""

    model_name = "command-r"

    model: str
    """the model_id"""
    temperature: Optional[float] = 0.1
    """the temperature"""
    top_k: Optional[int] = 1
    """top_k"""
    top_p: Optional[float] = 0.1
    max_tokens: Optional[int] = 512
    """max num of tokens in output"""
    service_endpoint: str = None
    """service endpoint in OCI"""
    compartment_id: str = None
    auth_type: Optional[str] = "API_KEY"
    auth_profile: Optional[str] = "DEFAULT"
    is_streaming: Optional[bool] = False

    def __init__(
        self,
        model: str,
        service_endpoint: str = None,
        compartment_id: str = None,
        temperature: Optional[float] = 0.1,
        top_k: Optional[int] = 1,
        top_p: Optional[float] = 0.1,
        max_tokens: Optional[int] = 512,
        auth_type: Optional[str] = "API_KEY",
        auth_profile: Optional[str] = "DEFAULT",
        is_streaming: Optional[bool] = False,
    ):
        """
        model: the id of the model
        temperature: the temperature
        top_k
        top_p
        max_tokens
        """
        super().__init__(
            model=model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens,
            service_endpoint=service_endpoint,
            compartment_id=compartment_id,
            auth_type=auth_type,
            auth_profile=auth_profile,
            is_streaming=is_streaming,
        )
        # init the client to OCI
        self.client = get_generative_ai_dp_client(
            self.service_endpoint,
            self.auth_profile,
            use_session_token=False,
        )

    def invoke(self, query: str, chat_history: List, documents: List):
        chat_detail = ChatDetails()

        content = TextContent()
        # here we set the user request
        content.text = f"{query}"
        message = Message()
        message.role = "USER"
        message.content = [content]

        chat_request = GenericChatRequest()
        chat_request.api_format = BaseChatRequest.API_FORMAT_GENERIC

        #  here we should also send the chat history
        chat_request.messages = [message]
        # parameters
        chat_request.max_tokens = self.max_tokens
        chat_request.temperature = self.temperature
        chat_request.top_p = self.top_p
        chat_request.top_k = self.top_k

        chat_detail.serving_mode = OnDemandServingMode(model_id=self.model)
        chat_detail.chat_request = chat_request
        chat_detail.compartment_id = self.compartment_id

        #
        # here we call the LLM
        #
        try:
            chat_response = self.client.chat(chat_detail)
        except Exception as e:
            logger.error("Error in invoke: %s", e)
            chat_response = None

        return chat_response

    def print_response(self, chat_response):
        """
        helper function to print handling streaming/no_streaming
        """
        print("")

        if self.is_streaming:
            for event in chat_response.data.events():
                res = json.loads(event.data)
                if "text" in res.keys():
                    print(res["text"], end="", flush=True)

            print("\n")
        else:
            # no streaming
            print(chat_response.data.chat_response.choices[0].message.content[0].text)
            print("")

    # for LangChain compatibility
    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return self.model_name

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Cohere API."""
        base_params = {
            "model": self.model,
            "temperature": self.temperature,
            "preamble": self.preamble,
        }
        return {k: v for k, v in base_params.items() if v is not None}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return self._default_params

    # We need this, for LangChain compatibility... complete !
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # simulate an output
        message = AIMessage(content="Hello,...")

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
