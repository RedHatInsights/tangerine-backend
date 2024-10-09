import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage as LangchainSystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.messages import SystemMessage as MistralSystemMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from pydantic import SecretStr

import connectors.config as cfg
from connectors.db.vector import vector_db

log = logging.getLogger("tangerine.llm")


class LLMInterface:
    def __init__(self):
        pass

    def ask(self, system_prompt, previous_messages, question, agent_id, stream):
        results = vector_db.search(question, agent_id)

        prompt_params = {"question": question}
        prompt = ChatPromptTemplate.from_template("{question}")
        extra_doc_info = []

        context_text = ""

        if len(results) == 0:
            log.debug("unable to find results")
            context_text = "No matching search results found"
        else:
            for i, doc in enumerate(results):
                page_content = doc.page_content
                metadata = doc.metadata
                extra_doc_info.append({"metadata": metadata, "page_content": page_content})
                log.debug("metadata: %s", metadata)
                context_text += f"\n<<Search result {i+1}"
                if "title" in metadata:
                    title = metadata["title"]
                    context_text += f", document title: '{title}'"
                context_text += (">>\n\n" f"{page_content}\n\n" f"<<Search result {i+1} END>>\n")

        prompt = ChatPromptTemplate.from_template(cfg.USER_PROMPT_TEMPLATE)
        prompt_params = {"context": context_text, "question": question}
        log.debug("search result: %s", context_text)

        text_splitter = MistralTokenizer.v3(is_tekken=True)

        # Adding system prompt and memory
        msg_list = []
        msg_list.append(
            LangchainSystemMessage(content=system_prompt or cfg.DEFAULT_SYSTEM_PROMPT)
        )

        prompt_content = MistralSystemMessage(content=msg_list[0].content)
        total_tokens = len(text_splitter.encode_chat_completion(ChatCompletionRequest(messages=[prompt_content])).tokens)
        if previous_messages:
            # Reverse list so most recent msgs are in context
            for msg in reversed(previous_messages):
                # Declare before if to avoid 'unbound' errors
                m = HumanMessage(content=f"")
                token_list = 0

                if msg["sender"] == "human":
                    token_list = len(text_splitter.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=msg["text"])])).tokens)
                    m = HumanMessage(content=f"[INST] {msg['text']} [/INST]")
                if msg["sender"] == "ai":
                    # The tokenizer requires that every request begins with a
                    # SystemMessage or a UserMessage, so we tokenize the AI
                    # response as a UserMessage, but append to the list as an
                    # AIMessage.
                    token_list = len(text_splitter.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=f"{msg['text']}</s>")])).tokens)
                    m = AIMessage(content=f"{msg['text']}</s>")

                total_tokens += token_list
                if token_list + total_tokens >= cfg.MAX_TOKENS_CONTEXT:
                    print()
                    log.debug("Too many tokens, trimming context...")
                    break

                msg_list.append(m)

        prompt.messages = msg_list + prompt.messages

        log.debug("prompt: %s", prompt)
        model = ChatOpenAI(
            model=cfg.LLM_MODEL_NAME,
            base_url=cfg.LLM_BASE_URL,
            api_key=SecretStr(cfg.LLM_API_KEY),
            temperature=cfg.LLM_TEMPERATURE,
        )

        chain = prompt | model | StrOutputParser()

        if stream:

            def stream_generator():
                for chunks in chain.stream(prompt_params):
                    log.debug("chunks: %s", chunks)
                    json_data = json.dumps({"text_content": chunks})
                    yield f"data: {json_data}\r\n"
                json_data = json.dumps({"search_metadata": extra_doc_info})
                yield f"data: {json_data}\r\n"

            return stream_generator

        response_text = chain.invoke(prompt_params)
        response = {"text_content": response_text, "search_metadata": extra_doc_info}
        return response


llm = LLMInterface()
