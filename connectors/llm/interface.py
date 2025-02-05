import json
import logging
import time

from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import connectors.config as cfg
from connectors.db.vector import vector_db

log = logging.getLogger("tangerine.llm")


class LLMInterface:
    def __init__(self):
        pass

    def ask(self, system_prompt, previous_messages, question, agent_id, stream):
        log.debug("querying vector DB")
        results = vector_db.search(question, agent_id)

        prompt_params = {"question": question}
        prompt = ChatPromptTemplate.from_template("{question}")
        extra_doc_info = []

        context_text = ""

        if len(results) == 0:
            log.debug("unable to find results")
            context_text = "No matching search results found"
        else:
            log.debug("fetched %d results", len(results))
            for i, doc in enumerate(results):
                page_content = doc.page_content
                metadata = doc.metadata
                extra_doc_info.append({"metadata": metadata, "page_content": page_content})

                context_text += f"\n<<Search result {i+1}"
                if "title" in metadata:
                    title = metadata["title"]
                    context_text += f", document title: '{title}'"
                context_text += ">>\n\n" f"{page_content}\n\n" f"<<Search result {i+1} END>>\n"

        prompt = ChatPromptTemplate.from_template(cfg.USER_PROMPT_TEMPLATE)
        prompt_params = {"context": context_text, "question": question}

        # Adding system prompt and memory
        msg_list = []
        msg_list.append(SystemMessage(content=system_prompt or cfg.DEFAULT_SYSTEM_PROMPT))
        if previous_messages:
            for msg in previous_messages:
                if msg["sender"] == "human":
                    msg_list.append(HumanMessage(content=f"[INST] {msg['text']} [/INST]"))
                if msg["sender"] == "ai":
                    msg_list.append(AIMessage(content=f"{msg['text']}</s>"))
        prompt.messages = msg_list + prompt.messages

        llm = ChatOpenAI(
            model=cfg.LLM_MODEL_NAME,
            openai_api_base=cfg.LLM_BASE_URL,
            openai_api_key=cfg.LLM_API_KEY,
            temperature=cfg.LLM_TEMPERATURE,
        )

        chain = prompt | llm

        log.debug("prompting llm...")

        def get_llm_response():
            timer_start = None

            with get_openai_callback() as cb:
                for chunk in chain.stream(prompt_params, stream_usage=True):
                    if not timer_start:
                        # this is the first output token received
                        timer_start = time.time()
                    if len(chunk.content):
                        text_content = {"text_content": chunk.content}
                        yield text_content

                timer_end = time.time()
                search_metadata = {"search_metadata": extra_doc_info}
                yield search_metadata

            log.debug(
                "input tokens: %s, output tokens: %s, total tokens: %s",
                cb.prompt_tokens,
                cb.completion_tokens,
                cb.total_tokens,
            )
            output_rate = cb.completion_tokens / (timer_end - timer_start)
            log.debug("output rate: %f tokens/sec", output_rate)

        def generator():
            for data in get_llm_response():
                yield f"data: {json.dumps(data)}\r\n"

        if stream:
            return generator

        # else, if stream=False ...
        response = {"text_content": None, "search_metadata": None}
        for data in get_llm_response():
            if "text_content" in data:
                if response["text_content"] is None:
                    response["text_content"] = data["text_content"]
                else:
                    response["text_content"] += data["text_content"]
            if "search_metadata" in data:
                response["search_metadata"] = data["search_metadata"]
        return response


llm = LLMInterface()
