import json
import logging
import time
from typing import Generator, List

from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import connectors.config as cfg
from connectors.db.vector import vector_db
from resources.metrics import get_counter, get_gauge

log = logging.getLogger("tangerine.llm")

agent_response_counter = get_counter(
    "agent_response_counter", "Total number of responses for an agent", ["agent_id", "agent_name"]
)
llm_completion_tokens_metric = get_counter("llm_completion_tokens", "LLM completion tokens usage")
llm_prompt_tokens_metric = get_counter("llm_prompt_tokens", "LLM prompt tokens usage")
llm_completion_rate = get_gauge(
    "llm_completion_rate", "Observed tokens per sec from most recent LLM chat completion"
)
llm_processing_rate = get_gauge(
    "llm_processing_rate", "Observed tokens per sec for most recent LLM processing after prompted"
)
llm_no_answer = get_counter("llm_no_answer", "No Answer provided by the bot")


class LLMInterface:
    def __init__(self):
        pass

    @staticmethod
    def _record_metrics(
        cb: OpenAICallbackHandler,
        processing_start: float,
        completion_start: float,
        completion_end: float,
    ) -> None:
        if not completion_start:
            log.error("no content in llm response stream")
            return

        processing_time = completion_start - processing_start
        completion_time = completion_end - completion_start

        try:
            processing_rate = cb.prompt_tokens / processing_time
            completion_rate = cb.completion_tokens / completion_time
        except ZeroDivisionError:
            log.error("unexpected time diff of 0")
            completion_rate = 0

        log.debug(
            (
                "prompt tokens: %s, completion tokens: %s, "
                "processing time: %fsec (%f tokens/sec), completion time: %fsec (%f tokens/sec)"
            ),
            cb.prompt_tokens,
            cb.completion_tokens,
            processing_time,
            processing_rate,
            completion_time,
            completion_rate,
        )
        llm_completion_tokens_metric.inc(cb.completion_tokens)
        llm_prompt_tokens_metric.inc(cb.prompt_tokens)
        llm_processing_rate.set(processing_rate)
        llm_completion_rate.set(completion_rate)

    @staticmethod
    def _get_response(
        chat: ChatOpenAI,
        prompt: ChatPromptTemplate,
        prompt_params: dict,
        extra_doc_info: List[dict],
    ) -> Generator[dict, None, None]:
        chain = prompt | chat

        completion_start = 0.0
        processing_start = time.time()

        with get_openai_callback() as cb:
            for chunk in chain.stream(prompt_params, stream_usage=True):
                if not completion_start:
                    # this is the first output token received
                    completion_start = time.time()
                if len(chunk.content):
                    text_content = {"text_content": chunk.content}
                    yield text_content

            completion_end = time.time()
            search_metadata = {"search_metadata": extra_doc_info}
            yield search_metadata

        LLMInterface._record_metrics(cb, processing_start, completion_start, completion_end)

    def ask(self, system_prompt, previous_messages, question, agent_id, agent_name, stream, interaction_id=None):
        log.debug("querying vector DB")
        results = vector_db.search(question, agent_id)

        prompt_params = {"question": question}
        prompt = ChatPromptTemplate.from_template("{question}")
        extra_doc_info = []

        context_text = ""

        if len(results) == 0:
            log.debug("unable to find results")
            context_text = "No matching search results found"
            llm_no_answer.labels(agent_id=agent_id, agent_name=agent_name).inc()
        else:
            agent_response_counter.labels(agent_id=agent_id, agent_name=agent_name).inc()
            log.debug("fetched %d relevant search results from vector db", len(results))
            for i, doc in enumerate(results):
                page_content = doc.document.page_content
                metadata = doc.document.metadata
                extra_doc_info.append({"interactionId": interaction_id, "metadata": metadata, "page_content": page_content})
                

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

        chat = ChatOpenAI(
            model=cfg.LLM_MODEL_NAME,
            openai_api_base=cfg.LLM_BASE_URL,
            openai_api_key=cfg.LLM_API_KEY,
            temperature=cfg.LLM_TEMPERATURE,
        )

        log.debug("prompting llm...")
        llm_response = LLMInterface._get_response(chat, prompt, prompt_params, extra_doc_info)

        def api_response_generator():
            for data in llm_response:
                yield f"data: {json.dumps(data)}\r\n"

        if stream:
            log.debug("streaming response...")
            return api_response_generator

        # else, if stream=False ...
        response = {"text_content": None, "search_metadata": None}
        for data in llm_response:
            if "text_content" in data:
                if response["text_content"] is None:
                    response["text_content"] = data["text_content"]
                else:
                    response["text_content"] += data["text_content"]
            if "search_metadata" in data:
                response["search_metadata"] = data["search_metadata"]
        return response


llm = LLMInterface()
