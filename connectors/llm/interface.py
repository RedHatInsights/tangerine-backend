import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import connectors.config as cfg
from connectors.db.vector import vector_interface

log = logging.getLogger("tangerine.llm")


class LLMInterface:
    def __init__(self):
        pass

    def ask(self, system_prompt, previous_messages, question, agent_id, stream):
        results = vector_interface.search(question, agent_id)

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
                context_text += ">>\n\n" f"{page_content}\n\n" f"<<Search result {i+1} END>>\n"

        prompt = ChatPromptTemplate.from_template(cfg.USER_PROMPT_TEMPLATE)
        prompt_params = {"context": context_text, "question": question}
        log.debug("search result: %s", context_text)

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

        log.debug("prompt: %s", prompt)
        model = ChatOpenAI(
            model=cfg.LLM_MODEL_NAME,
            openai_api_base=cfg.LLM_BASE_URL,
            openai_api_key=cfg.LLM_API_KEY,
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
