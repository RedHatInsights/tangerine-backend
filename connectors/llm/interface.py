import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import connectors.config as cfg
from connectors.vector_store.db import vector_interface

USER_PROMPT_TEMPLATE = """
Question: {question}

Answer the question using the following search results as context:

{context}
""".lstrip(
    "\n"
).rstrip(
    "\n"
)

DEFAULT_SYSTEM_PROMPT = """
<s>[INST] You are a helpful assistant for software developers who answers questions based on
information found in technical documents. You will be provided with a question and 6 search
results that may be useful information in answering the question. Each search result is
in markdown format. The search results are not ordered according to relevance. First,
select which search results appear to be most relevant for answering the user's question. Next,
provide a concise answer to the question by summarizing the relevant search results. You must
answer the question based solely on content found in the search results. Answers must also
consider chat history. If you are not able to answer a question, you should say "I do not have
enough information available to be able to answer your question. [/INST]</s>
""".lstrip(
    "\n"
).replace(
    "\n", " "
)

log = logging.getLogger("tangerine.llm")


class LLMInterface:
    def __init__(self):
        pass

    def ask(self, system_prompt, previous_messages, question, agent_id, stream):
        results = vector_interface.search(question, agent_id)

        prompt_params = {"question": question}
        prompt = ChatPromptTemplate.from_template("{question}")
        extra_doc_info = []
        if len(results) == 0:
            log.debug("unable to find results")
        else:
            context_text = ""
            for i, doc in enumerate(results):
                page_content = doc.page_content
                metadata = doc.metadata
                extra_doc_info.append({"metadata": metadata, "page_content": page_content})
                log.debug("metadata: %s", metadata)
                context_text += (
                    f"\n<<Search result {i+1}>>\n"
                    f"{page_content}\n\n"
                    f"<<Search result {i+1} END>>\n"
                )
            prompt = ChatPromptTemplate.from_template(USER_PROMPT_TEMPLATE)
            prompt_params = {"context": context_text, "question": question}
            log.debug("search result: %s", context_text)

        # Adding system prompt and memory
        msg_list = []
        msg_list.append(SystemMessage(content=system_prompt or DEFAULT_SYSTEM_PROMPT))
        if previous_messages:
            for msg in previous_messages:
                if msg["sender"] == "human":
                    msg_list.append(HumanMessage(content=msg["text"]))
                if msg["sender"] == "ai":
                    msg_list.append(AIMessage(content=msg["text"]))
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
