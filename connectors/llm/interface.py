import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import connectors.config as cfg
from connectors.vector_store.db import vector_interface

USER_PROMPT_TEMPLATE = """
[INST]
Question: {question}

Answer the above question using the below search results as context:

{context}
[/INST]
""".lstrip(
    "\n"
).rstrip(
    "\n"
)

DEFAULT_SYSTEM_PROMPT = """
 <s>[INST] You are an assistant who helps software developers quickly find answers to their
questions by reviewing technical documents. You answer their questions as concisely as possible and
point them to the technical documents where they can find more details. You will be provided with a
question and search results that appear to be most relevant for answering the question. The start
marker for each search result is similar to this: <<Search result 1>>. If the title of the document
is known, then the start marker result is similar to this: <<Search result 1, Document title: An
Example Title>>. The end marker of each search result is similar to this: <<Search result 1 END>>.
The content of the search result is found between the start marker and the end marker and is a
snippet of technical documentation in markdown format. The search results are ordered with the most
relevant results listed first and least relevant results listed last. Answer the question as
concisely as possible by using the content of the search results. Try to use the most relevant
search results when answering the question. If the first search result clearly answers the
question, then just use that search result and discard the others. If you are not able to answer a
question, you should say "I do not have enough information available to be able to answer your
question." Answers must consider chat history. Always assist with care, respect, and truth. Respond
with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure
replies promote fairness and positivity. [/INST]
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
                context_text += f"\n<<Search result {i+1}"
                if "title" in metadata:
                    title = metadata["title"]
                    context_text += f", document title: '{title}'"
                context_text += ">>\n\n" f"{page_content}\n\n" f"<<Search result {i+1} END>>\n"
            prompt = ChatPromptTemplate.from_template(USER_PROMPT_TEMPLATE)
            prompt_params = {"context": context_text, "question": question}
            log.debug("search result: %s", context_text)

        # Adding system prompt and memory
        msg_list = []
        msg_list.append(SystemMessage(content=system_prompt or DEFAULT_SYSTEM_PROMPT))
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
