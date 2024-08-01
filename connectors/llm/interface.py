import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

import connectors.config as cfg
from connectors.vector_store.db import vector_interface

PROMPT_TEMPLATE = """
<s>[INST]
You are an assistant that answers questions based on information found in technical documents.
You are given a set of document search results and must concisely answer a user's question.
Answers need to consider chat history. You must answer the question based solely on the content
in the search results. If you do not know the answer to a question, simply respond to the user
and tell them "I do not have enough context to be able to answer your question."

This is the user's question: {question}

Below are the document search results:
---

{context}

---
[/INST]
"""


class LLMInterface:
    def __init__(self):
        pass

    def ask(self, system_prompt, previous_messages, question, agent_id, stream):
        results = vector_interface.search(question, agent_id)

        prompt_params = {"question": question}
        prompt = ChatPromptTemplate.from_template("{question}")
        extra_doc_info = []
        if len(results) == 0 :
            print(f"Unable to find results")
            #return "I am lost"
        else:
            context_text = ""
            for i, doc_with_score in enumerate(results):
                page_content = doc_with_score[0].page_content
                metadata = doc_with_score[0].metadata
                extra_doc_info.append({"metadata": metadata, "page_content": page_content})
                print("metadata:", metadata)
                context_text += f"---\n<<Search Result {i+1}>>\n---\n{page_content}\n\n<<Search Result {i+1} END>>\n---\n"
            # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt_params = {"context": context_text, "question": question}
            print("search result:", context_text)

        # Adding system prompt and memory
        msg_list = []
        msg_list.append(SystemMessage(content=system_prompt))
        if previous_messages:
            for msg in previous_messages:
                if msg["sender"] == "human":
                    msg_list.append(HumanMessage(content=msg["text"]))
                if msg["sender"] == "ai":
                    msg_list.append(AIMessage(content=msg["text"]))
        prompt.messages = msg_list + prompt.messages

        print(prompt)
        model = ChatOpenAI(
            model=cfg.LLM_MODEL_NAME,
            openai_api_base=cfg.LLM_BASE_URL,
            openai_api_key=cfg.LLM_API_KEY
        )

        chain = prompt | model | StrOutputParser()

        if stream:
            def stream_generator():
                for chunks in chain.stream(prompt_params):
                    print("chunks:", chunks)
                    json_data = json.dumps({"text_content": chunks})
                    yield f"data: {json_data}\r\n"
                json_data = json.dumps({"search_metadata": extra_doc_info})
                yield f"data: {json_data}\r\n"
            return stream_generator

        response_text = chain.invoke(prompt_params)
        response = {"text_content": response_text, "search_metadata": extra_doc_info}
        return response

llm = LLMInterface()
