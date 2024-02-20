import json

from connectors.vector_store.db import vector_interface
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
Here are some search results:

{context}

---

Answer this question based solely on the above context: {question}
"""

class LLMInterface:
    def __init__(self):
        pass

    def ask(self, system_prompt, previous_messages, question, agent_id, stream):
        results = vector_interface.search(question,agent_id)

        prompt_params = {"question": question}
        prompt = ChatPromptTemplate.from_template("{question}")
        if len(results) == 0 :
            print(f"Unable to find results")
            #return "I am lost"
        else:
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt_params = {"context": context_text, "question": question}

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
        model = ChatOllama(model="mistral")

        chain = prompt | model | StrOutputParser()

        if stream:
            def stream_generator():
                for chunks in chain.stream(prompt_params):
                    print("chunks:", chunks)
                    yield json.dumps({"text_content": chunks}) + "\n"
            return stream_generator

        response_text = chain.invoke(prompt_params)
        return response_text

llm = LLMInterface()
