from connectors.vector_store.db import vector_interface
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

class LLMInterface:
    def __init__(self):
        pass


    def ask(self, question, agent_id):
        results = vector_interface.search(question,agent_id)
        if len(results) == 0 :
            print(f"Unable to find results")
            #return "I am lost"
        
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=question)
        print(prompt)
        model = ChatOllama(model="mistral")
        response_text = model.predict(prompt)
        return response_text
    
llm = LLMInterface()