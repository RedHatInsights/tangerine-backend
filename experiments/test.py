from langchain_community.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.pgvector import PGVector

from langchain_community.embeddings import OllamaEmbeddings

CONNECTION_STRING = 'postgresql://citrus:citrus@localhost/citrus'

embeddings = OllamaEmbeddings(model="mistral")


loader = TextLoader("./experiments/generated_docs/2.txt")
documents = loader.load()
# documents = [Document(page_content=big_string, metadata={"agent_id": "1"})]
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
for doc in docs:
    doc.metadata["agent_id"] = "2"

print(docs)

# COLLECTION_NAME = "collection"

# # Initialize store
# store = PGVector(
#     collection_name=COLLECTION_NAME,
#     connection_string=CONNECTION_STRING,
#     embedding_function=embeddings,
# )

# # Add documents
# store.add_documents(docs)

# # Search documents
# docs_with_score = store.max_marginal_relevance_search_with_score(query="amazon civilsion", filter={"agent_id": "2"})
# for doc, score in docs_with_score:
#     print("-" * 80)
#     print("Score: ", score)
#     print(doc.page_content)
#     print(doc.metadata)
#     print("-" * 80)
