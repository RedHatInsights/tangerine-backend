from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="mistral")

text = "This is a test document. Hello! Hello! Hello!"

query_result = embeddings.embed_query(text)

print("len: ", len(query_result))

print(query_result[:5])
