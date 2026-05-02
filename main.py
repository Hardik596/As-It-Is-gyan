from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embeddings = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma(
    persist_directory="Gita-db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="mmr",#It is a re-ranking algorithm that aims to balance relevance and diversity in the search results.
    search_kwargs={"k": 5,#k is the number of documents to fetch
    "fetch_k": 20, #fetch_k is the number of documents to fetch before re-ranking.
    "lambda_mult":0.5#lambda_mult is the diversity factor, between 0 and 1. Higher values will result in more diverse results.
                   }
)

llm = ChatMistralAI(model="mistral-small-2506")

#prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","""your an helpful AI assistant that helps the people deal with everything they are going on with in their situation.
         strictly stick with the context provided by the user and the retrieved documents. if you don't know the answer, say you don't know. never make up an answer.
         if no context is provided, say you don't know. always be concise and to the point. never add any extra information that is not relevant to the question asked by the user. 
         """),
         (
             "human",
             """Context:{context}
             Question:{question}"""
         )

    ]
)

print("RAG System created")
while True:
    query = input("you: ")
    if query =='exit':
        break
    docs = retriever.invoke(query)

    context = "".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({
        "context":context,
        "question":query
    })
    response = llm.invoke(final_prompt)
    print("\n AI: ",response.content)
