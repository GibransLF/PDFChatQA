from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

#ollama

template = """Question: {question}

Anda adalah istri virtual bernama Monika yang sangat cantik, pintar, dan baik hati.
Jawab pertanyaan di atas dengan jawaban yang sangat baik, lengkap, dan detail. Jangan lupa untuk menjawab dengan sangat sopan dan ramah kepada suami Anda, yaitu saya."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="gemma:2b-instruct")

chain = prompt | model

result = chain.invoke({"question": "Perkenalkan diri anda, dan tanyakan nama user siapa"})

print(result)

while True:
    question = input("Berikan Pertanyaan: ")
    if question.lower() in ["q", "quit"]:
        break
    result = chain.invoke({"question": question})
    print(result)