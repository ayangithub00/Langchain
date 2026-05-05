from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.5,
)


model = ChatHuggingFace(llm=llm)


while True:
    user_input = input('You')
    if user_input == "exit":
        break
    else:
       result =  model.invoke(user_input)
       print("AI: ",result.content)