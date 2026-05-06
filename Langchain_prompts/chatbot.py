from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from dotenv import load_dotenv


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.5,
)


model = ChatHuggingFace(llm=llm)

chat_history=[
    SystemMessage(content="You r a helpful AI Assistant ")
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    else:
       result =  model.invoke(chat_history)
       chat_history.append(AIMessage(content=result.content))
       print("AI: ",result.content)

print(chat_history)