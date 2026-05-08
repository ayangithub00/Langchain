from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.5,
)


model = ChatHuggingFace(llm=llm)


# 1st prompt detailed prompt 
template1=PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

# 1st prompt detailed prompt 
template2=PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables=['text']
)


prompt1 = template1.invoke({'topic':'black hole'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})

result = model.invoke(prompt2)

print(result.content)