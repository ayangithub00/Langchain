from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel
from dotenv import load_dotenv


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.5,
)


model = ChatHuggingFace(llm=llm)


prompt1 = PromptTemplate(
    template="Genrerate a tweet about /n {topic}",
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template="Give me a linkdin post about /n {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1,model,parser),
    'linkdin': RunnableSequence(prompt2,model,parser)
})

result = chain.invoke({'topic':'AI'})

print(result)