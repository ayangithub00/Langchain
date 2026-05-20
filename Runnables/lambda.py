from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel , RunnableLambda , RunnablePassthrough
from dotenv import load_dotenv


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.5,
)

def word_count(text):
    return len(text.split())

model = ChatHuggingFace(llm=llm)


prompt1 = PromptTemplate(
    template="Genrerate a tweet about /n {topic}",
    input_variables=['topic'])

parser = StrOutputParser()

chain = RunnableSequence(prompt1,model,parser)

par_chain = RunnableParallel({
    "tweet":RunnablePassthrough(),
    "word_count":RunnableLambda(word_count)
})

final_chain = RunnableSequence(chain,par_chain)

result = final_chain.invoke({'topic':"AI"})

final_result = """{} \n word count - {}""".format(result['tweet'], result['word_count'])

print(final_result)