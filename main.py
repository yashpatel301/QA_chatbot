from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
load_dotenv()
def get_openai_response(question, temp=0.0, randomness=0.95, max_tokens=2000):
    #model = ChatOpenAI(temperature=0.5)
    model = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=temp, top_p=randomness, max_tokens=max_tokens)
    prompt = PromptTemplate(
        input_variables=["question"],
        template="You are a helpful assistant that provides answers to questions. {question}"
    )
    #response = llm(prompt.format(question=question))
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({"question": question})
    return response


st.set_page_config(page_title="Q&A Chatbot", page_icon=":guardsman:", layout="wide")
st.header("Q&A Chatbot")
question = st.text_input("Enter your question:", key="question_input")
creativity_level = st.slider("Select the creativity level (0-100):", 0, 100, 1)/100
randomness = st.slider("Select the randomness level (0-100):", 0, 100, 95)/100
max_tokens = st.slider("Select the max tokens:", 0, 4000, 2000)
submit = st.button("Ask the question")
if submit:
    st.header("The response is: ")
    if question:
        response = get_openai_response(question, temp=creativity_level, randomness=randomness, max_tokens=max_tokens)
        st.text_area("Response:", value=response, height=300)
    else:
        st.warning("Please enter a question.")