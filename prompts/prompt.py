from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
import streamlit as st

load_dotenv()
model_google = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


# llm = HuggingFaceEndpoint(


#     repo_id="deepseek-ai/DeepSeek-R1",
#     task="text-generation",
# )

# model = ChatHuggingFace(llm=llm)

st.header("Chat Models with Langchain")

st.title("huggingface chat model")
research_paper = st.selectbox("select research paper name", options=["select...", "Attention all you need", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "RoBERTa: A Robustly Optimized BERT Pretraining Approach"])

explanation_style = st.selectbox("set Explanation style", options=["select...","beginner friendly" ,"simple","code oriented", "detailed", "technical"])

length=st.selectbox("set response length", options=["select...", "short(1-200 words)", "medium(201-400 words)", "long(401+ words)"])

template = load_prompt("template.json")

if st.button("Submit"):
    if research_paper == "select..." or length == "select..." or explanation_style == "select...":
        st.error("Please select all options")
    else:
        chain= template | model_google
        response = chain.invoke({
            "paper_title": research_paper,
            "length": length,
            "summary_style": explanation_style
        })
        st.write(response.content)