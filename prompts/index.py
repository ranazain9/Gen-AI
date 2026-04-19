from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import streamlit as st
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

st.header("Chat Models with Langchain")

st.title("huggingface chat model")
research_paper = st.selectbox("select research paper name", options=["select...", "Attention all you need", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "RoBERTa: A Robustly Optimized BERT Pretraining Approach"])

explanation_style = st.selectbox("set Explanation style", options=["select...","beginner friendly" ,"simple","code oriented", "detailed", "technical"])


length=st.selectbox("set response length", options=["select...", "short(1-200 words)", "medium(201-400 words)", "long(401+ words)"])
    

template = PromptTemplate(
    template="""
### ROLE
Identity: Elite Academic Synthesis Engine (v3.0)
Objective: Extract, Distill, and Project research data based on specific user constraints.

### INPUT PARAMETERS
- Document Title: "{paper_title}"
- Word Budget (L): {length} words
- Persona Filter (S): "{summary_style}"

### SYSTEM CONSTRAINTS & LOGIC
1. **Verification Protocol**: Confirm "{paper_title}" via internal knowledge or search. 
   - If match = 0, Output: "STATUS: NULL - DATASET MISMATCH."
2. **Entropy Control**: Maintain word count within 5% of {length}.
3. **Information Distribution**: 
   - Context/Thesis: 0.25 * L
   - Methodology/Logic: 0.35 * L
   - Outcomes/Style-Specific Detail: 0.40 * L

### STYLE-DRIVEN PROJECTION (S)
Modify the linguistic 'Texture' based on "{summary_style}":
- [Standard]: Balanced, neutral, IMRaD focus.
- [The Skeptic]: High-gravity focus on p-hacking, sample size, and limitations.
- [The Architect]: Focus on the 'Building Blocks' (Code/Math/Theorems).
- [The Layman]: Use a 'Bridge Analogy'—explain the tech using a common household object.

### OUTPUT SCHEMA (Markdown Required)
# 📄 Document: {paper_title}
**Profile:** {summary_style} | **Constraint:** {length} words

---

## 🔍 The Abstract View
[Summarize the 'What' and the 'Why']

## 🛠️ Technical Synthesis & Analogy
[Explain the core mechanism. If {summary_style} is "Architect", include a snippet of logic or pseudocode.]

## 📊 Qualitative/Quantitative Impact
[The 'So What?' of the research results]

---
**TL;DR Vector:** [One-sentence impact statement]
""",
    input_variables=["paper_title", "length", "summary_style"]

)
prompt = template.invoke({
    "paper_title":research_paper,
    "length":length,
   "summary_style":explanation_style
})
if st.button("Submit"):
    response = model.invoke(prompt)
    st.write(response.content)