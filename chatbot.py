import os ## interacts with enviornment variable
import streamlit as st ## streamlit for UI
from langchain_groq import ChatGroq ## Groq llms integration 
from langchain.memory import ConversationBufferMemory ## memory backed for chat
from langchain.chains import ConversationChain
from dotenv import load_dotenv ## load .env into os.environ
from langchain_core.prompts import ChatPromptTemplate ## Define chat prompt template with system and user roles

## Load API KEY

load_dotenv() ## read .env file
os.environ["GROQ_API_KEY"] = os.getenv("Groq_API_KEY") # set groq key

## streamlit App setup

st.set_page_config(page_title="💬 Conversational Chatbot") ## title in browser tab 
st.title(" Medical  Assistant") ## app header

## sidebar controls

model_name = st.sidebar.selectbox( ## Pick a supported model
    "Select Groq Model",
    ["deepseek-r1-distill-llama-70b","gemma2-9b-it","llama3-70b-8192","llama3-8b-8192"]
)

temperature = st.sidebar.slider( ## Fix the randomness of the response
    "Temperature",0.0,1.0,0.7
)

max_tokens = st.sidebar.slider( ## max response length
     "max_tokens", 50,300,150
)


prompt= ChatPromptTemplate.from_messages( ## Define the chat prompt template
    [("system","you are a helpful medical assistant. Answer only health and medical related questions.if user asks something unrelated ,politely say its out of scope."), ## set system message to answer only health related questions
    ("user","{history}:{input}")
    ]
)
clear_chat = st.sidebar.button( ## button to clear chat history
    " 🗑 clear chat"
)
if clear_chat:
    st.session_state.history=[]
    st.session_state.memory=ConversationBufferMemory(return_messages=True)
    st.rerun()


# Intialize memory and history

if "memory" not in st.session_state:

## persist memory across rerun
    st.session_state.memory= ConversationBufferMemory(
        return_messages=True  ## return as list of messages, not one big string
    )

if "history" not in st.session_state:
    ## store role\content pairs display
    st.session_state.history=[]



## User Input 
user_input = st.chat_input("You :") ## clears itself on enter

if user_input:
    st.session_state.history.append(("user",user_input))



## Instantiated a fresh llm for this turn

    llm= ChatGroq(
        model_name=model_name,
        temperature=temperature,
        max_tokens = max_tokens
)

    conv = ConversationChain(
        llm = llm,
        memory= st.session_state.memory,
        prompt=prompt,
        verbose=True
)
    
## get ai response (memory is updated internally)
    ai_response = conv.predict(input=user_input)

## append assistant turn to visible history
    st.session_state.history.append(("assistant",ai_response))

## Render chat buble

for role, text in st.session_state.history:
    if role== "user":
        st.chat_message("user").write(text) ## User style
    else:
        st.chat_message("assistant").write(text) ## Assistant style
