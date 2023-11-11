from email import message
from email.policy import default
import streamlit as st
from decouple import config
import utils

response = False
prompt_tokens = 0
completion_tokes = 0
total_tokens_used = 0
cost_of_response = 0

def make_request(question_input: str, rag_on: bool = False):
    
    if (rag_on):
        embds = utils.get_embeddings(question_input)
        contexts = utils.get_contexts_from_pinecone(embds)
        message = utils.get_prompt_message(question_input, contexts)
        response = utils.get_summary_resp(message)
    else: 
        message = utils.get_no_rag_prompt_message(question_input)
        response = utils.get_summary_resp(message)
    
    return response


st.header("CRO Document Store Search")
on = st.toggle('With Context 🤖')

st.markdown("""---""")

question_input = st.text_input("Enter question")
rerun_button = st.button("Run")

st.markdown("""---""")

if question_input:
    response = make_request(question_input, on)
else:
    pass

if rerun_button:
    response = make_request(question_input, on)
else:
    pass

if response:
    st.write("Response:")
    st.write(response.choices[0].message.content)
    prompt_tokens = response.usage.prompt_tokens
    completion_tokes = response.usage.completion_tokens
    total_tokens_used = response.usage.total_tokens
    cost_of_response = total_tokens_used * 0.000002
else:
    pass


with st.sidebar:
    # st.title("Usage Stats:")
    # st.markdown("""---""")
    # st.write("Promt tokens used :", prompt_tokens)
    # st.write("Completion tokens used :", completion_tokes)
    # st.write("Total tokens used :", total_tokens_used)
    # st.write("Total cost of request: ${:.8f}".format(cost_of_response))
    
    st.title("Team With Context")
    # st.markdown("""---""")
    st.write("Ajay Bhargava")
    st.write("Audrey Acken")
    st.write("Emma Chen")
    st.write("Swarad Gat")
    st.write("Sugam Devare")

