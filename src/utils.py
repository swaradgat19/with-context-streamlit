from multiprocessing import context
from pathlib import Path
from decouple import config
import pinecone
from openai import OpenAI
import json


OPENAI_API_KEY = config('OPENAI_API_KEY')
PINECONE_API_KEY = config('PINECONE_API_KEY')
PINECONE_ENV = config('PINECONE_ENV')
LIMIT = float("inf")

client = OpenAI(api_key=OPENAI_API_KEY)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

index_name = "openai-wc-hackathon-embeddings"
index = pinecone.Index(index_name)
index.describe_index_stats()

#! create embeddings

# query = 'Can you summarize the findings of MRNA-6231 in monkeys?'
# query = "Was there flaking skin in the monkey study using MRNA-6231?";

def get_embeddings(query, model = "text-embedding-ada-002"):
  
  embeddingResp = client.embeddings.create(
    input=[query],
    model=model
  )
  
  embeddings = embeddingResp.data[0].embedding
  
  return embeddings

def get_contexts_from_pinecone(embeddings):
  res = index.query(embeddings, top_k=3, include_metadata=True, include_values=True)
  contexts = [x['metadata']['context'] for x in res['matches']]
  return contexts

def get_prompt_message(query, contexts):
  prompt_start = (
      "Answer the question based on the context below.\n\n"+
      "Context:\n"
  )
  prompt_end = (
      f"\n\nQuestion: {query}"
  )

  for i in range(1, len(contexts)):
      joined_context = "\n\n---\n\n".join(contexts[:i])
      if len(joined_context) >= LIMIT:
          if i == 1:
              prompt = prompt_start + joined_context + prompt_end
          else:
              prompt = prompt_start + "\n\n---\n\n".join(contexts[:i - 1]) + prompt_end
          break
  else:
      prompt = prompt_start + "\n\n---\n\n".join(contexts) + prompt_end
  message = {"role": "system", "content": prompt}
  print('{MESSAGE}', message)
  return message

def get_no_rag_prompt_message(prompt):
  seed = "Use scientific reasoning to answer the following question \n Question: " + prompt
  message = {"role": "system", "content": seed}
  return message

def get_summary_resp(message, model = "gpt-4"):
  res = client.chat.completions.create(
      model=model,
      messages=[message],
      temperature=0,
      max_tokens=1024,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      stop=None
  )
  
  return res

def process_json(data):
  parsed_data = {}
  with open('data.json', 'r') as json_file:
      parsed_data = json.load(json_file)
      print(parsed_data)
        