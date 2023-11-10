from cgitb import text
import csv
import json
from multiprocessing import context
import pandas as pd
import utils

parsed_data = {}
with open('data.json', 'r') as json_file:
    parsed_data = json.load(json_file)
    print(parsed_data)
    

for idx, data in enumerate(parsed_data):
  table = data[f'{idx}']['Tables']
  df = pd.DataFrame(table)
  csv_string = df.to_csv(None, sep=",", index=False)
  # print(csv_string)

  
def text_to_vector_embeddings(data):
    """This is a function that takes in the S3Response and returns a list of embeddings."""
    MODEL = "text-embedding-ada-002"
    body = data
    text = []
    animal = ["Monkey", "Rat"]
    tables = []
    for i, item in enumerate(body):
        passage = {"Text": item[str(i)]['Paragraphs']['text']}
        text.append(passage)
        
        table = item[str(i)]['Tables']
        df = pd.DataFrame(table)
        csv_string = df.to_csv(None, sep=",", index=False)
        print(csv_string)
        tables.append(csv_string)
  
    passage = [
        f"Animal the study was conducted in: {animal}\n\n" + 
        f"The Drug: MRNA-6231\n\n" + 
        f"The study summary: {text['Text']} \n\n" +
        f"Tables regarding the study: \n\n" + table
        for animal, text, table in zip(animal, text, tables)
    ]
    print(passage)
    embeddings = []
    dims = 0
    for i in range(len(passage)):
        response = utils.client.embeddings.create(
            input = [
                passage[i]
            ],
            model = MODEL,
        )
        embeds = [record.embedding for record in response.data]
        dims = len(embeds[0])
        embeddings.append({
            "id": str(i+2),
            "values": embeds[0],
            "metadata": {
                "animal": animal[i],
                "drug": "MRNA-6231",
                "context": passage[i]
            }
        })
    return embeddings, dims
  
embeddings, dim = text_to_vector_embeddings(parsed_data)
utils.index.upsert(embeddings)