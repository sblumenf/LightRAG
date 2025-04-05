from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://sergeblumenfeld:n9umsVxdrmjeBf8n@lightrag-tutor.6riqact.mongodb.net/?retryWrites=true&w=majority&appName=lightrag-tutor"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)