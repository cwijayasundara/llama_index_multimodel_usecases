import warnings
from dotenv import load_dotenv
import nest_asyncio
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

warnings.filterwarnings('ignore')
_ = load_dotenv()

nest_asyncio.apply()

claim_form_store_location = "./claim_forms_store"

# Index the documents.
embed_model = OpenAIEmbedding(model="text-embedding-3-large")

llm = OpenAI("gpt-4o")

Settings.llm = llm
Settings.embed_model = embed_model

ctx = StorageContext.from_defaults(persist_dir=claim_form_store_location)
index = load_index_from_storage(ctx)

query_engine = index.as_query_engine()

response = query_engine.query(
    "Who filed the insurance claim for the accident that happened on Sunset Blvd?"
)
print((str(response)))

response = query_engine.query("How did Ms. Patel's accident happen?")
print((str(response)))

chat_engine = index.as_chat_engine()
response = chat_engine.chat(
    "Given the accident that happened on Lombard Street, name a party that is liable for the damages and explain why."
)
print((str(response)))