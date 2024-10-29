import warnings
import nest_asyncio
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser

warnings.filterwarnings('ignore')
_ = load_dotenv()

nest_asyncio.apply()

embed_model = OpenAIEmbedding(model="text-embedding-3-small")

llm = OpenAI(model="gpt-4o-mini")

Settings.llm = llm

policy_doc = "docs/pb116349-business-health-select-handbook-1024-pdfa.pdf"

# Vanilla Approach - Parse the Policy with LlamaParse into Markdown

documents = LlamaParse(result_type="markdown").load_data(policy_doc)

# Markdown Element Node Parser
node_parser = MarkdownElementNodeParser(
    llm=OpenAI(model="gpt-4o-mini"), num_workers=8
)

nodes = node_parser.get_nodes_from_documents(documents)

base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

recursive_index = VectorStoreIndex(nodes=base_nodes + objects)

recursive_index.storage_context.persist(persist_dir="./insurance_policy_vanilla")

query_engine = recursive_index.as_query_engine(similarity_top_k=25)

# Querying the model for coverage
query_1 = "I have paid £450 for my glasses. How much of it can I claim back?"
response_1 = query_engine.query(query_1)
print(str(response_1))

# parsing with instructions
documents_with_instruction = LlamaParse(
    result_type="markdown",
    parsing_instruction="""
This document is an insurance policy.
When a benefits/coverage/exclusion is describe in the document amend to it add a text in the following benefits string 
format (where coverage could be an exclusion).

For {nameofrisk} and in this condition {whenDoesThecoverageApply} the coverage is {coverageDescription}. 

If the document contain a benefits TABLE that describe coverage amounts, do not output it as a table, but instead as a 
list of benefits string.

""",
).load_data(policy_doc)

node_parser_instruction = MarkdownElementNodeParser(
    llm=OpenAI(model="gpt-4o-mini"), num_workers=8
)

nodes_instruction = node_parser.get_nodes_from_documents(documents_with_instruction)
(
    base_nodes_instruction,
    objects_instruction,
) = node_parser_instruction.get_nodes_and_objects(nodes_instruction)

recursive_index_instruction = VectorStoreIndex(
    nodes=base_nodes_instruction + objects_instruction
)

recursive_index_instruction.storage_context.persist(persist_dir="./insurance_policy_instruction")

query_engine_instruction = recursive_index_instruction.as_query_engine(
    similarity_top_k=25
)

query_1 = "I have paid £450 for my glasses. How much of it can I claim back?"

response_1 = query_engine.query(query_1)
print("Vanilla:")
print(response_1)

print("With instructions:")
response_1_i = query_engine_instruction.query(query_1)
print(response_1_i)

query_2 = "I consulted a specialist for my back pain. How much can I claim back?"

response_2 = query_engine.query(query_2)
print("Vanilla:")
print(response_2)

print("With instructions:")
response_2_i = query_engine_instruction.query(query_2)
print(response_2_i)

query_3 = "Whats the claim amount for mental health consultations and how old should I be ?"

response_3 = query_engine.query(query_3)
print("Vanilla:")
print(response_3)

print("With instructions:")
response_3_i = query_engine_instruction.query(query_3)
print(response_3_i)