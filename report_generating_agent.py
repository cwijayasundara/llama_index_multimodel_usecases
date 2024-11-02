import warnings
from dotenv import load_dotenv
import nest_asyncio
from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse
from arxiv_loader import list_pdf_files
from llama_cloud.types import CloudDocumentCreate
from pydantic import BaseModel, Field
from typing import List
from llama_cloud.client import LlamaCloud
from llama_index.core.prompts import PromptTemplate
from llama_index.core.async_utils import run_jobs
import os

warnings.filterwarnings('ignore')
_ = load_dotenv()

nest_asyncio.apply()

llm = OpenAI(model="gpt-4o-mini")

def parse_files(pdf_files):
    """Function to parse the pdf files using LlamaParse in markdown format"""

    parser = LlamaParse(
        result_type="markdown",  # "markdown" and "text" are available
        num_workers=4,  # if multiple files passed, split in `num_workers` API calls
        verbose=True,
    )

    documents = []

    for index, pdf_file in enumerate(pdf_files):
        print(f"Processing file {index + 1}/{len(pdf_files)}: {pdf_file}")
        document = parser.load_data(pdf_file)
        documents.append(document)

    return documents

directory = './arxiv_papers'
pdf_files = list_pdf_files(directory)
documents = parse_files(pdf_files)

# utils
class Metadata(BaseModel):
    """Output containing the authors names, authors companies, and general AI tags."""
    author_names: List[str] = Field(..., description="List of author names of the paper. Give empty list if "
                                                     "not available")
    author_companies: List[str] = Field(..., description="List of author companies of the paper. Give empty "
                                                         "list if not available")
    ai_tags: List[str] = Field(..., description="List of general AI tags related to the paper. Give empty "
                                                "list if not available")

def create_llamacloud_pipeline(pipeline_name, embedding_config, transform_config, data_sink_id=None):
    """Function to create a pipeline in llamacloud"""

    client = LlamaCloud(token=os.environ["LLAMA_CLOUD_API_KEY"])

    pipeline = {
        'name': pipeline_name,
        'transform_config': transform_config,
        'embedding_config': embedding_config,
        'data_sink_id': data_sink_id
    }

    pipeline = client.pipelines.upsert_pipeline(request=pipeline)

    return client, pipeline

async def get_papers_metadata(text):
    """Function to get the metadata from the given paper"""
    prompt_template = PromptTemplate("""Generate authors names, authors companies, and general top 3 AI tags for the given research paper.

    Research Paper:

    {text}""")

    metadata = await llm.astructured_predict(
        Metadata,
        prompt_template,
        text=text,
    )

    return metadata

async def get_document_upload(document, llm):
    text_for_metadata_extraction = document[0].text + document[1].text + document[2].text
    full_text = "\n\n".join([doc.text for doc in document])
    metadata = await get_papers_metadata(text_for_metadata_extraction)
    return CloudDocumentCreate(
        text=full_text,
        metadata={
            'author_names': metadata.author_names,
            'author_companies': metadata.author_companies,
            'ai_tags': metadata.ai_tags
        }
     )

async def upload_documents(client, documents):
    """Function to upload the documents to the cloud"""

    # Upload the documents to the cloud
    extract_jobs = []
    for document in documents:
        extract_jobs.append(get_document_upload(document, llm))

    document_upload_objs = await run_jobs(extract_jobs, workers=4)

    _ = client.pipelines.create_batch_pipeline_documents(pipeline.id, request=document_upload_objs)