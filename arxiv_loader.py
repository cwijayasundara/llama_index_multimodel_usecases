import arxiv
from pathlib import Path

research_paper_topics = ["RAG", "Agent"]

def download_papers(client, topics, num_results_per_topic):
    """Function to download papers from arxiv for given topics and number of results per topic"""
    for topic in topics:

        # sort by recent data and with max results
        search = arxiv.Search(
        query = topic,
        max_results = num_results_per_topic,
        sort_by = arxiv.SortCriterion.SubmittedDate
        )

        # get the results
        results = client.results(search)

        # download the pdf
        for r in results:
            r.download_pdf()

def list_pdf_files(directory):
    # List all .pdf files using pathlib
    pdf_files = [file.name for file in Path(directory).glob('*.pdf')]
    return pdf_files

client = arxiv.Client()

download_papers(client, research_paper_topics, 3)