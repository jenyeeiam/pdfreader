from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline, DocumentSearchPipeline
from haystack.utils import print_answers
from functools import lru_cache

# Initialize Elasticsearch document store
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

# Read and process the extracted text
with open("../data/fdd_text.txt", "r") as file:
    pdf_text = file.read()

# Split the text into smaller chunks for better indexing
chunk_size = 1000
documents = [
    {"content": pdf_text[i:i+chunk_size], "meta": {"name": "fdd.pdf", "chunk_id": str(i/chunk_size)}}
    for i in range(0, len(pdf_text), chunk_size)
]

# Write documents to the document store
document_store.write_documents(documents)

# Initialize Retriever and Reader
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# Save the reader to disk
reader.save("../data/reader")

# Create a document search pipeline
search_pipeline = DocumentSearchPipeline(retriever)

# Create a QA pipeline with the Retriever and Reader
qa_pipeline = ExtractiveQAPipeline(reader, retriever)

# Caching retrieved documents
@lru_cache(maxsize=64)
def get_retrieved_documents(query):
    return retriever.retrieve(query=query)

# Warm-up function to preload index data
def warm_up_index(pipeline):
    warm_up_queries = [
        "How long is the franchise term?",
        "How much do training days cost?"
    ]
    for query in warm_up_queries:
        pipeline.run(query=query)

# Warm-up the pipeline
warm_up_index(qa_pipeline)

# Interactive Question Processing
reader = FARMReader(model_name_or_path="../data/reader")

while True:
    question = input("Enter your question (or 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    
    # Step 1: Retrieve relevant documents using Elasticsearch
    search_results = search_pipeline.run(query=question)
    retrieved_docs = search_results['documents']

    # Step 2: Use QA pipeline on the retrieved documents
    qa_results = qa_pipeline.run(
        query=question,
        params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}},
    )
    
    print(f"Question: {question}")
    print_answers(qa_results, details="minimum")
    print("\n" + "-"*50 + "\n")
