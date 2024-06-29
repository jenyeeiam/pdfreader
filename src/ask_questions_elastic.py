from functools import lru_cache

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers

# Initialize Elasticsearch document store
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

# Initialize the retriever and reader
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="../data/reader")

# Create a pipeline with the Retriever and Reader
pipeline = ExtractiveQAPipeline(reader, retriever)

@lru_cache(maxsize=32)
def get_retrieved_documents(query):
    return retriever.retrieve(query)

# Warm-up function to preload index data
def warm_up_index(pipeline):
    warm_up_queries = [
        "How long is the franchise term?",
        "How much do training days cost?"
    ]
    for query in warm_up_queries:
        pipeline.run(query=query)

# Warm-up the pipeline
warm_up_index(pipeline)

# List of questions to ask
questions = [
    "What is the initial investment fee for this franchise?",
    "What happens if I want to transfer the franchise to someone else?",
    "What lights are required for the decor? Is there a certain brand or type of light that must be used?",
]

for question in questions:
    print(f"Question: {question}")
    prediction = pipeline.run(
        query=question,
        params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
    )
    print_answers(prediction, details="minimum")
    print("\n" + "-"*50 + "\n")
