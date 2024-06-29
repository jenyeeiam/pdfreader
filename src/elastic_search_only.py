from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever
from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import print_documents

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

# Initialize Retriever
retriever = BM25Retriever(document_store=document_store)

# Create a document search pipeline
search_pipeline = DocumentSearchPipeline(retriever)

# Warm-up function to preload index data
def warm_up_index(pipeline):
    warm_up_queries = [
        "How long is the franchise term?",
        "How much do training days cost?"
    ]
    for query in warm_up_queries:
        pipeline.run(query=query)

# Warm-up the pipeline
warm_up_index(search_pipeline)

# List of questions to ask
questions = [
    "What is the initial investment fee for this franchise?",
    "What happens if I want to transfer the franchise to someone else?",
    "What lights are required for the decor? Is there a certain brand or type of light that must be used?",
]

# Function to print retrieved documents
def print_retrieved_documents(documents, max_text_len=200):
    for doc in documents:
        print(f"Document ID: {doc.id}")
        print(f"Content: {doc.content[:max_text_len]}...")
        print(f"Meta: {doc.meta}")
        print("\n" + "-"*50 + "\n")

# Run the questions through the search pipeline
for question in questions:
    search_results = search_pipeline.run(query=question)
    documents = search_results['documents']
    
    print(f"Question: {question}")
    print_retrieved_documents(documents, max_text_len=200)
    print("\n" + "-"*50 + "\n")
