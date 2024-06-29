from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers

# Initialize document store with BM25 support
document_store = InMemoryDocumentStore(use_bm25=True)

# Read the extracted text
with open("../data/fdd_text.txt", "r") as file:
    pdf_text = file.read()

# Convert the extracted text into a Haystack document format
documents = [{"content": pdf_text, "meta": {"name": "fdd.pdf"}}]

# Write documents to the document store
document_store.write_documents(documents)

# Load the retriever and reader
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="../data/reader")

# Create a pipeline with the Retriever and Reader
pipeline = ExtractiveQAPipeline(reader, retriever)

# Ask a question
question = "What is the initial investment fee for this franchise?"
prediction = pipeline.run(
    query=question,
    params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
)

# Print the answers
print_answers(prediction, details="minimum")
