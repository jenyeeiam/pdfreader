
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline

# Initialize document store with BM25 support
document_store = InMemoryDocumentStore(use_bm25=True)

# Read the extracted text
with open("../data/fdd_text.txt", "r") as file:
    pdf_text = file.read()

# Convert the extracted text into a Haystack document format
documents = [{"content": pdf_text, "meta": {"name": "fdd.pdf"}}]

# Write documents to the document store
document_store.write_documents(documents)

# Initialize Retriever and Reader
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# Save the pipeline
pipeline = ExtractiveQAPipeline(reader, retriever)

# Save document store, retriever, and reader to disk for later use
# document_store.save("../data/document_store")
# retriever.save("../data/retriever")
reader.save("../data/reader")
