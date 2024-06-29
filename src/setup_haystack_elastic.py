from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline

# Initialize Elasticsearch document store
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

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

# Create a pipeline with the Retriever and Reader
pipeline = ExtractiveQAPipeline(reader, retriever)

# Save the reader to disk
reader.save("../data/reader")
