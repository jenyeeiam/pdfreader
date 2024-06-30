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

# Function to ask question and get answers
def ask_question(question, qa_pipeline):
    qa_results = qa_pipeline.run(
        query=question,
        params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}},
    )

    # Extract answers and their context
    answers = qa_results['answers']

    # Display the answers with their context
    for i, answer in enumerate(answers):
        print(f"Answer {i+1}: {answer.answer}")
        print(f"Context: {answer.context}")
        print(f"Score: {answer.score}")
        print("\n" + "-"*50 + "\n")

    return answers

# Function to choose an answer
def choose_answer(answers):
    chosen_index = int(input("Choose the answer number you prefer (1, 2, 3, etc.): ")) - 1
    return answers[chosen_index]

# Function to find the chosen answer in the original text
def find_answer_in_text(answer, pdf_text):
    # Get the answer text and the surrounding context
    answer_text = answer.answer
    context_text = answer.context
    
    # Find the position of the context text in the original pdf_text
    context_start_idx = pdf_text.find(context_text)
    if context_start_idx == -1:
        print("Context not found in the original text.")
        return
    
    # Find the line containing the answer
    start_idx = pdf_text.find(answer_text, context_start_idx)
    end_idx = start_idx + len(answer_text)

    # Split the text into lines
    lines = pdf_text.split('\n')

    # Find the line number of the answer
    start_line_num = pdf_text.count('\n', 0, start_idx)
    end_line_num = pdf_text.count('\n', 0, end_idx)

    # Determine the range of lines to print
    start_print_line = max(start_line_num - 5, 0)
    end_print_line = min(end_line_num + 5, len(lines))

    # Print the relevant lines
    print(f"Answer line and context:")
    for line_num in range(start_print_line, end_print_line):
        print(lines[line_num])

# Interactive Question Processing
reader = FARMReader(model_name_or_path="../data/reader")

while True:
    question = input("Enter your question (or 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    
    # Ask the question and get answers
    answers = ask_question(question, qa_pipeline)
    
    # Allow the user to choose an answer
    chosen_answer = choose_answer(answers)
    
    # Find the chosen answer in the original text
    find_answer_in_text(chosen_answer, pdf_text)
