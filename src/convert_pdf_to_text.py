import fitz  # PyMuPDF

def convert_pdf_to_text(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page in document:
        text += page.get_text()
    return text

if __name__ == "__main__":
    pdf_path = "../data/FSHD - FDD - Signed.pdf"
    text = convert_pdf_to_text(pdf_path)
    with open("../data/fdd_text.txt", "w") as file:
        file.write(text)
