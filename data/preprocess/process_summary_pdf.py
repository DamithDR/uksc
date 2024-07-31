import os

import fitz


def pdf_to_text(pdf_path):
    # Open the PDF file
    document = fitz.open(pdf_path)
    text = ""

    # Iterate through each page
    for page_num in range(document.page_count):
        # Extract text from the current page
        page = document.load_page(page_num)
        text += page.get_text()

    return text

import PyPDF2

def pdf_to_text_pypdf2(pdf_path):
    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        # Iterate through each page
        for page in reader.pages:
            text += page.extract_text()
    return text



def run():
    paths = ['data/press-summaries/2024/', 'data/press-summaries/2023/']

    for path in paths:
        files = os.listdir(path)
        for file in files:
            if str(file).endswith('.pdf'):
                text = pdf_to_text_pypdf2(f'{path}{file}')
                name = file.split('.')[0]
                # Save the text to a file or print it
                with open(f'data/press-summaries/extracted_txt/{name}.txt', 'w', encoding='utf-8') as f:
                    f.write(text)

                print("PDF text extraction complete. Check the file.")
    print('running')


if __name__ == '__main__':
    run()
