import bibtexparser
from collections import defaultdict

# Load the BibTeX file
def load_bibtex(file_path):
    with open(file_path, 'r') as file:
        bib_database = bibtexparser.load(file)
    return bib_database

# Find duplicate entries by title
def find_duplicates(entries):
    title_count = defaultdict(int)
    duplicates = []

    for entry in entries:
        title = entry.get('title', '').strip().lower()
        title_count[title] += 1
        if title_count[title] > 1:
            if title not in duplicates:
                duplicates.append(title)

    return duplicates

# Main function to check for duplicates
def check_duplicates(file_path):
    bib_database = load_bibtex(file_path)
    entries = bib_database.entries
    duplicates = find_duplicates(entries)

    if duplicates:
        print("Duplicate entries found with titles:")
        for title in duplicates:
            print(f"- {title.title()}")
    else:
        print("No duplicate entries found.")

# Path to your BibTeX file
file_path = 'util/custom.bib'
check_duplicates(file_path)
