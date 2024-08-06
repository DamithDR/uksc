import pandas as pd

tsv = pd.read_csv('outputs/chatgpt_outputs.tsv',sep='\t')

tsv.to_excel('outputs/chatgpt_outputs.tsv', index=False)