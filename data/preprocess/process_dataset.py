import pandas as pd

import re


def remove_square_brackets(text):
    pattern = r'\[.*?\]'
    result = re.sub(pattern, '', text)
    return result.strip()


if __name__ == '__main__':
    df = pd.read_excel('data/UKSC_dataset.xlsx', sheet_name='data')
    reasons = df['reasoning']
    reasons = list(map(lambda x: remove_square_brackets(x), reasons))
    reasons = list(map(lambda x: x.replace('\n\n', '\n'), reasons))
    reasons = list(map(lambda x: x.replace(', , .', '.'), reasons))
    reasons = list(map(lambda x: x.replace(' –,', ''), reasons))
    reasons = list(map(lambda x: x.replace(' .', '.'), reasons))
    reasons = list(map(lambda x: x.replace('. –', '.'), reasons))
    reasons = list(map(lambda x: x.replace(' , .', '.'), reasons))
    reasons = list(map(lambda x: x.replace('–.', '.'), reasons))
    df['reasoning'] = reasons
    df = df.drop(['full_judgement_text', 'gold_standard_reasoning'], axis=1)
    df.to_excel('data/test_data.xlsx', sheet_name='data', index=False)

    print('finished')
