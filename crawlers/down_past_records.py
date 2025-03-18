import json

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

total_pages = 55
ids = []
names = []
for p in tqdm(range(total_pages)):

    base_path = 'https://www.supremecourt.uk/cases?cs=Judgment+given&jded=2019-12-31&jdsd=2009-01-01&p=2&sort=judgmentDateOldest'

    data = [
        {
            "caseStatus": ["Judgment given"],
            "dateOfIssue": {"options": [], "date": {"startDate": "", "endDate": ""}},
            "permissionToAppealDecision": {"options": [], "date": {"startDate": "", "endDate": ""}},
            "hearingDate": {"startDate": "", "endDate": ""},
            "judgmentDate": {"startDate": "2009-01-01", "endDate": "2019-12-31"},
            "areaOfLaw": [],
            "jurisdiction": []
        },
        p,
        "",
        "judgmentDateOldest"
    ]

    headers = {
        "accept": "text/x-component",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "text/plain;charset=UTF-8",
        "cookie": "moj-consent=true; ARRAffinity=ec28c7dc1677604cdc4b664bae6a78096f0a18b3be3bff20e6d47356faea6b16; ARRAffinitySameSite=ec28c7dc1677604cdc4b664bae6a78096f0a18b3be3bff20e6d47356faea6b16; _ga=GA1.1.444261339.1718841811; _ga_RF2TC9QCFD=GS1.1.1731519871.73.1.1731519896.0.0.0; muxData=mux_viewer_id=4b3aa55e-e269-4c15-8500-8f040597a5fe&msn=0.6862646024407295&sid=b7882168-e326-4115-b91b-0e9fe9495cda&sst=1739259771567.4001&sex=1739261271567.4001; _ga_0MGBDT1XG3=GS1.1.1739259645.35.1.1739261303.0.0.0",
        "next-action": "12c792521f89313dc15b018b714c502aeb5fee9b",
        "next-router-state-tree": "%5B%22%22%2C%7B%22children%22%3A%5B%22cases%22%2C%7B%22children%22%3A%5B%22__PAGE__%22%2C%7B%7D%2C%22%2Fcases%3Fcs%3DJudgment%2Bgiven%26jded%3D2019-12-31%26jdsd%3D2010-01-01%26p%3D1%26sort%3DjudgmentDateOldest%22%2C%22refresh%22%5D%7D%5D%7D%2Cnull%2Cnull%2Ctrue%5D",
        "origin": "https://www.supremecourt.uk",
        "priority": "u=1, i",
        "sec-ch-ua": '"Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132")',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
    }

    response = requests.post(base_path, headers=headers, data=json.dumps(data))

    cases = json.loads(response.text.split("1:")[1])['caseData']

    for case in cases:
        ids.append(case['caseId'].replace('/', '-'))
        names.append(case['caseName'])

df = pd.DataFrame({'id': ids, 'name': names})
df.to_csv('data/past_data/download/historic_data.tsv', sep='\t', index=False)
