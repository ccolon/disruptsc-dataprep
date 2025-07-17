import pandas as pd
import requests
import comtradeapicall

subscription_key = '4c1c9f93c54b4f54a14c48fc6c19133a'  # comtrade api subscription key (from comtradedeveloper.un.org)
directory = ''  # output directory for downloaded files


# comtradeapicall.listReference().to_csv('references.csv', index=False)

# ref_flow = comtradeapicall.getReference('flow')  # .to_csv('ref_flow.csv', index=False)
# ref_flow = ref_flow.set_index('text')['id'].to_dict()
# print(ref_flow)
#
# countries = ['AZE', 'ARM', 'GEO', 'KAZ', 'TKM', 'UZB', 'KGZ', 'TJK']
# ref_partner = comtradeapicall.getReference('partner')
# ref_partner = ref_partner[ref_partner['PartnerCodeIsoAlpha3'].isin(countries)]
# ref_partner = ref_partner.set_index('PartnerCodeIsoAlpha3')['id'].to_dict()
# print(ref_partner)

import requests

# URL of the JSON data
# url = "https://comtradeapi.un.org/files/v1/app/reference/HS.json"
# response = requests.get(url)
# response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
# data = response.json()  # Parse JSON response
# pd.DataFrame(data['results']).to_csv("ref_cmd_HS.csv", index=False)

mydf = comtradeapicall.previewFinalData(typeCode='C', freqCode='A', clCode='HS', period='2023',
                                        reporterCode=398, cmdCode='07', flowCode='X', partnerCode=0,
                                        partner2Code=None,
                                        customsCode=None, motCode=None, maxRecords=500, format_output='JSON',
                                        aggregateBy=None, breakdownMode='classic', countOnly=None, includeDesc=True)
print(mydf)
mydf.to_csv('test.csv')

data = mydf.iloc[0]
if data['qtyUnitAbbr'] == "kg":
    print(data["primaryValue"] / (data['qty'] / 1000))
