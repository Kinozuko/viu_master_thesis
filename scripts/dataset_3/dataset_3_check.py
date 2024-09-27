import pandas as pd

filename = "datasets/MRIFreeDataset/" \
    "Initial & repeat MRI in MS-Free Dataset/PatientCodes-Names.xls"

test = pd.read_excel(filename)
print(test.columns)
print(test.head())
print(test['CODE'].head())