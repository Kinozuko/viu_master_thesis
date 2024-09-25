import pandas as pd
import os

dataset_1 = os.path.join("datasets", 
    "Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion "\
    "Segmentation and Patient Meta Information")

def check_excels():
    """
    Check Excel files and see the columns they have
    """
    excel_1 = os.path.join(dataset_1,"Supplementary Table 1 for patient info .xlsx")
    excel_2  = os.path.join(dataset_1,"Supplementary Table 2 for  sequence parameters .xlsx")

    excel_1_pd = pd.read_excel(excel_1, header=1)
    print(f"Columns for {excel_1.split('/')[-1]}:\n")
    print(excel_1_pd.columns)

    print("\n")

    excel_2_pd = pd.read_excel(excel_2, header=1)
    print(f"Columns for {excel_2.split('/')[-1]}:\n")
    print(excel_2_pd.columns)

if __name__=="__main__":
    check_excels()