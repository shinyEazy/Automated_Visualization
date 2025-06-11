import pandas as pd

class GetDataInCsv:
    def __init__(self):
        pass

    def get_data_in_csv(self, csv_path: str) -> str:
        df = pd.read_csv(csv_path)
        if df.empty:
            return "CSV file is empty."
        first_row = df.iloc[0]
        return first_row.to_json() 

if __name__ == "__main__":
    csv_path = r"C:\Users\ASUS\Desktop\workspace\Automated_Visualization\backend\problems\text_classification_verified\data\goemotions.csv"
    print(GetDataInCsv().get_data_in_csv(csv_path))