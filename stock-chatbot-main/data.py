import pandas as pd

def row_to_text(row):
    return (
        f"Company {row['Company Name']} (symbol: {row['Symbol']}) in the {row['Industry']} industry "
        f"had an opening price of ₹{row['Open']}. The stock reached a high of ₹{row['High']} and a low of ₹{row['Low']}. "
        f"Its previous close was ₹{row['Previous Close']}, and the last traded price was ₹{row['Last Traded Price']}, "
        f"showing a change of ₹{row['Change']} or {row['Percentage Change']}%. The share volume was {row['Share Volume']}, "
        f"with a value of ₹{row['Value (Indian Rupee)']}. The 52-week high is ₹{row['52 Week High']} and the low is ₹{row['52 Week Low']}. "
        f"Over the past year, it changed {row['365 Day Percentage Change']}%, and in the last 30 days, it changed {row['30 Day Percentage Change']}%."
    )

def prepare_data(csv_path, output_path="/home/mt/Downloads/code/llm_rag/dataset.csv"):
    df = pd.read_csv(csv_path)
    df.fillna("unknown", inplace=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            text = row_to_text(row)
            f.write(text + "\n")

    print(f"Formatted {len(df)} rows into natural language format and saved to '{output_path}'.")

prepare_data("/home/mt/Downloads/code/llm_rag/nifty_500.csv")
