import base64
import pandas as pd
import io

def generate_download_link(df: pd.DataFrame, filename: str = "cleaned_data.csv") -> str:
    try:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV File</a>'
        return href
    except Exception as e:
        return f"⚠️ Error generating CSV download link: {e}"

def generate_excel_download_link(df: pd.DataFrame, filename: str = "cleaned_data.xlsx") -> str:
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        b64 = base64.b64encode(output.getvalue()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">📥 Download Excel File</a>'
        return href
    except Exception as e:
        return f"⚠️ Error generating Excel download link: {e}"
