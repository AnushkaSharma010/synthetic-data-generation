from models.schemas import GeneratedData
import pandas as pd
import json
from io import StringIO, BytesIO

class OutputFormatterAgent:
    def format(self, data: list, output_format: str) -> GeneratedData:
        """Formats data to requested output type"""
        if output_format == "json":
            return GeneratedData(data=data, format="json")
        
        df = pd.DataFrame(data)
        
        if output_format == "csv":
            buffer = StringIO()
            df.to_csv(buffer, index=False)
            return GeneratedData(data=buffer.getvalue(), format="csv")
        
        elif output_format == "excel":
            buffer = BytesIO()
            with pd.ExcelWriter(buffer) as writer:
                df.to_excel(writer, index=False)
            return GeneratedData(data=buffer.getvalue(), format="excel")
        
        raise ValueError(f"Unsupported format: {output_format}")