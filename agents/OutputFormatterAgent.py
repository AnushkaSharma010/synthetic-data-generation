from models.schemas import GeneratedData
import pandas as pd
from io import StringIO, BytesIO

class OutputFormatterAgent:
    def format(self, data: list, output_format: str) -> GeneratedData:
        """Formats data to requested output type"""

        # JSON output returns as-is
        if output_format == "json":
            return GeneratedData(data=data, format="json")

        df = pd.DataFrame(data)

        if output_format == "csv":
            buffer = StringIO()
            df.to_csv(buffer, index=False)
            csv_bytes = buffer.getvalue().encode("utf-8")
            return GeneratedData(file_content=csv_bytes, format="csv")

        elif output_format == "excel":
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False)
            return GeneratedData(file_content=buffer.getvalue(), format="excel")

        raise ValueError(f"Unsupported format: {output_format}")
