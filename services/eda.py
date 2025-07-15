import pandas as pd
from fastapi import UploadFile
import io

async def read_csv(file: UploadFile) -> pd.DataFrame:
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    return df


def generate_summary(df: pd.DataFrame):
    return df.describe(include='all').to_dict()