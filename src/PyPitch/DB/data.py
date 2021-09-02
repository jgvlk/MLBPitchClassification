from pathlib import Path

from src.PyPitch.DB import query_raw_data


ret, df = query_raw_data()

cwd = Path.cwd()

out_file = cwd / 'MLBPitchData_2019.csv'

df.to_csv(out_file, index=False)

