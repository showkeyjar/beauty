import pandas as pd
import pandas_profiling

"""
分析数据
"""

df_lables = pd.read_csv('../data/face/label.csv')

profile = df_lables.profile_report(style={'full_width': True})

profile.to_file(output_file="output1.html")