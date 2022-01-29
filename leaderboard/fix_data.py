import os
import glob
import pandas as pd

"""
修复 dataframe 错误
"""

all_images = glob.glob("../data/face/*.jpg")
all_images = [img.replace("\\", "/").replace("../", "") for img in all_images]
df_records = pd.DataFrame({'img_path': all_images})
df_records['img_id'] = df_records['img_path'].apply(lambda x:x.split("/")[-1].replace(".jpg", ""))
df_records['class'] = None
df_records['score'] = None
df_records.to_hdf("../data/face_record.h5", "df")
