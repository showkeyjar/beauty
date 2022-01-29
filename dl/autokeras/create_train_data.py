import shutil
import pandas as pd

"""
创建训练目录
"""
database = "/opt/data/SCUT-FBP5500_v2/"
df_rates = pd.read_csv(database + "All_Ratings.csv")
df_rates = df_rates[df_rates['Filename'].str.find("AF")>=0]
df_rates_mean = df_rates.groupby('Filename').mean()


def copy_file(se):
    try:
        filename = str(se['Filename'])
        rate = str(int(se['Rating']))
        shutil.copyfile(database + "Images/train/face/" + filename, database + "score_data/train/" + rate + "/" + filename)
    except Exception as e:
        print(e)
        return False
    return True


if __name__=="__main__":
    result = df_rates.apply(copy_file, axis=1)
    print(result)
