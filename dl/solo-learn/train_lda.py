import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation

"""
使用lda自动聚类boyl学习到的特征
"""

def train_model(X, class_num=20):
    lda = LatentDirichletAllocation(n_components=class_num, random_state=0)
    lda.fit(X)
    return lda


if __name__=="__main__":
    df_data = pd.read_feather("/mnt/nfs196/soft/skin/cheek_boyl.ftr")
    train_x = df_data['data'].values
    lda_model = train_model(train_x)

