
import glob
import numpy as np
import pandas as pd
import os
import json
import joblib
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tarfile

try:
    from sagemaker_containers.beta.framework import (
        content_types,
        encoders,
        env,
        modules,
        transformer,
        worker,
        server,
    )
except ImportError:
    pass

label_column = "class"

base_dir = "/opt/ml/processing"

if __name__ == "__main__":
    df = pd.read_csv(f"{base_dir}/input/train.csv")
    df[df['categories']=='All Beauty'] = 'Beauty'
    df[df['categories']=='All Electronics'] = 'Electronics'
    df[df['categories']=='Baby'] = 'Baby Products'
    df['description'][df['description'].isna()] = ''
    df['text'] = df['title'].str.lower() + '. ' + df['description'].str.lower()
    train, test = train_test_split(df, test_size=0.2, stratify=df['categories'])
    
    map_cat = {k:v for v,k in enumerate(list(train['categories'].unique()))}
    train['categories_id'] = train['categories'].map(map_cat).astype(int)
    test['categories_id'] = test['categories'].map(map_cat).astype(int)
    train = train[['categories_id', 'text']]
    test = test[['ImgId', 'categories', 'categories_id', 'text']]

    train.to_csv(f"{base_dir}/train/data.csv", header=False, index=False)
    test.to_csv(f"{base_dir}/test/test.csv", header=True, index=False)

    json_string = json.dumps(map_cat)
    with open(f"{base_dir}/category_map/category_map.json", 'w') as outfile:
        outfile.write(json_string)