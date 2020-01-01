

## kaggle

```py
import os
os.environ['KAGGLE_USERNAME'] = "naesaranderzhang" # username from the json file
os.environ['KAGGLE_KEY'] = "d580334ca87c88ae66a1995d318dca80" # key from the json file
!kaggle datasets download -d sogun3/uspollution # api copied from kaggle

```

## pandas
在读取数据时就对数据类型进行转换，一步到位
```py

data2 = pd.read_csv("data.csv",
                   converters={
                               '客户编号': str,
                               '2016': convert_currency,
                               '2017': convert_currency,
                               '增长率': convert_percent,
                               '所属组': lambda x: pd.to_numeric(x, errors='coerce'),
                               '状态': lambda x: np.where(x == "Y", True, False)
                              },
                   encoding='gbk')
```


