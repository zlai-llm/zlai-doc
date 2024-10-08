# 股票业绩数据

## 股票业绩

> 代码

```python
from zlai.tools.report import *

stock = StockPerformance(industry="酿酒行业", quarter="2024Q1", size=10)
data = stock.load_data()
print(data.metadata)
print(data.to_frame(columns=data.metadata.get("columns")).to_markdown())
```

> 数据

- `SECURITY_CODE`: `证券代码`
- `SECURITY_NAME_ABBR`: `证券简称`
- `TRADE_MARKET_CODE`: `交易市场代码`
- `TRADE_MARKET`: `交易市场`
- `SECURITY_TYPE_CODE`: `证券类型代码`
- `SECURITY_TYPE`: `证券类型`
- `UPDATE_DATE`: `更新日期`
- `REPORTDATE`: `报告日期`
- `BASIC_EPS`: `基本每股收益`
- `DEDUCT_BASIC_EPS`: `扣除后的基本每股收益`
- `TOTAL_OPERATE_INCOME`: `总营业收入`
- `PARENT_NETPROFIT`: `母公司净利润`
- `WEIGHTAVG_ROE`: `加权平均净资产收益率`
- `YSTZ`: `预计投资`
- `SJLTZ`: `实际投资`
- `BPS`: `每股净资产`
- `MGJYXJJE`: `每股经营现金流量`
- `XSMLL`: `销售毛利率`
- `YSHZ`: `预计收益回报`
- `SJLHZ`: `实际利润回报`
- `ASSIGNDSCRPT`: `分配描述`
- `PAYYEAR`: `分配年度`
- `PUBLISHNAME`: `发布名称`
- `ZXGXL`: `最新公告`
- `NOTICE_DATE`: `公告日期`
- `ORG_CODE`: `机构代码`
- `TRADE_MARKET_ZJG`: `交易市场指标`
- `ISNEW`: `是否新数据`
- `QDATE`: `报告季度`
- `DATATYPE`: `数据类型`
- `DATAYEAR`: `数据年度`
- `DATEMMDD`: `数据月份日期`
- `EITIME`: `导出时间`
- `SECUCODE`: `证券代号`


```text
|    |   证券代码 | 证券简称   |   交易市场代码 | 交易市场   |   证券类型代码 | 证券类型   | 更新日期            | 报告日期            |   基本每股收益 | 扣除后的基本每股收益   |   总营业收入 |   母公司净利润 |   加权平均净资产收益率 |   预计投资 |   实际投资 |   每股净资产 |   每股经营现金流量 |   销售毛利率 |   预计收益回报 |   实际利润回报 | 分配描述   | 分配年度   | 发布名称   | 最新公告   | 公告日期            |   机构代码 |   交易市场指标 |   是否新数据 | 报告季度   | 数据类型      |   数据年度 | 数据月份日期   | 导出时间            | 证券代号   |
|---:|-----------:|:-----------|---------------:|:-----------|---------------:|:-----------|:--------------------|:--------------------|---------------:|:-----------------------|-------------:|---------------:|-----------------------:|-----------:|-----------:|-------------:|-------------------:|-------------:|---------------:|---------------:|:-----------|:-----------|:-----------|:-----------|:--------------------|-----------:|---------------:|-------------:|:-----------|:--------------|-----------:|:---------------|:--------------------|:-----------|
|  0 |     600519 | 贵州茅台   |   069001001001 | 上交所主板 |      058001001 | A股        | 2024-04-27 00:00:00 | 2024-03-31 00:00:00 |        19.16   |                        |  4.64847e+10 |    2.40653e+10 |                  10.57 |   18.0436  |      15.73 |     190.839  |           7.31367  |      92.6133 |         2.7414 |        10.0989 |            |            | 酿酒行业   |            | 2024-04-27 00:00:00 |   10002602 |           0101 |            0 | 2024Q1     | 2024年 一季报 |       2024 | 一季报         | 2024-04-26 20:38:18 | 600519.SH  |
|  1 |     000596 | 古井贡酒   |   069001002001 | 深交所主板 |      058001001 | A股        | 2024-04-27 00:00:00 | 2024-03-31 00:00:00 |         3.91   |                        |  8.28632e+09 |    2.06584e+09 |                   9.16 |   25.8539  |      31.61 |      44.5978 |           4.64249  |      80.3496 |        92.6667 |       166.035  |            |            | 酿酒行业   |            | 2024-04-27 00:00:00 |   10005537 |           0201 |            0 | 2024Q1     | 2024年 一季报 |       2024 | 一季报         | 2024-04-27 01:18:19 | 000596.SZ  |
|  2 |     200596 | 古井贡B    |   069001002001 | 深交所主板 |      058001002 | B股        | 2024-04-27 00:00:00 | 2024-03-31 00:00:00 |         3.91   |                        |  8.28632e+09 |    2.06584e+09 |                   9.16 |   25.8539  |      31.61 |      44.5978 |           4.64249  |      80.3496 |        92.6667 |       166.035  |            |            | 酿酒行业   |            | 2024-04-27 00:00:00 |   10005537 |           0220 |            0 | 2024Q1     | 2024年 一季报 |       2024 | 一季报         | 2024-04-27 01:18:19 | 200596.SZ  |
|  3 |     002304 | 洋河股份   |   069001002001 | 深交所主板 |      058001001 | A股        | 2024-04-27 00:00:00 | 2024-03-31 00:00:00 |         4.0195 |                        |  1.62549e+10 |    6.05523e+09 |                  11.02 |    8.03329 |       5.02 |      38.4947 |           3.21981  |      76.0349 |       471.714  |      3328.78   |            |            | 酿酒行业   |            | 2024-04-27 00:00:00 |   10096578 |           0201 |            0 | 2024Q1     | 2024年 一季报 |       2024 | 一季报         | 2024-04-27 01:21:56 | 002304.SZ  |
|  4 |     000858 | 五粮液     |   069001002001 | 深交所主板 |      058001001 | A股        | 2024-04-29 00:00:00 | 2024-03-31 00:00:00 |         3.618  |                        |  3.48329e+10 |    1.40451e+10 |                  10.28 |   11.8631  |      11.98 |      36.9958 |           0.132985 |      78.4264 |        67.9854 |        90.374  |            |            | 酿酒行业   |            | 2024-04-29 00:00:00 |   10005740 |           0201 |            0 | 2024Q1     | 2024年 一季报 |       2024 | 一季报         | 2024-04-28 16:19:24 | 000858.SZ  |
|  5 |     000568 | 泸州老窖   |   069001002001 | 深交所主板 |      058001001 | A股        | 2024-04-27 00:00:00 | 2024-03-31 00:00:00 |         3.12   |                        |  9.1884e+09  |    4.57395e+09 |                  10.43 |   20.7387  |      23.2  |      31.4069 |           2.96159  |      88.3724 |        10.8294 |        70.6524 |            |            | 酿酒行业   |            | 2024-04-27 00:00:00 |   10005514 |           0201 |            0 | 2024Q1     | 2024年 一季报 |       2024 | 一季报         | 2024-04-27 01:19:21 | 000568.SZ  |
|  6 |     600809 | 山西汾酒   |   069001001001 | 上交所主板 |      058001001 | A股        | 2024-04-26 00:00:00 | 2024-03-31 00:00:00 |         5.13   |                        |  1.53381e+10 |    6.26243e+09 |                  20.22 |   20.9394  |      29.95 |      27.9493 |           5.7716   |      77.4609 |       195.873  |       521.945  |            |            | 酿酒行业   |            | 2024-04-26 00:00:00 |   10634767 |           0101 |            0 | 2024Q1     | 2024年 一季报 |       2024 | 一季报         | 2024-04-25 18:54:52 | 600809.SH  |
|  7 |     600702 | 舍得酒业   |   069001001001 | 上交所主板 |      058001001 | A股        | 2024-04-26 00:00:00 | 2024-03-31 00:00:00 |         1.6632 |                        |  2.10543e+09 |    5.50283e+08 |                   7.01 |    4.13391 |      -3.35 |      23.0275 |           0.325411 |      74.1585 |        14.6969 |        15.5688 |            |            | 酿酒行业   |            | 2024-04-26 00:00:00 |   10004012 |           0101 |            0 | 2024Q1     | 2024年 一季报 |       2024 | 一季报         | 2024-04-25 17:36:54 | 600702.SH  |
|  8 |     600600 | 青岛啤酒   |   069001001001 | 上交所主板 |      058001001 | A股        | 2024-04-30 00:00:00 | 2024-03-31 00:00:00 |         1.175  |                        |  1.015e+10   |    1.59724e+09 |                   5.65 |   -5.19482 |      10.06 |      21.3074 |           2.09146  |      40.4406 |       243.083  |       349.626  |            |            | 酿酒行业   |            | 2024-04-30 00:00:00 |   10002658 |           0101 |            0 | 2024Q1     | 2024年 一季报 |       2024 | 一季报         | 2024-04-29 16:58:46 | 600600.SH  |
|  9 |     603589 | 口子窖     |   069001001001 | 上交所主板 |      058001001 | A股        | 2024-04-30 00:00:00 | 2024-03-31 00:00:00 |         0.98   |                        |  1.7675e+09  |    5.8932e+08  |                   5.87 |   11.0524  |      10.02 |      17.2103 |          -0.173267 |      76.4818 |        16.5629 |        58.0885 |            |            | 酿酒行业   |            | 2024-04-30 00:00:00 |   10276866 |           0101 |            0 | 2024Q1     | 2024年 一季报 |       2024 | 一季报         | 2024-04-29 18:18:12 | 603589.SH  |
```

------
