# forecast-of-commodity-prices

Analysis of regression with some algorithms

## 任务 - 商品价格预测

### 任务背景

商品质量与其价格密切相关。通常，存在多种因素影响产品的质量。因此，如何根据产品质量影响因素，确定商品价格一个比较重要的问题。

围绕上述问题，请根据课程中所学到的数据预处理、预测模型，实现3种以上红酒质量预测方法，并比较不同方法的性能。

数据存在缺失，类别描述不统一等问题，需要进行数据清洗和预处理。

### 数据说明

本题目提供的数据来自西班牙的7500种不同类型的红酒，有11个特征描述了它们的价格、评级，甚至还有一些味道描述,其具体含义如下表所示

| 字段 | 解释 |
|-----|-----|
| Winery | 酒庄名称 |
| wine | 酒名 |
| year | 年份 |
| Rating | 用户打分 |
| Num_reviews | 评价用户数量 |
| Country | 国家 |
| Region | 地区 |
| Price | 价格 |
| Type | 类型 |
| Body | 身体评分，葡萄酒在口中的丰富性和重量 |
| acidity | 酸度分数 |

### 评测指标

请汇报常用的回归评价指标：MSE和回归曲线。

## 文件夹说明

- `datasets`
  - 存放原数据集【商品价格预测.csv】和处理后的中间数据
- `model`
  - 存放使用的模型，包括
    - Lasso
    - KNR
    - SVR
    - Design Tree
    - XGBoost
    - LightGBM
- `result`
  - 存放模型输出的 mat 文件
- `training_log`
  - 存放部分模型训练日志

## 文件说明

- `data_segmentation.ipynb`
  - 用于分割数据集，它会输出一个 `index.mat`，存放的是每一折的训练集样本索引和测试集样本索引。
  - `index.mat` 将会在模型运行时被读取。
- `feature_engineering_no_analysis.ipynb`
  - 基于原始 csv 文件生成特征，会输出一个 `candidate_data.mat`，包含了均值填充的特征，中值填充的特征，one-hot 特征以及 one-hot 降维的特征。
  - 这些特征还不能直接用于模型训练，还需要进行特征选择。
- `feature_selection.ipynb`
  - 基于 `candidate_data.mat` 选择特征
  - 代码中构造了 ridge 回归，每次基于 ridge 回归的性能变化来选择到底保留哪一种特征。
- `result_analysis.ipynb`
  - 用于分析结果，包含了指标的统计，回归图的绘制以及重要特征的分析。
