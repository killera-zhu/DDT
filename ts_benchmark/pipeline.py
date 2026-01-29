# -*- coding: utf-8 -*-
# 导入必要的模块和类
from dataclasses import dataclass
# 用于定义数据类，数据类是一种轻量级的类，主要用于存储数据，它会自动生成一些特殊方法，如 __init__、__repr__ 等，使代码更简洁
from functools import reduce
# 用于对可迭代对象进行累计操作，例如对一系列布尔值进行逻辑与操作
from operator import and_
# 逻辑与操作符，结合 reduce 函数可以对多个布尔表达式进行逻辑与运算
from typing import List, Dict, Type, Optional
# 类型提示，增强代码的可读性和可维护性，明确变量的类型。List 表示列表，Dict 表示字典，Type 用于表示类型，Optional 表示该变量可以为指定类型或 None

import pandas as pd
# 用于数据处理和分析，提供了 DataFrame 等数据结构，方便进行数据筛选、排序等操作

from ts_benchmark.data.data_source import (
    LocalForecastingDataSource,
    # 本地预测数据源类，用于从本地加载预测相关的数据
    DataSource,
    # 数据源基类，定义了数据源的基本接口和行为
)
from ts_benchmark.data.suites.global_storage import GlobalStorageDataServer
# 全局存储数据服务器类，负责管理和提供数据，支持异步数据加载
from ts_benchmark.evaluation.evaluate_model import eval_model
# 评估模型的函数，用于对模型在给定数据集上进行评估
from ts_benchmark.models import get_models
# 获取模型工厂列表的函数，根据配置信息返回一系列模型工厂
from ts_benchmark.recording import save_log
# 保存日志的函数，将评估结果保存到日志文件中
from ts_benchmark.utils.parallel import ParallelBackend
# 并行后端类，用于实现并行计算，提高程序的执行效率

# 定义一个数据类，用于存储数据集的信息
@dataclass
class DatasetInfo:
    # 数据集元信息中'size'字段的可能取值，是一个列表
    size_value: List
    # 该数据集的数据源类，使用 Type[DataSource] 表示它是 DataSource 类或其子类的类型
    datasrc_class: Type[DataSource]

# 预定义的数据集信息字典
PREDEFINED_DATASETS = {
    "large_forecast": DatasetInfo(
        size_value=["large", "small"],
        # 该数据集的 size 字段可能取值为 "large" 或 "small"
        datasrc_class=LocalForecastingDataSource
        # 该数据集使用本地预测数据源
    ),
    "small_forecast": DatasetInfo(
        size_value=["small"],
        # 该数据集的 size 字段可能取值为 "small"
        datasrc_class=LocalForecastingDataSource
        # 该数据集使用本地预测数据源
    ),
    "user_forecast": DatasetInfo(
        size_value=["user"],
        # 该数据集的 size 字段可能取值为 "user"
        datasrc_class=LocalForecastingDataSource
        # 该数据集使用本地预测数据源
    ),
}

# 根据给定的筛选条件过滤数据集
def filter_data(
    metadata: pd.DataFrame, size_value: List[str], feature_dict: Optional[Dict] = None
):
    """
    Filters the dataset based on given filters

    :param metadata: The meta information DataFrame.
    :param size_value: The allowed values of the 'size' meta-info field.
    :param feature_dict: A dictionary of filters where each key is a meta-info field
        and the corresponding value is the field value to keep. If None is given,
        no extra filter is applied.
    :return:
    """
    # 移除feature_dict中值为None的键值对
    if feature_dict is not None:
        feature_dict = {k: v for k, v in feature_dict.items() if v is not None}

    # 使用reduce和and_函数过滤符合条件的数据文件名
    filt_metadata = metadata
    if feature_dict is not None:
        # 对 feature_dict 中的每个键值对生成一个布尔表达式，判断 metadata 中对应列的值是否等于该键值对的值
        # 然后使用 reduce 和 and_ 函数将这些布尔表达式进行逻辑与运算，得到一个总的布尔索引
        filt_metadata = metadata[
            reduce(and_, (metadata[k] == v for k, v in feature_dict.items()))
        ]
    # 进一步筛选出 size 字段的值在 size_value 列表中的数据
    filt_metadata = filt_metadata[filt_metadata["size"].isin(size_value)]

    # 返回符合条件的数据的文件名列表
    return filt_metadata["file_name"].tolist()

# 重命名重复的模型名称
def _get_model_names(model_names: List[str]):
    """
    Rename models if there exists duplications.

    If a model A appears multiple times in the list, each appearance will be renamed to
    `A`, `A_1`, `A_2`, ...

    :param model_names: A list of model names.
    :return: The renamed list of model names.
    """
    # 将模型名称列表转换为 pandas 的 Series 对象
    s = pd.Series(model_names)
    # 对 Series 进行分组，统计每个模型名称出现的次数，返回一个包含累计计数的 Series
    cumulative_counts = s.groupby(s).cumcount()
    # 根据累计计数对模型名称进行重命名，如果计数大于 0，则在模型名称后面添加 "_计数"
    return [
        f"{model_name}_{cnt}" if cnt > 0 else model_name
        for model_name, cnt in zip(model_names, cumulative_counts)
    ]

# 执行基准测试的管道函数
def pipeline(
    data_config: dict,
    model_config: dict,
    evaluation_config: dict,
):
    """
    Execute the benchmark pipeline process

    The pipline includes loading data, building models, evaluating models, and generating reports.

    :param data_config: Configuration for data loading.
    :param model_config: Configuration for model construction.
    :param evaluation_config: Configuration for model evaluation.
    """
    # 准备数据
    # TODO: 在管道接口统一后，将这些代码移到数据模块中
    # 从数据配置中获取数据集名称列表，如果没有指定则默认使用 ["small_forecast"]
    dataset_name_list = data_config.get("data_set_name", ["small_forecast"])
    if not dataset_name_list:
        # 如果数据集名称列表为空，则使用默认值
        dataset_name_list = ["small_forecast"]
    if isinstance(dataset_name_list, str):
        # 如果数据集名称是字符串，则将其转换为列表
        dataset_name_list = [dataset_name_list]
    for dataset_name in dataset_name_list:
        if dataset_name not in PREDEFINED_DATASETS:
            # 检查数据集名称是否在预定义数据集中，如果不在则抛出异常
            raise ValueError(f"Unknown dataset {dataset_name}.")

    # 获取第一个数据集的数据源类型
    data_src_type = PREDEFINED_DATASETS[dataset_name_list[0]].datasrc_class
    if not all(
        PREDEFINED_DATASETS[dataset_name].datasrc_class is data_src_type
        for dataset_name in dataset_name_list
    ):
        # 检查所有数据集的数据源类型是否一致，如果不一致则抛出异常
        raise ValueError("Not supporting different types of data sources.")

    # 创建数据源对象
    data_src: DataSource = PREDEFINED_DATASETS[dataset_name_list[0]].datasrc_class()
    # 从数据配置中获取数据名称列表
    data_name_list = data_config.get("data_name_list", None)
    if not data_name_list:
        # 如果数据名称列表为空，则根据数据集名称和筛选条件获取数据名称列表
        data_name_list = []
        for dataset_name in dataset_name_list:
            size_value = PREDEFINED_DATASETS[dataset_name].size_value
            feature_dict = data_config.get("feature_dict", None)
            # 调用 filter_data 函数过滤数据
            data_name_list.extend(
                filter_data(
                    data_src.dataset.metadata, size_value, feature_dict=feature_dict
                )
            )
    # 去除数据名称列表中的重复项
    data_name_list = list(set(data_name_list))
    if not data_name_list:
        # 如果数据名称列表为空，则抛出异常
        raise ValueError("No dataset specified.")
    # 加载指定的数据系列
    data_src.load_series_list(data_name_list)
    # 创建全局存储数据服务器对象，传入数据源和并行后端
    data_server = GlobalStorageDataServer(data_src, ParallelBackend())
    # 启动数据服务器的异步加载
    data_server.start_async()

    # 建模
    # 根据模型配置获取模型工厂列表
    model_factory_list = get_models(model_config)

    # 对每个模型工厂在数据名称列表上进行评估，得到评估结果列表
    result_list = [
        eval_model(model_factory, data_name_list, evaluation_config)
        for model_factory in model_factory_list
    ]
    # 对模型名称进行重命名，确保名称唯一
    model_save_names = [
        it.split(".")[-1]
        for it in _get_model_names(
            [model_factory.model_name for model_factory in model_factory_list]
        )
    ]

    log_file_names = []
    for model_factory, result_itr, model_save_name in zip(
        model_factory_list, result_list, model_save_names
    ):
        # 遍历每个模型的评估结果
        for i, result_df in enumerate(result_itr.collect()):
            # 调用 save_log 函数保存评估结果到日志文件
            log_file_names.append(
                save_log(
                    result_df,
                    evaluation_config["save_path"],
                    model_save_name if i == 0 else f"{model_save_name}-{i}",
                )
            )

    return log_file_names