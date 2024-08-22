# 本地部署模型缓存管理

> 本模块用于管理本地部署的大模型缓存，包括查看缓存列表、清理缓存等。

```python
from zlai.models.cahce import ModelCache
model_cache = ModelCache(base_url="http://192.168.98.240:8000")
```

## 查看GPU内存占用

> 参数

- `:param device[int]:` GPU设备编号

> 示例

```python
model_cache.gpu_memory(device=0)
```

> 返回

```text
{'0': {'free': 69.42, 'total': 79.15, 'percent': 87.71}}
```

*以上输出结果表示，当前空闲显存`69.42GB`，显存总量`79.15GB`，空闲比例为`87.71GB`。*

## 查看模型缓存列表

> 示例

```python
model_cache.list_models()
```

> 返回

```text
['Qwen2-0.5B-Instruct', 'Qwen2-1.5B-Instruct']
```

*输出显示，当前有两个模型存在缓存，模型名称为`['Qwen2-0.5B-Instruct', 'Qwen2-1.5B-Instruct']`*

## 清理指定模型缓存

> 参数

- `:param model_name[Union[str, List[str]]]:` 模型名称，支持单个模型名称或模型名称列表

> 示例

```python
model_cache.clear_model_cache("Qwen2-0.5B-Instruct")
```

> 返回

```text
{'msg': ['Model Qwen2-0.5B-Instruct cache cleared.']}
```

## 清理所有模型缓存

> 示例

```python
model_cache.clear_gpu_memory()
```

> 返回

```text
{'message': 'GPU memory cache cleared.',
 'cleared_models': ['Qwen2-0.5B-Instruct']}
```

-----
