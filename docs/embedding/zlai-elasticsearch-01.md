# ElasticSearch

> 什么是ElasticSearch？

Elasticsearch是一个强大的开源搜索和分析引擎，构建在Apache Lucene库之上。它为真实的实时存储、搜索和分析数据提供了一个分布式、可扩展的高性能平台。使用Elasticsearch，您可以快速索引大量数据，并使用复杂的查询检索相关信息。它的多功能性使其适用于广泛的用例，包括日志记录，监控和电子商务应用程序。

> 为什么使用Elasticsearch？

在深入研究技术细节之前，让我们简单地了解一下Elasticsearch比较优秀的原因：

* 全文检索：Elasticsearch擅长全文搜索，使用户能够在大量非结构化数据中搜索和检索特定的单词或短语。
* 可扩展性：Elasticsearch被设计为分布式的，允许您通过向集群添加更多节点来水平扩展，从而提高性能和容量。
* 实时分析：借助Elasticsearch，您可以对数据进行实时分析，获得有价值的见解和可视化，用于监控和决策。
* 易于集成：Elasticsearch与各种编程语言、库和工具无缝集成，使其成为不同生态系统的开发人员的选择。

## Install

> 使用Docker运行Elasticsearch

Docker提供了一个平台，用于将应用程序及其依赖项打包到轻量级、可移植的容器中。这种容器化方法允许您更方便地使用Elasticsearch实例，使其易于在不同环境中管理和部署。

如果您还没有安装Docker，请按照操作系统的官方安装说明在系统上安装Docker。

接下来，使用以下命令运行Elasticsearch容器：

```bash
# 下载ElasticSearch容器，8.12.2是版本号，您也可以使用其他版本号
docker pull elasticsearch:8.12.2

# 运行Elasticsearch容器
docker run \
  --name elasticsearch_container \    # 设置容器名称
  -p 9200:9200 \                      # 将容器端口9200映射到主机端口9200
  -p 9300:9300 \                      # 将容器端口9300映射到主机端口9300
  -e "discovery.type=single-node" \   # 设置单节点模式
  -e "xpack.security.enabled=false" \ # 禁用安全功能（可选）
  elasticsearch:8.12.2                 # 使用8.12.2版本镜像，您可以根据需要使用其他版本号
```

*Tips:*
* `--name`: 表示镜像启动后的容器名称
* `-d`: 后台运行容器，并返回容器ID；
* `-e`: 指定容器内的环境变量
* `-p`: 指定端口映射，格式为：主机(宿主)端口:容器端口


> 使用Docker Compose运行

Docker Compose是一个用于定义和运行多容器Docker应用程序的工具。它使您能够在声明性YAML文件中指定Elasticsearch配置，包括环境变量，网络设置和卷。让我们看看如何使用Docker Compose设置Elasticsearch：

如果您的系统上尚未安装Docker Compose，请参阅官方Docker Compose文档以获取安装说明。

创建一个名为docker-compose.yml的新文件，并在文本编辑器中打开它。将以下内容添加到文件中：
```bash
# docker-compose.yml

services:
  elasticsearch:
    image: elasticsearch:8.12.2
    container_name: elasticsearch
    ports:
      - 9200:9200
      - 9300:9300
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
```

然后，请在终端或命令提示符下，导航到包含`docker-compose.yml`文件的目录，并运行以下命令：

```bash
docker-compose up -d
```

> 验证安装，请访问：[http://localhost:9200/](http://localhost:9200/)

> 安装[`analysis-ik`](https://github.com/infinilabs/analysis-ik)

`Analysis-ik`是一个用于中文文本分析和处理的开源工具，主要用于中文分词。它是基于`ElasticSearch`的中文分词插件，提供了高效准确的中文分词功能。首先你需要下载与`ElasticSearch`版本一致的[analysis-ik](https://github.com/infinilabs/analysis-ik/releases)。然后按照以下步骤配置：

```bash
# 将压缩包移动到容器中
docker cp /tmp/elasticsearch-analysis-ik-8.12.2.zip elasticsearch:/usr/share/elasticsearch/plugins
# 进入容器
docker exec -it elasticsearch /bin/bash  
# 创建目录
mkdir /usr/share/elasticsearch/plugins/ik
# 将文件压缩包移动到ik中
mv /usr/share/elasticsearch/plugins/elasticsearch-analysis-ik-8.12.2.zip /usr/share/elasticsearch/plugins/ik
# 进入目录
cd /usr/share/elasticsearch/plugins/ik
# 解压
unzip elasticsearch-analysis-ik-8.12.2.zip
# 删除压缩包
rm -rf elasticsearch-analysis-ik-8.12.2.zip
```

*tips: 上面的安装目录需要根据自己的时间情况进行调整。*

**退出并重启镜像**

----
@2024/05/14

- [elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/8.13/docker.html)
- [elasticsearch-docker](https://geshan.com.np/blog/2023/06/elasticsearch-docker/)
