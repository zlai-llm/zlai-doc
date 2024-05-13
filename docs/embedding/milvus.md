# milvus向量数据库

## milvus

> 什么是Milvus？

随着信息量和复杂性的增长，对存储和搜索大规模非结构化数据集的工具的需求也在增长。Milvus是一个开源的矢量数据库，可以有效地处理复杂的非结构化数据，如图像，音频和文本。对于需要高速和可扩展地访问大量数据集合的应用程序来说，它是一个受欢迎的选择。Milvus 是一个开源的向量数据库，用于为机器学习提供向量支持。它支持向量搜索、向量插入、向量更新和向量删除操作。Milvus 2.0是一个云原生矢量数据库，存储和计算被设计分离。这个重构版本的Milvus中的所有组件都是无状态的，以增强弹性和灵活性。

Milvus是一个开源矢量数据库，旨在处理大规模非结构化数据。它由先进的索引系统提供支持，并提供各种搜索算法，以有效地处理图像，音频和文本等高维数据。 您可以从Milvus获得以下功能：

1. 提高高维数据的搜索效率
2. 处理大规模数据集的可扩展性
3. 广泛支持各种搜索算法和索引技术
4. 广泛的应用，包括图像搜索、自然语言处理、推荐系统、异常检测、生物信息学和音频分析

要学习本教程，您需要安装最新版本的Docker。本教程依赖于Docker Compose，它已经包含在最新版本的Docker运行时中。要使用Milvus，您需要下载Milvus Python库和命令行界面（CLI）。确保您使用Python 3.9或更高版本，并注意CLI与Windows、macOS和Linux兼容。下面提供的示例shell命令适用于Linux系统，但也可用于macOS或Linux的Windows子系统。

## milvus向量数据库安装

*本小节需要先安装[docker](https://docs.docker.com/get-docker/)*

环境前提：docker-compose已经安装

1. 按照官网下载[docker-compose](https://github.com/milvus-io/milvus/releases)文件

```text
mkdir milvus
cd milvus
wget https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
```


2.修改docker-compose文件，添加attu部分
坑1:让attu和milvus在一个docker的network中，否则连接会报，因为他们不在一个docker网络中（docker-compse知识范畴）：

Error: Failed to connect to Milvus: Error: 14 UNAVAILABLE: No
connection establised

修改后的docker-compse.yml如下

```text
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.0
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2020-12-03T00-03-10Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
    depends_on:
      - "etcd"
      - "minio"
  # 增加attu
  attu:
    container_name: attu
    image: zilliz/attu:v2.3.10
    environment:
      MILVUS_URL: standalone:19530
    ports:
      - "3000:3000"
    depends_on:
      - "standalone"

networks:
  default:
    name: milvus
```


3.在文件夹milvus中执行docker-compose命令
windows环境，linux环境需要加sudo

docker-compse up -d
1
安装成功后如下图，我用的docker-desktop，否则需要用docker ps 命令查看。


3.http://localhost:3000/ 进入attu登录界面

直接点击connet按钮，成功进入！

进入[attu登录](http://localhost:3000/)界面

-------

- [Install Milvus Standalone with Docker](https://milvus.io/docs/install_standalone-docker.md)
- [Install Milvus Standalone with Docker](https://github.com/milvus-io/milvus-docs/blob/v2.4.x/site/en/getstarted/standalone/install_standalone-docker-compose.md)
- [milvus+attu向量数据库docker安装踩坑记录](https://blog.csdn.net/weixin_44535385/article/details/136902371)
- [How to Get Started with Milvus](https://milvus.io/blog/how-to-get-started-with-milvus.md)