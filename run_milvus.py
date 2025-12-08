from pymilvus import connections

# 使用本地文件作为存储
connections.connect("default", uri="./milvus_lite.db")