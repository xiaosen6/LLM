# 在 PostgreSQL 中进行向量存储与检索

为了在 PostgreSQL 中完成向量（Embedding）数据的存储与相似度检索，推荐使用官方扩展 [`pgvector`](https://github.com/pgvector/pgvector)。下面按照准备环境、创建表结构、写入数据、建立索引以及执行检索的顺序，给出一个完整的流程示例。

## 1. 安装 `pgvector` 扩展

```bash
# 以 Ubuntu + PostgreSQL 16 为例
sudo apt-get install postgresql-16-pgvector

# 或者在已有数据库实例中执行（需要超级用户权限）
CREATE EXTENSION IF NOT EXISTS vector;
```

## 2. 创建向量存储表

假设希望保存文本及其向量，可创建如下数据表（以 1536 维 OpenAI Embedding 为例）：

```sql
CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    content     TEXT NOT NULL,
    embedding   vector(1536)
);
```

## 3. 写入向量数据

```sql
INSERT INTO documents (content, embedding)
VALUES
    ('示例文本 A', '[0.01, 0.02, ...]'),
    ('示例文本 B', '[0.03, 0.04, ...]');
```

> 注意：`pgvector` 支持直接通过字符串字面量 `[...]` 写入向量；在实际代码中可以使用参数化 SQL 绑定数组或 Python `list`。

如果使用 Python，可以结合 `psycopg` 与大模型接口（如 OpenAI / DeepSeek）批量写入：

```python
import psycopg
from openai import OpenAI

conn = psycopg.connect('postgresql://user:password@localhost:5432/demo')
client = OpenAI()

texts = [
    "向量数据库让语义检索更简单",
    "PostgreSQL 也能做向量检索"
]

with conn, conn.cursor() as cur:
    for text in texts:
        emb = client.embeddings.create(model="text-embedding-3-large", input=text)
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (text, emb.data[0].embedding)
        )
```

## 4. 创建相似度索引

```sql
-- 选择常见的 HNSW 近似最近邻索引
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw
    ON documents
    USING hnsw (embedding vector_cosine_ops);

-- 也可以根据需求选择 L2、内积等操作符类
-- vector_l2_ops / vector_ip_ops / vector_cosine_ops
```

## 5. 相似度检索示例

### 5.1 直接在 SQL 中按余弦相似度排序

```sql
SELECT id, content,
       1 - (embedding <=> '[0.05, 0.06, ...]') AS cosine_similarity
FROM documents
ORDER BY embedding <=> '[0.05, 0.06, ...]'
LIMIT 5;
```

`<=>` 是 `pgvector` 定义的距离运算符：

- `vector_l2_ops` 下为欧氏距离（越小越相似）；
- `vector_ip_ops` 下为内积（越大越相似，通常结合 `ORDER BY embedding <#> query_vector DESC`）；
- `vector_cosine_ops` 下为余弦距离（值越小越相似）。

### 5.2 通过应用服务检索

```python
import psycopg
from openai import OpenAI

query = "Postgres 怎么做向量搜索？"
query_emb = client.embeddings.create(
    model="text-embedding-3-large",
    input=query
).data[0].embedding

with conn.cursor() as cur:
    cur.execute(
        """
        SELECT id, content, 1 - (embedding <=> %s) AS score
        FROM documents
        ORDER BY embedding <=> %s
        LIMIT 5
        """,
        (query_emb, query_emb)
    )
    for row in cur.fetchall():
        print(row)
```

## 6. 更新与删除

向量列与普通列一样支持 `UPDATE` 与 `DELETE`：

```sql
UPDATE documents
SET embedding = '[0.11, 0.12, ...]'
WHERE id = 1;

DELETE FROM documents
WHERE id = 2;
```

## 7. 日常维护建议

1. **定期重建索引**：大量更新 / 删除后执行 `REINDEX` 或重建 `hnsw` 索引，保证检索质量。  
2. **向量归一化**：若使用余弦相似度，可在写入前将向量单位化，提高查询准确度。  
3. **批量导入**：大量向量导入时先关闭索引，导入完再创建索引以提升速度。  
4. **监控存储**：向量列会占用较多空间，注意 `TOAST` 参数与表膨胀情况。

按照以上步骤，即可在 PostgreSQL 中完成向量数据的存储和高效检索，结合现有业务系统实现语义搜索、推荐等功能。
