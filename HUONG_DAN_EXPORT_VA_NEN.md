# Hướng dẫn Export Workflow từ DB & Nén File Docker

> Tài liệu này dùng ví dụ thực tế từ project `~/FIS/tradeflat/n8n`

---

## Phần 1: Export N8N Workflows từ PostgreSQL

### Bối cảnh

- N8N chạy với `DB_TYPE=postgresdb`, workflows lưu trong bảng `workflow_entity`
- Dữ liệu PostgreSQL nằm tại: `~/FIS/tradeflat/n8n/database/postgres_data/`
- Image PostgreSQL: `pgvector/pgvector:0.6.2-pg15`
- Credentials: `postgres / 123456aA@ / database: n8n`

---

### Bước 1: Khởi PostgreSQL container tạm

```bash
docker run -d --name temp-postgres \
  -e POSTGRES_DB=n8n \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=123456aA@ \
  -v /home/vietpv/FIS/tradeflat/n8n/database/postgres_data:/var/lib/postgresql/data \
  -p 3099:5432 \
  pgvector/pgvector:0.6.2-pg15 \
  postgres -c max_connections=300
```

**Lưu ý:**
- Port `3099` → chọn port chưa bị chiếm (port `3004` đã dùng bởi pptx-slides-postgres)
- Volume mount trỏ đúng tới folder `postgres_data` chứa raw data
- Image phải đúng với image đã tạo database (`pgvector/pgvector:0.6.2-pg15`)
- Credentials phải khớp với file `.env` gốc (`~/FIS/tradeflat/n8n/.env`)

---

### Bước 2: Kiểm tra database

```bash
# Đợi postgres khởi động
sleep 3

# Liệt kê bảng
docker exec temp-postgres psql -U postgres -d n8n -c "\dt"

# Xem danh sách workflows
docker exec temp-postgres psql -U postgres -d n8n \
  -c "SELECT id, name, active FROM workflow_entity ORDER BY id;"
```

Kết quả mẫu:
```
        id        |                 name                 | active
------------------+--------------------------------------+--------
 fsJY9c1sKbhykI7m | OpenAI 0. Main HS Code V2            | t
 4R4MUeXpvcfsfM3T | OpenAI 1. Workflow Select Chapter     | t
 gSv6TXEKSXYsyYFP | OpenAI 2. Workflow Select Group      | t
 3xFYvC25yLSfzpg4 | OpenAI 3. Select Item ID             | t
 hegiWhArOy6a8imy | VBPL Chatbot                         | t
 ...              | (tổng 33 workflows)                  |
```

---

### Bước 3: Export workflows

#### Cách 1: Dùng Python script (khuyên dùng)

```bash
# Cài thư viện
pip install psycopg2-binary

# Chạy script export
python3 /home/vietpv/FIS/n8n/database/export_workflows.py
```

Script sẽ tự kết nối tới `127.0.0.1:3099`, query bảng `workflow_entity`, và lưu ra:
- `~/FIS/tradeflat/n8n/database/tradeflat_export/` — mỗi workflow 1 file JSON
- `~/FIS/tradeflat/n8n/database/tradeflat_export/_all_workflows.json` — file tổng hợp

Kết quả mẫu:
```
  ✅ active | OpenAI 0. Main HS Code V2 -> OpenAI_0._Main_HS_Code_V2_fsJY9c1sKbhykI7m.json
  ✅ active | OpenAI 1. Workflow Select Chapter -> OpenAI_1._Workflow_Select_Chapter_4R4MUeXpvcfsfM3T.json
  ⬜ inactive | My workflow -> My_workflow_KTgWqq9O3zDeoX8q.json
  ...
============================================================
Exported 33 workflows to: /home/vietpv/FIS/tradeflat/n8n/database/tradeflat_export
```

#### Cách 2: Dùng SQL thuần (nhanh, không cần Python)

Export tất cả workflows ra 1 file JSON:
```bash
docker exec temp-postgres psql -U postgres -d n8n -t -A -c \
  "SELECT json_agg(row_to_json(w)) FROM workflow_entity w;" \
  > /home/vietpv/FIS/tradeflat/n8n/database/all_workflows.json
```

Export 1 workflow cụ thể:
```bash
docker exec temp-postgres psql -U postgres -d n8n -t -A -c \
  "SELECT row_to_json(w) FROM workflow_entity w WHERE name = 'OpenAI 0. Main HS Code V2';" \
  > /home/vietpv/FIS/tradeflat/n8n/database/workflow_openai_main.json
```

---

### Bước 4: Dọn dẹp container tạm

```bash
docker stop temp-postgres && docker rm temp-postgres
```

Container tạm chỉ đọc data, không ghi gì vào `postgres_data`. Xóa container không ảnh hưởng data gốc.

---

### Bước 5: Import vào n8n mới

```bash
# Import qua n8n CLI (chạy trong container n8n)
docker exec n8n-services n8n import:workflow \
  --input=/home/node/.n8n/workflows/tradeflat_export/OpenAI_0._Main_HS_Code_V2_fsJY9c1sKbhykI7m.json

# Import qua n8n API
curl -X POST http://157.10.186.122:3002/api/v1/workflows \
  -H "Content-Type: application/json" \
  -H "X-N8N-API-KEY: <your-api-key>" \
  -d @OpenAI_0._Main_HS_Code_V2_fsJY9c1sKbhykI7m.json
```

Hoặc import qua **n8n UI**: Vào giao diện `https://157.10.186.122:3002` → nhấn `...` → `Import from File`.

---
---

## Phần 2: Nén folder có quyền Docker

### Vấn đề

Folder `postgres_data` được tạo bởi user postgres trong Docker container.
Trên máy host, user `vietpv` không có quyền đọc:

```
$ ls -la ~/FIS/tradeflat/n8n/database/
drwx------ 19 ollama vietpv 4096 ... postgres_data    ← Permission denied!
drwxrwxr-x  7 vietpv vietpv 4096 ... n8n_data_postgres ← OK
```

Chạy `tar` bình thường sẽ bị lỗi:
```
$ tar -czf database.tar.gz -C ~/FIS/tradeflat/n8n database/
tar: database/postgres_data: Cannot open: Permission denied
```

---

### Giải pháp 1: Dùng Docker Alpine (KHÔNG cần sudo) ← KHUYÊN DÙNG

```bash
docker run --rm \
  -v /home/vietpv/FIS/tradeflat/n8n/database:/source:ro \
  -v /home/vietpv/FIS/tradeflat/n8n:/output \
  alpine:latest \
  tar -czf /output/database.tar.gz -C /source .
```

**Giải thích:**

| Phần | Ý nghĩa |
|------|---------|
| `--rm` | Tự xóa container sau khi chạy xong |
| `-v .../database:/source:ro` | Mount folder cần nén vào `/source` (chỉ đọc) |
| `-v .../n8n:/output` | Mount folder lưu file nén vào `/output` |
| `alpine:latest` | Image siêu nhẹ (~5MB), có sẵn lệnh `tar` |
| `tar -czf /output/database.tar.gz` | Tạo file nén gzip |
| `-C /source .` | Vào `/source` rồi nén toàn bộ nội dung |

**Tại sao cách này được?**
Docker daemon chạy với quyền root → container đọc được tất cả file bất kể permission trên host.

**Kết quả:**
```
$ ls -lh ~/FIS/tradeflat/n8n/database.tar.gz
-rw-r--r-- 1 root root 692M Thg 6  4 16:35 database.tar.gz
```

---

### Giải pháp 2: Dùng sudo

```bash
sudo tar -czf /home/vietpv/FIS/tradeflat/n8n/database.tar.gz \
  -C /home/vietpv/FIS/tradeflat/n8n database/
```

---

### Giải nén file .tar.gz

```bash
# Giải nén bình thường
tar -xzf /home/vietpv/FIS/tradeflat/n8n/database.tar.gz

# Giải nén vào thư mục cụ thể
tar -xzf /home/vietpv/FIS/tradeflat/n8n/database.tar.gz \
  -C /home/vietpv/FIS/tradeflat/n8n/database_restored/

# Nếu bị permission denied → dùng Docker
docker run --rm \
  -v /home/vietpv/FIS/tradeflat/n8n/database.tar.gz:/input/database.tar.gz:ro \
  -v /home/vietpv/FIS/tradeflat/n8n/database_restored:/output \
  alpine:latest \
  tar -xzf /input/database.tar.gz -C /output
```

---
---

## Tóm tắt lệnh nhanh

```
=== EXPORT WORKFLOW ===

# 1. Khởi postgres tạm từ data cũ
docker run -d --name temp-postgres \
  -e POSTGRES_DB=n8n -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=123456aA@ \
  -v /home/vietpv/FIS/tradeflat/n8n/database/postgres_data:/var/lib/postgresql/data \
  -p 3099:5432 pgvector/pgvector:0.6.2-pg15 postgres -c max_connections=300

# 2. Export
sleep 3
python3 /home/vietpv/FIS/n8n/database/export_workflows.py

# 3. Dọn dẹp
docker stop temp-postgres && docker rm temp-postgres


=== NÉN FILE DOCKER ===

docker run --rm \
  -v /home/vietpv/FIS/tradeflat/n8n/database:/source:ro \
  -v /home/vietpv/FIS/tradeflat/n8n:/output \
  alpine:latest \
  tar -czf /output/database.tar.gz -C /source .
```
