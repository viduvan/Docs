# Hướng dẫn Export Workflow từ DB & Nén File Docker

> Tất cả ví dụ dùng đường dẫn và credentials thực tế từ project `~/FIS/tradeflat/n8n`

---

## Phần 1: Export N8N Workflows từ PostgreSQL

### Bối cảnh

- N8N lưu workflows trong bảng `workflow_entity` của PostgreSQL
- Dữ liệu PostgreSQL raw nằm tại: `/home/vietpv/FIS/tradeflat/n8n/database/postgres_data/`
- Image PostgreSQL: `pgvector/pgvector:0.6.2-pg15`
- Credentials: `postgres / 123456aA@ / database: n8n`
- Kết quả export lưu tại: `/home/vietpv/FIS/tradeflat/n8n/database/tradeflat_export/`

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
- Port `3099` → chọn port chưa bị chiếm (`3004` đã dùng bởi pptx-slides-postgres)
- Volume mount trỏ đúng folder `postgres_data` chứa raw data
- Image phải đúng với image đã tạo database (`pgvector/pgvector:0.6.2-pg15`)
- Credentials phải khớp với file `.env` gốc (`/home/vietpv/FIS/tradeflat/n8n/.env`)

---

### Bước 2: Kiểm tra database

```bash
# Đợi postgres khởi động
sleep 3

# Liệt kê bảng
docker exec temp-postgres psql -U postgres -d n8n -c "\dt"
```

Kết quả:
```
 Schema |            Name            | Type  |  Owner
--------+----------------------------+-------+----------
 public | credentials_entity         | table | postgres
 public | execution_data             | table | postgres
 public | execution_entity           | table | postgres
 public | workflow_entity            | table | postgres
 public | workflow_history           | table | postgres
 ...                                  (tổng 41 bảng)
```

```bash
# Xem danh sách workflows
docker exec temp-postgres psql -U postgres -d n8n \
  -c "SELECT id, name, active FROM workflow_entity ORDER BY id;"
```

Kết quả:
```
        id        |                 name                 | active
------------------+--------------------------------------+--------
 2sCBouStFwuoCknd | Update Data Super base               | f
 3bQpwmAJJhtdbaVf | LLM Medical Mapping                  | t
 3xFYvC25yLSfzpg4 | OpenAI 3. Select Item ID             | t
 3Z8r9zIHTmzZpEul | LLM Generator                        | t
 4R4MUeXpvcfsfM3T | OpenAI 1. Workflow Select Chapter     | t
 fsJY9c1sKbhykI7m | OpenAI 0. Main HS Code V2            | t
 gSv6TXEKSXYsyYFP | OpenAI 2. Workflow Select Group      | t
 hegiWhArOy6a8imy | VBPL Chatbot                         | t
 ...              | (tổng 33 workflows)                  |
```

---

### Bước 3: Export workflows

#### Cách 1: Dùng Python script (khuyên dùng)

```bash
# Cài thư viện (chỉ cần chạy 1 lần)
pip install psycopg2-binary

# Chạy script export
python3 /home/vietpv/FIS/n8n/database/export_workflows.py
```

Kết quả:
```
  ⬜ inactive | Update Data Super base -> Update_Data_Super_base_2sCBouStFwuoCknd.json
  ✅ active | LLM Medical Mapping -> LLM_Medical_Mapping_3bQpwmAJJhtdbaVf.json
  ✅ active | OpenAI 3. Select Item ID -> OpenAI_3._Select_Item_ID_3xFYvC25yLSfzpg4.json
  ✅ active | OpenAI 0. Main HS Code V2 -> OpenAI_0._Main_HS_Code_V2_fsJY9c1sKbhykI7m.json
  ✅ active | OpenAI 1. Workflow Select Chapter -> OpenAI_1._Workflow_Select_Chapter_4R4MUeXpvcfsfM3T.json
  ✅ active | OpenAI 2. Workflow Select Group -> OpenAI_2._Workflow_Select_Group_gSv6TXEKSXYsyYFP.json
  ✅ active | VBPL Chatbot -> VBPL_Chatbot_hegiWhArOy6a8imy.json
  ⬜ inactive | My workflow -> My_workflow_KTgWqq9O3zDeoX8q.json
  ...
============================================================
Exported 33 workflows to: /home/vietpv/FIS/tradeflat/n8n/database/tradeflat_export
Combined file: /home/vietpv/FIS/tradeflat/n8n/database/tradeflat_export/_all_workflows.json
```

File kết quả:
```
/home/vietpv/FIS/tradeflat/n8n/database/tradeflat_export/
├── _all_workflows.json                                        (tổng hợp 33 wf)
├── OpenAI_0._Main_HS_Code_V2_fsJY9c1sKbhykI7m.json           (108KB)
├── OpenAI_1._Workflow_Select_Chapter_4R4MUeXpvcfsfM3T.json    (12KB)
├── OpenAI_2._Workflow_Select_Group_gSv6TXEKSXYsyYFP.json     (16KB)
├── OpenAI_3._Select_Item_ID_3xFYvC25yLSfzpg4.json            (13KB)
├── VBPL_Chatbot_hegiWhArOy6a8imy.json                        (31KB)
├── Main_V2_EQZPlzXrIBT7rLUu.json                             (10KB)
├── ... (26 file khác)
└── (tổng 34 files)
```

#### Cách 2: Dùng pg_dump export toàn bộ database (để import vào K8s)

```bash
# Dump toàn bộ database n8n ra file SQL
docker exec temp-postgres pg_dump -U postgres -d n8n --clean --if-exists > /tmp/n8n_dump.sql

# Copy về thư mục project
cp /tmp/n8n_dump.sql /home/vietpv/FIS/n8n/database/n8n_workflows_dump.sql
```

Kết quả:
```
$ ls -lh /home/vietpv/FIS/n8n/database/n8n_workflows_dump.sql
-rw-rw-r-- 1 vietpv vietpv 9,5M ... n8n_workflows_dump.sql
```

#### Cách 3: Dùng SQL thuần (export nhanh, không cần Python)

```bash
# Export tất cả workflows ra 1 file JSON
docker exec temp-postgres psql -U postgres -d n8n -t -A -c \
  "SELECT json_agg(row_to_json(w)) FROM workflow_entity w;" \
  > /home/vietpv/FIS/tradeflat/n8n/database/all_workflows_sql.json

# Export 1 workflow cụ thể
docker exec temp-postgres psql -U postgres -d n8n -t -A -c \
  "SELECT row_to_json(w) FROM workflow_entity w WHERE name = 'OpenAI 0. Main HS Code V2';" \
  > /home/vietpv/FIS/tradeflat/n8n/database/workflow_openai_main.json
```

---

### Bước 4: Dọn dẹp container tạm

```bash
docker stop temp-postgres && docker rm temp-postgres
```

Kết quả:
```
temp-postgres
temp-postgres
```

> Container tạm chỉ đọc data, không ghi gì vào postgres_data. Xóa container không ảnh hưởng data gốc.

---

### Bước 5: Import vào n8n mới

#### Trên K8s (OpenShift) — xem file HUONG_DAN_IMPORT_K8S.md

```bash
# Tìm pod postgres
oc get pods | grep postgres

# Copy + Import
oc cp ~/FIS/n8n/database/n8n_workflows_dump.sql <pod>:/tmp/dump.sql
oc exec <pod> -- psql -U postgres -d n8n -f /tmp/dump.sql
oc rollout restart deployment/n8n-services
```

#### Trên Docker local

```bash
# Import qua n8n CLI
docker exec n8n-services n8n import:workflow \
  --input=/home/node/.n8n/workflows/tradeflat_export/OpenAI_0._Main_HS_Code_V2_fsJY9c1sKbhykI7m.json

# Import qua n8n API
curl -X POST http://157.10.186.122:3002/api/v1/workflows \
  -H "Content-Type: application/json" \
  -H "X-N8N-API-KEY: <your-api-key>" \
  -d @OpenAI_0._Main_HS_Code_V2_fsJY9c1sKbhykI7m.json
```

Hoặc import qua **n8n UI**: Vào `https://157.10.186.122:3002/` → nhấn `...` → `Import from File`

---
---

## Phần 2: Nén folder có quyền Docker

### Vấn đề

Folder `postgres_data` được tạo bởi user postgres trong Docker container.
User `vietpv` không có quyền đọc:

```bash
$ ls -la /home/vietpv/FIS/tradeflat/n8n/database/
drwx------ 19 ollama vietpv 4096 ... postgres_data    ← Permission denied!
drwxrwxr-x  7 vietpv vietpv 4096 ... n8n_data_postgres ← OK
```

Chạy `tar` bình thường sẽ lỗi:
```bash
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
| `--rm` | Tự xóa container sau khi nén xong |
| `-v .../database:/source:ro` | Mount folder cần nén vào `/source` (chỉ đọc) |
| `-v .../n8n:/output` | Mount folder lưu file nén vào `/output` |
| `alpine:latest` | Image siêu nhẹ (~5MB), có sẵn lệnh `tar` |
| `tar -czf /output/database.tar.gz` | Tạo file nén gzip tại `/output` |
| `-C /source .` | Vào `/source` rồi nén toàn bộ nội dung |

**Tại sao cách này được?**
Docker daemon chạy với quyền root → container đọc được tất cả file bất kể permission trên host.

Kết quả:
```bash
$ ls -lh /home/vietpv/FIS/tradeflat/n8n/database.tar.gz
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
mkdir -p /home/vietpv/FIS/tradeflat/n8n/database_restored
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

### Ví dụ khác: Nén folder bất kỳ bị permission denied

```bash
# Template chung
docker run --rm \
  -v <folder-cần-nén>:/source:ro \
  -v <folder-lưu-kết-quả>:/output \
  alpine:latest \
  tar -czf /output/<tên-file>.tar.gz -C /source .

# Ví dụ: nén folder postgres_data của ~/FIS/n8n/ (bản mới)
docker run --rm \
  -v /home/vietpv/FIS/n8n/database/postgres_data:/source:ro \
  -v /home/vietpv/FIS/n8n/database:/output \
  alpine:latest \
  tar -czf /output/n8n_postgres_backup.tar.gz -C /source .

# Ví dụ: nén folder n8n_data_postgres
docker run --rm \
  -v /home/vietpv/FIS/n8n/database/n8n_data_postgres:/source:ro \
  -v /home/vietpv/FIS/n8n/database:/output \
  alpine:latest \
  tar -czf /output/n8n_data_backup.tar.gz -C /source .
```

---
---

## Tóm tắt lệnh nhanh (copy-paste)

### Export workflow

```bash
# 1. Khởi postgres tạm từ data cũ
docker run -d --name temp-postgres \
  -e POSTGRES_DB=n8n -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=123456aA@ \
  -v /home/vietpv/FIS/tradeflat/n8n/database/postgres_data:/var/lib/postgresql/data \
  -p 3099:5432 pgvector/pgvector:0.6.2-pg15 postgres -c max_connections=300

# 2. Đợi khởi động + Export JSON
sleep 3 && python3 /home/vietpv/FIS/n8n/database/export_workflows.py

# 3. Export SQL dump (để import vào K8s)
docker exec temp-postgres pg_dump -U postgres -d n8n --clean --if-exists > /tmp/n8n_dump.sql
cp /tmp/n8n_dump.sql /home/vietpv/FIS/n8n/database/n8n_workflows_dump.sql

# 4. Dọn dẹp
docker stop temp-postgres && docker rm temp-postgres
```

### Nén file Docker

```bash
docker run --rm \
  -v /home/vietpv/FIS/tradeflat/n8n/database:/source:ro \
  -v /home/vietpv/FIS/tradeflat/n8n:/output \
  alpine:latest \
  tar -czf /output/database.tar.gz -C /source .
```
