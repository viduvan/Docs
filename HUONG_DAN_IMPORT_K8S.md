# Hướng dẫn Import Database N8N vào K8s PostgreSQL

## Tổng quan

Import file `n8n_workflows_dump.sql` (dump từ tradeflat local) vào PostgreSQL trên K8s
để n8n UI hiển thị đầy đủ workflows.

**File dump:** `~/FIS/n8n/database/n8n_workflows_dump.sql` (9.5MB, 33 workflows)

---

## Lưu ý quan trọng: Credentials khác nhau

| Môi trường | DB User    | DB Password   | DB Name | Encryption Key             |
|-----------|------------|---------------|---------|----------------------------|
| Local     | postgres   | 123456aA@     | n8n     | N8N_ENCRYPTION_KEY_MASTER  |
| K8s       | n8n        | yAuG8K3S88DE  | n8n     | N8N_ENCRYPTION_KEY_MASTER  |

> **QUAN TRỌNG:** N8N_ENCRYPTION_KEY phải GIỐNG NHAU giữa local và K8s.
> Nếu khác nhau, n8n sẽ KHÔNG giải mã được credentials (API keys, tokens...)
> trong workflows và sẽ báo lỗi khi chạy workflow.
> 
> Hiện tại cả 2 đều dùng `N8N_ENCRYPTION_KEY_MASTER` → OK.

---

## Các bước thực hiện

### Bước 1: Copy file SQL vào pod PostgreSQL trên K8s

```bash
# Tìm tên pod postgres trên K8s
kubectl get pods -n fis-mbf-bidimex-uat | grep postgres

# Kết quả ví dụ:
# n8n-services-postgres-xxxxx-yyyyy   1/1   Running   0   10d

# Copy file SQL vào pod
kubectl cp ~/FIS/n8n/database/n8n_workflows_dump.sql \
  fis-mbf-bidimex-uat/<tên-pod-postgres>:/tmp/n8n_workflows_dump.sql
```

### Bước 2: Exec vào pod PostgreSQL và import

```bash
# Exec vào pod postgres
kubectl exec -it <tên-pod-postgres> -n fis-mbf-bidimex-uat -- bash

# Bên trong pod, chạy import:
psql -U postgres -d n8n < /tmp/n8n_workflows_dump.sql

# Kiểm tra workflows đã có chưa
psql -U postgres -d n8n -c "SELECT id, name, active FROM workflow_entity ORDER BY name;"

# Dọn file tạm
rm /tmp/n8n_workflows_dump.sql

# Thoát pod
exit
```

### Bước 3: Restart n8n pod để load lại data

```bash
# Restart n8n deployment (n8n sẽ đọc lại DB khi khởi động)
kubectl rollout restart deployment/n8n-services -n fis-mbf-bidimex-uat

# Hoặc nếu dùng K8s mới (~/FIS/n8n/k8s/):
kubectl rollout restart deployment/n8n-main -n default

# Kiểm tra pod đã restart xong
kubectl get pods -n fis-mbf-bidimex-uat | grep n8n
```

### Bước 4: Kiểm tra trên n8n UI

Truy cập n8n UI tại `https://157.10.186.122:3002/` → kiểm tra:
- Tất cả 33 workflows có hiển thị
- Các workflows active (✅) vẫn đúng trạng thái
- Mở 1 workflow để kiểm tra nodes/connections

---

## Cách gộp 1 lệnh (nếu đã quen)

```bash
# 1 lệnh duy nhất: copy + import + kiểm tra
NAMESPACE=fis-mbf-bidimex-uat
POD=$(kubectl get pods -n $NAMESPACE -l app=n8n-services-postgres -o jsonpath='{.items[0].metadata.name}')

kubectl cp ~/FIS/n8n/database/n8n_workflows_dump.sql $NAMESPACE/$POD:/tmp/dump.sql && \
kubectl exec -n $NAMESPACE $POD -- psql -U postgres -d n8n -f /tmp/dump.sql && \
kubectl exec -n $NAMESPACE $POD -- psql -U postgres -d n8n -c "SELECT count(*) as total_workflows FROM workflow_entity;" && \
kubectl rollout restart deployment/n8n-services -n $NAMESPACE
```

---

## Xử lý lỗi thường gặp

### Lỗi 1: "relation already exists"

File dump có `--clean --if-exists` nên sẽ DROP bảng cũ trước khi tạo mới.
Nếu vẫn lỗi, thêm flag:

```bash
psql -U postgres -d n8n -v ON_ERROR_STOP=0 < /tmp/n8n_workflows_dump.sql
```

### Lỗi 2: "permission denied for schema public"

User `n8n` không có quyền tạo bảng. Dùng user `postgres` (superuser):

```bash
psql -U postgres -d n8n < /tmp/n8n_workflows_dump.sql
```

### Lỗi 3: Credentials (API keys) bị lỗi sau import

N8N mã hóa credentials bằng `N8N_ENCRYPTION_KEY`.
Nếu key khác nhau giữa local và K8s → credentials bị hỏng.

Kiểm tra:
- Local: `N8N_ENCRYPTION_KEY=N8N_ENCRYPTION_KEY_MASTER` (file ~/.env)
- K8s: `N8N_ENCRYPTION_KEY=N8N_ENCRYPTION_KEY_MASTER` (file secret.yaml)

Nếu khớp → OK. Nếu không khớp → cần cập nhật lại credentials thủ công trên n8n UI.

### Lỗi 4: Workflow chạy nhưng gọi sai URL nội bộ

Workflows có thể hardcode URL local (ví dụ `http://n8n-services:5678`).
Trên K8s tên service có thể khác → cần sửa trong workflow nodes.

---

## Tóm tắt

```
1. kubectl cp dump.sql vào pod postgres
2. kubectl exec → psql import
3. kubectl rollout restart n8n
4. Kiểm tra trên UI
```

Sau khi import, n8n sẽ đọc trực tiếp từ bảng `workflow_entity` trong PostgreSQL
và hiển thị tất cả workflows trên giao diện UI.
