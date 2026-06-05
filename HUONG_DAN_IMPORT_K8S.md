# Hướng dẫn Import Database N8N vào K8s PostgreSQL

## Tổng quan

Import file `n8n_workflows_dump.sql` (dump từ tradeflat local) vào PostgreSQL trên K8s (OpenShift)
để n8n UI hiển thị đầy đủ workflows.

**File dump:** `~/FIS/n8n/database/n8n_workflows_dump.sql` (9.5MB, 33 workflows)
**Nguồn dump:** `~/FIS/tradeflat/n8n/database/postgres_data/` (bản n8n cũ)

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

### Bước 0: Login vào OpenShift cluster

```bash
# Login vào cluster
oc login https://<cluster-url> -u <username> -p <password>

# Hoặc login bằng token
oc login --token=<token> --server=https://<cluster-url>

# Chuyển sang đúng project (namespace)
oc project fis-mbf-bidimex-uat
```

---

### Bước 1: Tìm pod PostgreSQL

```bash
# Liệt kê tất cả pods trong project
oc get pods | grep postgres

# Kết quả ví dụ:
# n8n-services-postgres-xxxxx-yyyyy   1/1   Running   0   10d

# Hoặc tìm chính xác hơn bằng label
oc get pods -l app=n8n-services-postgres
```

Ghi lại tên pod, ví dụ: `n8n-services-postgres-xxxxx-yyyyy`

---

### Bước 2: Copy file SQL vào pod

```bash
# Copy file SQL từ máy local vào pod postgres
oc cp ~/FIS/n8n/database/n8n_workflows_dump.sql <tên-pod-postgres>:/tmp/n8n_workflows_dump.sql

# Ví dụ thực tế:
oc cp ~/FIS/n8n/database/n8n_workflows_dump.sql n8n-services-postgres-xxxxx-yyyyy:/tmp/n8n_workflows_dump.sql

# Kiểm tra file đã copy thành công
oc exec <tên-pod-postgres> -- ls -lh /tmp/n8n_workflows_dump.sql
```

---

### Bước 3: Import database

```bash
# Exec vào pod và chạy import
oc exec <tên-pod-postgres> -- psql -U postgres -d n8n -f /tmp/n8n_workflows_dump.sql

# Hoặc exec vào pod rồi chạy từng lệnh (nếu muốn kiểm soát hơn)
oc rsh <tên-pod-postgres>

# Bên trong pod:
psql -U postgres -d n8n < /tmp/n8n_workflows_dump.sql

# Kiểm tra workflows đã có chưa
psql -U postgres -d n8n -c "SELECT id, name, active FROM workflow_entity ORDER BY name;"

# Đếm tổng workflows
psql -U postgres -d n8n -c "SELECT count(*) as total_workflows FROM workflow_entity;"

# Dọn file tạm
rm /tmp/n8n_workflows_dump.sql

# Thoát pod
exit
```

---

### Bước 4: Restart n8n pod để load lại data

```bash
# Cách 1: Restart deployment (rolling restart, không downtime)
oc rollout restart deployment/n8n-services

# Cách 2: Scale xuống 0 rồi lên lại (nếu cách 1 không hoạt động)
oc scale deployment/n8n-services --replicas=0
oc scale deployment/n8n-services --replicas=1

# Cách 3: Xóa pod để K8s tự tạo lại
oc delete pod <tên-pod-n8n>

# Kiểm tra pod đã restart xong
oc get pods | grep n8n

# Xem logs để đảm bảo n8n khởi động thành công
oc logs -f deployment/n8n-services --tail=50
```

---

### Bước 5: Kiểm tra trên n8n UI

Truy cập n8n UI tại `https://157.10.186.122:3002/` → kiểm tra:
- Tất cả 33 workflows có hiển thị
- Các workflows active (✅) vẫn đúng trạng thái
- Mở 1 workflow để kiểm tra nodes/connections

---

## Cách gộp 1 lệnh (copy-paste)

```bash
# Chuyển sang đúng project
oc project fis-mbf-bidimex-uat

# Tìm pod postgres tự động
POD=$(oc get pods -l app=n8n-services-postgres -o jsonpath='{.items[0].metadata.name}')
echo "Pod postgres: $POD"

# Copy + Import + Kiểm tra + Restart
oc cp ~/FIS/n8n/database/n8n_workflows_dump.sql $POD:/tmp/dump.sql && \
oc exec $POD -- psql -U postgres -d n8n -f /tmp/dump.sql && \
oc exec $POD -- psql -U postgres -d n8n -c "SELECT count(*) as total_workflows FROM workflow_entity;" && \
oc exec $POD -- rm /tmp/dump.sql && \
oc rollout restart deployment/n8n-services

echo "Import xong! Đợi n8n restart rồi kiểm tra UI."
```

---

## So sánh lệnh kubectl vs oc

| Thao tác              | kubectl                                    | oc (OpenShift)                          |
|-----------------------|--------------------------------------------|-----------------------------------------|
| Login                 | Dùng kubeconfig                            | `oc login`                              |
| Chọn namespace        | `kubectl -n <namespace>`                   | `oc project <namespace>`                |
| Liệt kê pods         | `kubectl get pods`                         | `oc get pods`                           |
| Copy file vào pod     | `kubectl cp file pod:/path`                | `oc cp file pod:/path`                  |
| Exec vào pod          | `kubectl exec -it pod -- bash`             | `oc rsh pod` hoặc `oc exec pod --`      |
| Xem logs              | `kubectl logs pod`                         | `oc logs pod`                           |
| Restart deployment    | `kubectl rollout restart deployment/name`  | `oc rollout restart deployment/name`    |
| Scale                 | `kubectl scale deployment/name --replicas` | `oc scale deployment/name --replicas`   |

> Hầu hết lệnh `kubectl` và `oc` tương tự nhau.
> `oc` có thêm một số lệnh riêng như `oc rsh` (remote shell), `oc new-app`, `oc project`.

---

## Xử lý lỗi thường gặp

### Lỗi 1: "relation already exists"

File dump có `--clean --if-exists` nên sẽ DROP bảng cũ trước khi tạo mới.
Nếu vẫn lỗi, bỏ qua lỗi:

```bash
oc exec <pod> -- psql -U postgres -d n8n -v ON_ERROR_STOP=0 -f /tmp/n8n_workflows_dump.sql
```

### Lỗi 2: "permission denied for schema public"

User `n8n` không có quyền tạo bảng. Dùng user `postgres` (superuser):

```bash
oc exec <pod> -- psql -U postgres -d n8n -f /tmp/n8n_workflows_dump.sql
```

### Lỗi 3: "oc cp" bị lỗi tar

```bash
# Thử cách khác: pipe file qua stdin
cat ~/FIS/n8n/database/n8n_workflows_dump.sql | oc exec -i <pod> -- tee /tmp/dump.sql > /dev/null

# Rồi import bình thường
oc exec <pod> -- psql -U postgres -d n8n -f /tmp/dump.sql
```

### Lỗi 4: Credentials (API keys) bị lỗi sau import

N8N mã hóa credentials bằng `N8N_ENCRYPTION_KEY`.
Kiểm tra:
- Local:  `N8N_ENCRYPTION_KEY=N8N_ENCRYPTION_KEY_MASTER` (file .env)
- K8s:    `N8N_ENCRYPTION_KEY=N8N_ENCRYPTION_KEY_MASTER` (file secret.yaml)

Nếu khớp → OK. Nếu không → cập nhật lại credentials thủ công trên n8n UI.

### Lỗi 5: Workflow chạy nhưng gọi sai URL nội bộ

Workflows có thể hardcode URL local (ví dụ `http://n8n-services:5678`).
Trên K8s tên service có thể khác → kiểm tra và sửa trong workflow nodes.

---

## Tóm tắt

```
1. oc login + oc project fis-mbf-bidimex-uat
2. oc cp dump.sql vào pod postgres
3. oc exec → psql import
4. oc rollout restart n8n
5. Kiểm tra trên UI
```

Sau khi import, n8n đọc trực tiếp từ bảng `workflow_entity` trong PostgreSQL
và hiển thị tất cả workflows trên giao diện UI.
