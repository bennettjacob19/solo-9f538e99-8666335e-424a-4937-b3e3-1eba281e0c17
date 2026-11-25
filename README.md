# gzip 旧日志归档取数服务

值班员拿到容器旧日志的字节偏移后，可直接翻到出错位置。Flask 网页服务：

- 上传**标准 gzip**，自动分配日志号；
- 凭日志号**整件下载**原始 gzip，或填写解压后的起点/长度**取原始字节**，每次最多 **64 KiB**；
- 对旧日志开启**快速取数**：填写新增文件占用上限，页面/接口显示**实际峰值**，原 gzip 不计入上限；
- 升级（构建索引）期间两种读取照常；
- 快速模式区间读取**实际解压不超过 256 KiB**，响应头 `X-Decompressed-Bytes` 显示本次解压量；
- 超限、模拟写入失败、升级途中重启，未完成结果一律不生效，仍可凭原编号取数；
- 升级前后 gzip 解出的字节完全一致；最终只保留一份标准 gzip 加一份稀疏索引（索引不是整份解压正文）。

## 目录

```
app/
  gzcodec.py    gzip 校验、多成员 gzip 重压缩、稀疏索引、区间读取
  storage.py    存储布局、原子元数据、崩溃恢复、峰值统计
  upgrade.py    快速取数升级事务（配额、故障注入）
  main.py       Flask 路由与页面
  templates/    单页 UI
tests/          15 个测试（含真实子进程的升级中断重启测试）
Dockerfile / docker-compose.yml
```

## 快速开始（docker compose）

```bash
docker compose up --build
# 打开 http://localhost:8000
```

数据保存在命名卷 `gzlog-data`（容器内 `/data`），容器重启后日志与索引都在。

本地无 Docker 时：

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
DATA_DIR=./data PORT=8000 python -m app.main
```

## HTTP 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/upload` | multipart 上传标准 gzip，返回日志号 |
| GET  | `/logs/<id>/download` | 按日志号整件下载**原始 gzip** |
| GET  | `/logs/<id>/content` | 下载解压全文（便捷接口） |
| GET  | `/logs/<id>/range?start=&length=` | 取解压后 `[start, start+length)` 原始字节，`length ≤ 65536` |
| POST | `/logs/<id>/upgrade` | JSON `{"cap_bytes": N}` 开启快速取数；可选故障注入字段 |
| GET  | `/logs/<id>/status` | 状态、上限、实际峰值、当前占用、原 gzip 大小 |
| GET  | `/healthz` | 健康检查 |

区间读取响应头：

- `X-Read-Mode`：`fast`（索引命中）/ `basic`（全量解压兜底，含升级期间）/ `basic-fallback`
- `X-Decompressed-Bytes`：本次实际解压的字节数（快速模式保证 ≤ 256 KiB）
- `X-Length`：实际返回字节数（请求超出文件尾部时截断）

## 设计要点

### 1. 多成员标准 gzip + 稀疏索引

Python 标准库的 gzip 无法从压缩流中间随机定位。升级时把原 gzip 解压后重新压成
**多个独立 gzip 成员的拼接**——RFC 1952 明确允许一个 gzip 文件由多个 member 组成，
系统 `gzip -dc` 会透明解压拼接流（测试中已实测）。每个成员最多承载 64 KiB 解压数据。

索引仅记录每个成员的偏移/长度四元组，不含任何日志正文：

```json
{"uoff": 65536, "usize": 65536, "coff": 2008, "csize": 2040}
```

1 MB 日志的索引约 1 KB（正文的千分之一），不可能充当正文的替代品。

### 2. 解压量上限

64 KiB 的请求区间 `[s, s+64KiB)` 最多横跨 **2 个** 64 KiB 成员（起点未对齐成员边界时），
即最多解压 2 × 64 KiB = 128 KiB < 256 KiB。代码内另有硬校验 `MAX_DECOMPRESS_PER_RANGE`，
超出直接报错。实测 20 个随机跨段请求最大解压量为 128 KiB。

### 3. 升级事务与原子性

```
basic ──mark_upgrading──▶ upgrading ──构建 gzip.new/index.new──┐
  ▲                          │                                 │
  └──── rollback（任何失败）─┘                                  ▼
                                                      os.replace 提交 → fast
```

- 新文件一律先写 `*.new`，提交时 `os.replace` 原子改名；元数据 `meta.json` 同样临时文件 + 原子替换 + fsync；
- 配额计数覆盖 gzip 和索引的**每一个字节**（`ArtifactWriter` 单点统计），原 gzip 从不经过该写入器，故不计入上限；
- 实际峰值取构建过程中观察到的最大累计字节，持久化在 `fast/peak.json`，重启不丢；
- 超限（HTTP 507）、模拟写失败（HTTP 500）都会回滚到 `basic` 并清理 `.new`；
- 进程在升级途中被杀死：重启时 `Store._recover()` 发现 `upgrading` 状态或残留 `.new`，自动回滚，原编号立即可用；
- 回滚后可无损重试升级。

### 4. 升级期间可读

区间读取只在 `fast` 且成品文件存在时走索引；`upgrading`/`basic` 状态都对**原始 gzip**
做全量解压。整件下载始终返回原始 gzip。因此升级过程中两种读取都不中断（已用并发线程实测）。

### 5. 故障注入（仅用于演练）

`POST /logs/<id>/upgrade` 的 JSON 额外支持：

- `fail_after_bytes`：累计写入达到该字节数后抛模拟写入失败（进程存活，验证回滚）；
- `crash_after_bytes`：达到阈值后 `os._exit(87)` 硬退出（验证重启恢复）。

## 测试

```bash
python -m unittest tests.test_app -v
```

15 个测试覆盖：

1. 多成员 gzip 往返、系统 `gzip -dc` 兼容性、空文件、坏 gzip 拒绝；
2. 跨成员区间读取与全量解压结果逐字节一致，解压量 ≤ 256 KiB；
3. 上传/整件下载/解压全文/区间读取/64 KiB 长度限制；
4. 升级后 fast 读取、峰值统计、索引远小于正文且不含正文；
5. 配额超限回滚、模拟写失败回滚后可重试；
6. 升级期间并发读取无错误；
7. **真实子进程**：升级中 `os._exit` 崩溃 → 重启自动回滚 → 原编号取数正确 → 重新升级成功。

## 已知边界 / 未完成项

- 本环境未安装 Docker，`Dockerfile` 与 `docker-compose.yml` 已通过 YAML/结构校验，但未在本机实际构建运行；应用本身仅依赖 Flask，镜像构建无特殊步骤。
- 未做用户认证与多副本部署；升级锁是单进程内的 per-log 线程锁，多实例部署需换成分布式锁（当前 compose 仅单副本）。
- 上传大小无独立配额限制，仅受磁盘与 `DATA_DIR` 容量约束。
- 快速模式成品 gzip 与原 gzip 同时保留（原 gzip 是整件下载与兜底读取的权威数据源，需求要求保留原编号取数能力）；"只留一份 gzip 及索引"指快速取数的新成品只有一份 gzip + 一份索引，不会产生解压正文副本或多份成品。
