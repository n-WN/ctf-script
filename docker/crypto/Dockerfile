# -----------------------------------------------------------------------------
# 阶段 1: 构建器 (Builder)
# -----------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-alpine AS builder

# 安装系统依赖 (gmpy2 的编译需要)
RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apk/repositories && \
    apk update && \
    apk add --no-cache gmp-dev mpfr-dev mpc1-dev build-base

WORKDIR /app

# 【修复】添加 --system 标志，并为 uv 创建一个虚拟环境来安装包
# 这样最终产物是干净的 .venv 目录，便于复制
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install pycryptodome gmpy2

# 复制应用程序的代码
COPY challenge.py .

# -----------------------------------------------------------------------------
# 阶段 2: 最终镜像 (Final Image)
# -----------------------------------------------------------------------------
FROM python:3.12-alpine

# 安装运行时系统依赖 (socat 和 gmpy2 需要的库)
RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apk/repositories && \
    apk update && \
    apk add --no-cache socat gmp mpfr mpc1

# 从构建器阶段复制已安装好依赖的虚拟环境
COPY --from=builder /app/.venv /app/.venv
# 复制应用程序代码
COPY --from=builder /app/challenge.py /app/challenge.py

# 添加一个非 root 用户以增强安全性
RUN adduser -D ctf
USER ctf

WORKDIR /app

# 将虚拟环境的 bin 目录添加到 PATH
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 1337

CMD ["socat", "-T60", "TCP-LISTEN:1337,fork,reuseaddr", "EXEC:python -u ./challenge.py"]