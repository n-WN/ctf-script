#!/bin/bash
#
# Author: n-WN
# Main inspiration from: rec

# ==============================================================================
#  setup_sage_venv.sh
#
#  本脚本根据 "A Fresh Take" on Sagemath environment (rec 2025-07-23)
#  的思路，自动化创建一个集成了 SageMath 库的 Python 虚拟环境，并
#  为 Visual Studio Code 生成相应的配置文件。
#
#  用法:
#    1. 将此脚本保存为 setup_sage_venv.sh
#    2. 赋予执行权限: chmod +x setup_sage_venv.sh
#    3. 在你的项目根目录下运行: ./setup_sage_venv.sh [虚拟环境名称]
#       如果未提供名称，将默认为 "sage_venv"。
#
# ==============================================================================

# --- 配置 ---
# 设置虚拟环境的默认名称
DEFAULT_VENV_NAME="sage_venv"

# --- 用于美化输出的颜色代码 ---
C_RESET='\033[0m'
C_RED='\033[0;31m'
C_GREEN='\033[0;32m'
C_YELLOW='\033[0;33m'
C_BLUE='\033[0;34m'

# --- 日志函数 ---
log_info() {
    echo -e "${C_BLUE}[信息]${C_RESET} $1"
}
log_success() {
    echo -e "${C_GREEN}[成功]${C_RESET} $1"
}
log_warn() {
    echo -e "${C_YELLOW}[警告]${C_RESET} $1"
}
log_error() {
    # 错误信息输出到 stderr
    echo -e "${C_RED}[错误]${C_RESET} $1" >&2
}

# --- 主脚本逻辑 ---

# 从命令行第一个参数获取虚拟环境名称，如果未提供则使用默认值
VENV_NAME="${1:-$DEFAULT_VENV_NAME}"

# 步骤 1: 检查 'sage' 命令是否存在
log_info "正在检查 SageMath 环境..."
if ! command -v sage &> /dev/null; then
    log_error "未在您的 PATH 环境变量中找到 'sage' 命令。"
    log_info "请确保您已正确安装 SageMath 并将其添加到了系统 PATH 中。"
    exit 1
fi
log_success "SageMath 环境已找到。"

# 步骤 2: 定位 SageMath 使用的 Python 解释器
log_info "正在定位 SageMath 的 Python 解释器..."
# 此命令让 sage 打印其内部 Python 解释器的绝对路径
SAGE_PYTHON_PATH=$(sage -c 'import sys; print(sys.executable)')

# 检查命令是否成功执行并返回了有效的路径
if [ -z "$SAGE_PYTHON_PATH" ] || [ ! -f "$SAGE_PYTHON_PATH" ]; then
    log_error "无法确定 SageMath 的 Python 解释器路径。"
    log_info "尝试执行的命令为: sage -c 'import sys; print(sys.executable)'"
    exit 1
fi
log_success "SageMath Python 解释器位于: ${C_YELLOW}$SAGE_PYTHON_PATH${C_RESET}"

# 步骤 3: 处理已存在的虚拟环境目录
if [ -d "$VENV_NAME" ]; then
    log_warn "目录 '$VENV_NAME' 已存在。"
    read -p "是否要删除并重新创建? (y/N) " -n 1 -r
    echo # 换行
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "操作已由用户取消。"
        exit 0
    fi
    log_info "正在删除已存在的目录: $VENV_NAME"
    rm -rf "$VENV_NAME"
fi

# 步骤 4: 创建继承 SageMath 包的虚拟环境
log_info "正在创建 Python 虚拟环境 '$VENV_NAME'..."
# 使用 SageMath 的 Python 来创建 venv，并使用 --system-site-packages
# 标志来继承 SageMath 的所有库。
"$SAGE_PYTHON_PATH" -m venv "$VENV_NAME" --system-site-packages

# 检查虚拟环境是否创建成功
if [ $? -ne 0 ]; then
    log_error "创建虚拟环境失败。"
    exit 1
fi
log_success "虚拟环境 '$VENV_NAME' 创建成功。"

# 步骤 5: 创建 VS Code 配置文件
log_info "正在生成 VS Code 配置文件..."
# 如果 .vscode 目录不存在，则创建它
mkdir -p .vscode

# 定义新虚拟环境中的 Python 解释器路径
VENV_PYTHON_PATH="${VENV_NAME}/bin/python"

# 使用 heredoc 语法创建 .vscode/settings.json 文件
# 这可以避免处理 JSON 字符串转义的麻烦
cat > .vscode/settings.json << EOL
{
    // 将此工作区的默认 Python 解释器设置为新创建的 SageMath 虚拟环境。
    "python.defaultInterpreterPath": "${VENV_PYTHON_PATH}",

    // 在 VS Code 的集成终端中自动激活此环境。
    "python.terminal.activateEnvironment": true,

    // 从文件浏览器中隐藏配置文件和虚拟环境文件夹，保持界面整洁。
    "files.exclude": {
        "**/.vscode": true,
        "**/${VENV_NAME}": true
    }
}
EOL

if [ $? -ne 0 ]; then
    log_error "创建 '.vscode/settings.json' 失败。"
    exit 1
fi
log_success "VS Code 配置文件 '.vscode/settings.json' 已生成。"

# --- 最终说明 ---
echo
log_success "全部设置完成!"
echo
echo -e "一个名为 '${C_YELLOW}${VENV_NAME}${C_RESET}' 的虚拟环境已经创建完毕。"
echo -e "一个 '${C_YELLOW}.vscode/settings.json${C_RESET}' 配置文件也已为本项目生成。"
echo
echo -e "要在当前 Shell 中手动激活此环境，请运行:"
echo -e "  ${C_GREEN}source ${VENV_NAME}/bin/activate${C_RESET}"
echo
echo -e "当您用 VS Code 打开此文件夹时，Python 扩展应该会自动检测并使用此环境。"
echo -e "现在您可以使用 '${C_YELLOW}pip install <package>${C_RESET}' 安装额外的包，并用 '${C_YELLOW}python your_script.py${C_RESET}' 运行您的代码。"
