#!/bin/bash
# 修复 CUDA 版本不匹配问题的脚本

set -e

# 检查是否需要完全重建虚拟环境
REBUILD_VENV=false
if [ "$1" == "--rebuild" ] || [ "$1" == "-r" ]; then
    REBUILD_VENV=true
fi

echo "=== 修复 CUDA 版本不匹配问题 ==="
echo ""

if [ "$REBUILD_VENV" = true ]; then
    echo "⚠️  将完全重建虚拟环境（推荐用于彻底解决问题）"
    echo ""
    read -p "确认删除 .venv 并重新创建？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除现有虚拟环境..."
        rm -rf .venv
        echo "重新创建虚拟环境..."
        uv venv
    else
        echo "取消操作"
        exit 0
    fi
fi

# 检查系统 CUDA 版本
echo "1. 检查系统 CUDA 版本..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "   系统 CUDA 版本: $CUDA_VERSION"
else
    echo "   警告: 未找到 nvcc，无法确定 CUDA 版本"
    CUDA_VERSION="unknown"
fi

# 确定 PyTorch CUDA 版本
if [[ "$CUDA_VERSION" == "12.2" ]]; then
    PYTORCH_CUDA="cu121"  # PyTorch 使用 cu121 表示 CUDA 12.x
    echo "   将使用 PyTorch CUDA 12.1/12.2 版本"
elif [[ "$CUDA_VERSION" == "12.1" ]]; then
    PYTORCH_CUDA="cu121"
    echo "   将使用 PyTorch CUDA 12.1 版本"
else
    echo "   警告: 未知的 CUDA 版本，尝试使用 cu121"
    PYTORCH_CUDA="cu121"
fi

echo ""
echo "2. 激活虚拟环境..."
if [ ! -d ".venv" ]; then
    echo "   虚拟环境不存在，正在创建..."
    uv venv
fi

source .venv/bin/activate

# 如果是重建的环境，需要先安装基础依赖
if [ "$REBUILD_VENV" = true ]; then
    echo "   安装基础依赖（不包括 flash-attn）..."
    uv sync --no-install-package flash-attn || true
fi

echo ""
echo "3. 卸载现有的 PyTorch 和 CUDA 相关包..."
# 先列出所有 nvidia 包并逐个卸载
echo "   正在卸载所有 nvidia-* 包..."
for pkg in $(uv pip list | grep -E "^nvidia-" | awk '{print $1}'); do
    echo "   卸载 $pkg..."
    uv pip uninstall "$pkg" 2>/dev/null || true
done

# 卸载 PyTorch 相关包
echo "   正在卸载 PyTorch 相关包..."
uv pip uninstall torch torchvision torchaudio 2>/dev/null || true

# 清理残留的 nvidia 库文件（如果存在）
if [ -d ".venv/lib/python3.12/site-packages/nvidia" ]; then
    echo "   清理残留的 nvidia 库文件..."
    rm -rf .venv/lib/python3.12/site-packages/nvidia
fi

echo ""
echo "4. 安装与系统 CUDA 版本匹配的 PyTorch..."
# 对于 CUDA 12.2，我们需要安装支持 CUDA 12.1/12.2 的 PyTorch 版本
# PyTorch 2.4.x 支持 CUDA 12.1/12.2，而 2.5.x 默认使用 CUDA 12.4
echo "   从 PyTorch 官方源安装兼容版本..."
# 尝试安装 PyTorch 2.4.1（最后一个支持 CUDA 12.1/12.2 的稳定版本）
uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA} || {
    echo "   PyTorch 2.4.1 安装失败，尝试安装最新兼容版本..."
    # 如果失败，尝试让 PyTorch 自动选择兼容版本
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}
}

echo ""
echo "5. 设置 CUDA 库路径..."
# 设置环境变量，优先使用系统 CUDA 库
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH}
export CUDA_HOME=/usr/local/cuda-12.2

# 将环境变量写入激活脚本，使其永久生效
cat >> .venv/bin/activate << 'EOF'

# CUDA 环境变量设置（由 fix_cuda_mismatch.sh 添加）
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH}
export CUDA_HOME=/usr/local/cuda-12.2
EOF

echo "   已设置 CUDA 环境变量"

echo ""
echo "6. 验证 PyTorch 安装..."
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" || {
    echo "   ⚠️  PyTorch 导入失败，但可能只是 cuDNN 问题"
    echo "   尝试安装 nvidia-cudnn-cu12..."
    uv pip install nvidia-cudnn-cu12==9.1.0.70 || {
        echo "   cuDNN 安装失败，但 PyTorch 可能仍可在 CPU 模式下工作"
        echo "   如果后续需要 GPU 支持，可能需要系统级安装 cuDNN"
    }
    # 再次尝试验证
    python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" || {
        echo "   ⚠️  PyTorch 仍然无法导入，但继续安装其他依赖..."
    }
}

echo ""
echo "7. 确保所有 nvidia 包版本正确..."
# 确保 nvidia-nvjitlink-cu12 是正确版本
CURRENT_NVJITLINK=$(uv pip list | grep nvidia-nvjitlink-cu12 | awk '{print $2}' || echo "")
if [[ "$CURRENT_NVJITLINK" == "12.4"* ]]; then
    echo "   降级 nvidia-nvjitlink-cu12 到 12.1 版本..."
    uv pip uninstall nvidia-nvjitlink-cu12
    rm -rf .venv/lib/python3.12/site-packages/nvidia/nvjitlink .venv/lib/python3.12/site-packages/nvidia_nvjitlink_cu12-12.4.*.dist-info 2>/dev/null || true
    uv pip install "nvidia-nvjitlink-cu12==12.1.105"
fi

# 确保 cusparse 是正确版本（这个包特别容易有符号链接问题）
CURRENT_CUSPARSE=$(uv pip list | grep nvidia-cusparse-cu12 | awk '{print $2}' || echo "")
if [[ "$CURRENT_CUSPARSE" == "12.3"* ]] || [[ "$CURRENT_CUSPARSE" == "12.4"* ]]; then
    echo "   降级 nvidia-cusparse-cu12 到 12.1 版本..."
    uv pip uninstall nvidia-cusparse-cu12
    uv pip install "nvidia-cusparse-cu12==12.1.0.106"
fi

# 确保 PyTorch 是正确版本
CURRENT_TORCH=$(uv pip list | grep "^torch " | awk '{print $2}' || echo "")
if [[ "$CURRENT_TORCH" == "2.5"* ]]; then
    echo "   降级 PyTorch 到 2.4.1 (CUDA 12.1 兼容版本)..."
    uv pip uninstall torch torchvision torchaudio
    uv pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
fi

echo ""
echo "8. 尝试安装 flash-attn..."
# 在安装 flash-attn 时设置环境变量
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH}
export CUDA_HOME=/usr/local/cuda-12.2

# 先尝试验证 PyTorch 可以正常导入
echo "   验证 PyTorch 安装..."
if ! python -c "import torch; print('PyTorch OK')" 2>/dev/null; then
    echo "   ⚠️  PyTorch 导入失败，但继续尝试安装 flash-attn..."
fi

if LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH} CUDA_HOME=/usr/local/cuda-12.2 uv pip install flash-attn==2.7.4.post1 --no-build-isolation 2>&1 | tee /tmp/flash_attn_install.log; then
    echo ""
    echo "✅ 成功！所有依赖已安装"
else
    echo ""
    echo "⚠️  flash-attn 安装可能失败，但其他依赖已安装"
    echo "   如果 flash-attn 是必需的，可能需要："
    echo "   - 从源码编译 flash-attn（需要 CUDA 开发工具）"
    echo "   - 或使用预编译的 wheel 文件"
    echo ""
    echo "   查看详细日志: cat /tmp/flash_attn_install.log"
fi

echo ""
echo "=== 修复完成 ==="
echo ""
echo "如果问题仍然存在，请尝试："
echo "  bash fix_cuda_mismatch.sh --rebuild"
echo "  这将完全重建虚拟环境（更彻底但需要重新安装所有包）"
