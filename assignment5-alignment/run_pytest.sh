#!/bin/bash
# 运行 pytest 的包装脚本，确保环境变量正确设置

export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH}
export CUDA_HOME=/usr/local/cuda-12.2

# 激活虚拟环境（如果存在）并修复 PyTorch 版本
if [ -d ".venv" ]; then
    source .venv/bin/activate
    
    # 检查并修复 PyTorch 版本
    CURRENT_TORCH=$(uv pip list | grep "^torch " | awk '{print $2}' || echo "")
    if [[ "$CURRENT_TORCH" == "2.5"* ]]; then
        echo "修复 PyTorch 版本..."
        uv pip uninstall torch torchvision torchaudio nvidia-nvjitlink-cu12 nvidia-cusparse-cu12
        uv pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
        uv pip install "nvidia-nvjitlink-cu12==12.1.105" "nvidia-cusparse-cu12==12.1.0.106"
    fi
fi

# 运行 pytest
pytest "$@"
