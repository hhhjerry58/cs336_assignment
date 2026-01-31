# CS336 Spring 2025 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)

We include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## 作业完成流程指南 (Complete Assignment Workflow)

### 第一步：环境设置 (Step 1: Environment Setup)

1. **安装依赖包** (Install dependencies)

   使用 `uv` 管理依赖。由于 `flash-attn` 的特殊性，需要分两步安装：

   ```bash
   # 先安装除 flash-attn 外的所有包
   uv sync --no-install-package flash-attn
   
   # 然后安装所有包（包括 flash-attn）
   uv sync
   ```

2. **验证安装** (Verify installation)

   ```bash
   # 运行测试，初始状态下所有测试应该失败并抛出 NotImplementedError
   uv run pytest
   ```

### 第二步：理解作业要求 (Step 2: Understand Assignment Requirements)

作业主要包含以下几个部分：

#### 核心部分 (Core Components)

1. **监督微调 (SFT - Supervised Fine-Tuning)**
   - `run_tokenize_prompt_and_output`: 对提示和输出进行分词，并构建响应掩码
   - `run_get_response_log_probs`: 获取响应在给定提示下的条件对数概率
   - `run_compute_entropy`: 计算对数概率的熵
   - `run_masked_mean`: 计算带掩码的张量均值
   - `run_masked_normalize`: 对带掩码的张量进行归一化
   - `run_sft_microbatch_train_step`: 实现SFT的微批次训练步骤

2. **策略梯度方法 (Policy Gradient Methods)**
   - `run_compute_naive_policy_gradient_loss`: 计算朴素策略梯度损失
   - `run_compute_grpo_clip_loss`: 计算GRPO-Clip损失
   - `run_compute_policy_gradient_loss`: 策略梯度损失包装函数
   - `run_grpo_microbatch_train_step`: 实现GRPO的微批次训练步骤

3. **GRPO奖励计算 (GRPO Reward Computation)**
   - `run_compute_group_normalized_rewards`: 计算组归一化奖励

#### 可选部分 (Optional Components)

4. **数据处理 (Data Processing)**
   - `get_packed_sft_dataset`: 构建打包的SFT数据集
   - `run_iterate_batches`: 实现批次迭代器

5. **响应解析 (Response Parsing)**
   - `run_parse_mmlu_response`: 解析MMLU响应
   - `run_parse_gsm8k_response`: 解析GSM8K响应

6. **DPO损失 (DPO Loss)**
   - `run_compute_per_instance_dpo_loss`: 计算每个实例的DPO损失

### 第三步：实现核心函数 (Step 3: Implement Core Functions)

所有需要实现的函数都在 `tests/adapters.py` 文件中，这些函数目前都抛出 `NotImplementedError`。

#### 实现顺序建议 (Recommended Implementation Order)

1. **基础工具函数** (Basic utility functions)
   - `run_masked_mean`: 带掩码的均值计算
   - `run_masked_normalize`: 带掩码的归一化
   - `run_compute_entropy`: 熵计算

2. **SFT相关函数** (SFT-related functions)
   - `run_tokenize_prompt_and_output`: 分词和掩码构建
   - `run_get_response_log_probs`: 获取对数概率
   - `run_sft_microbatch_train_step`: SFT训练步骤

3. **策略梯度函数** (Policy gradient functions)
   - `run_compute_naive_policy_gradient_loss`: 朴素策略梯度
   - `run_compute_grpo_clip_loss`: GRPO-Clip损失
   - `run_compute_policy_gradient_loss`: 策略梯度损失包装
   - `run_grpo_microbatch_train_step`: GRPO训练步骤

4. **奖励计算** (Reward computation)
   - `run_compute_group_normalized_rewards`: 组归一化奖励

5. **可选函数** (Optional functions)
   - 根据作业要求实现数据处理、响应解析和DPO相关函数

### 第四步：测试实现 (Step 4: Test Your Implementation)

1. **运行所有测试** (Run all tests)

   ```bash
   uv run pytest
   ```

2. **运行特定测试文件** (Run specific test files)

   ```bash
   # SFT相关测试
   uv run pytest tests/test_sft.py -v
   
   # GRPO相关测试
   uv run pytest tests/test_grpo.py -v
   
   # DPO相关测试
   uv run pytest tests/test_dpo.py -v
   
   # 指标相关测试
   uv run pytest tests/test_metrics.py -v
   
   # 数据相关测试
   uv run pytest tests/test_data.py -v
   ```

3. **运行单个测试** (Run a single test)

   ```bash
   uv run pytest tests/test_sft.py::test_tokenize_prompt_and_output -v
   ```

4. **查看详细输出** (View detailed output)

   ```bash
   # 显示更详细的输出
   uv run pytest -v -s
   ```

### 第五步：验证实现正确性 (Step 5: Verify Implementation Correctness)

测试使用快照测试（snapshot testing）来验证实现的正确性。确保：

1. **所有测试通过** (All tests pass)
   - 运行 `uv run pytest` 应该显示所有测试通过

2. **检查数值精度** (Check numerical precision)
   - 确保浮点数计算的精度在可接受范围内
   - 注意梯度计算的正确性

3. **验证掩码处理** (Verify mask handling)
   - 确保掩码正确应用于所有相关计算
   - 验证填充token被正确忽略

### 第六步：准备提交 (Step 6: Prepare Submission)

1. **运行测试并生成提交文件** (Run tests and create submission)

   ```bash
   # 运行测试脚本，会自动生成提交zip文件
   bash test_and_make_submission.sh
   ```

   这个脚本会：
   - 运行所有测试（即使失败也会继续）
   - 生成 `cs336-spring2025-assignment-5-submission.zip` 文件
   - 排除不必要的文件（如缓存、数据文件等）

2. **检查提交文件** (Check submission file)

   确保提交文件包含：
   - 所有源代码文件（`cs336_alignment/` 目录）
   - 所有测试文件（`tests/` 目录）
   - 配置文件（`pyproject.toml`, `uv.lock` 等）
   - 实现文件（`tests/adapters.py`）

   但不包含：
   - 数据文件（`data/` 目录）
   - 缓存文件（`__pycache__/`, `.pytest_cache/` 等）
   - 模型文件（`.bin`, `.pt`, `.pth`, `.safetensors`）
   - 日志文件（`.log`, `.out`, `.err`）

### 第七步：最终检查清单 (Step 7: Final Checklist)

在提交前，请确认：

- [ ] 所有核心函数已实现
- [ ] 所有测试通过（`uv run pytest`）
- [ ] 代码符合作业要求
- [ ] 提交文件已生成（`cs336-spring2025-assignment-5-submission.zip`）
- [ ] 提交文件大小合理（不应包含大型数据或模型文件）

### 常见问题 (Common Issues)

1. **依赖安装问题** (Dependency installation issues)
   - 如果 `flash-attn` 安装失败，检查CUDA版本兼容性
   - 确保Python版本在3.11-3.12之间

2. **测试失败** (Test failures)
   - 检查实现是否正确处理了边界情况
   - 验证掩码和填充的处理
   - 确保梯度计算正确

3. **内存问题** (Memory issues)
   - 如果遇到内存不足，考虑减少批次大小
   - 检查是否有内存泄漏

### 参考资料 (References)

- 作业PDF: `cs336_spring2025_assignment5_alignment.pdf`
- 补充材料: `cs336_spring2025_assignment5_supplement_safety_rlhf.pdf`
- GRPO论文: 
  - DeepSeekMath: https://arxiv.org/abs/2402.03300
  - DeepSeek-R1: https://arxiv.org/abs/2501.12948

### 获取帮助 (Getting Help)

如果遇到问题：
1. 查看作业PDF中的详细说明
2. 检查测试文件中的示例用法
3. 查看CHANGELOG.md了解更新
4. 在GitHub上提交issue或pull request

---

## 原始说明 (Original Instructions)

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).
