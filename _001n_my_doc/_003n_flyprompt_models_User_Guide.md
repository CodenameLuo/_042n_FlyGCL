# FlyPrompt 模型操作手册

> 从 `flyprompt_methods.py` 中提炼的核心调用逻辑，帮助你在自己的持续学习框架中正确驾驭 `flyprompt_models.py` 中的 FlyPrompt 模型。

---

## 1. 模型初始化

```python
from flyprompt_models import FlyPrompt

model = FlyPrompt(
    task_num      = 10,              # 总任务数（专家数）
    num_classes   = 100,             # 总类别数
    backbone_name = 'vit_base_patch16_224',  # ViT 预训练模型名
    len_prompt    = 20,              # 每个专家的 prompt token 数
    pos_prompt    = (0, 1, 2, 3, 4), # prompt 插入到 ViT 的哪些 block 层
    rp_dim        = 10000,           # 随机投影维度 M
    rp_ridge      = 1e4,             # 正则化参数 λ
    ema_ratio     = (0.9, 0.99),     # EMA 衰减率（短期记忆, 长期记忆）
).to(device)
```

---

## 2. 优化器设置

只有两组参数需要训练：**在线分类头** 和 **专家 prompt**。

```python
# 只收集需要梯度的参数
trainable_params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
```

其余参数（backbone、EMA 头、随机投影矩阵、路由头）全部冻结，不参与梯度更新。

---

## 3. 训练阶段（每个 batch）

每个 batch 按以下顺序执行 4 步操作：

```python
def train_one_batch(model, images, labels, optimizer, device):
    """
    images: [B, C, H, W] 原始图像
    labels: [B] 类别标签（已映射为 0 ~ num_classes-1 的整数）
    """
    model.train()

    images = images.to(device)
    labels = labels.to(device)

    # ====== 第 1 步：前向传播 + 构造掩码 + 计算损失 ======

    logits = model(images)  # 自动使用当前任务的专家 prompt + 在线头

    # 构造 logit 掩码：只保留当前 batch 中出现的类别
    mask = torch.full((model.num_classes,), float('-inf'), device=device)
    for c in torch.unique(labels):
        mask[c] = 0.0
    logits = logits + mask

    loss = F.cross_entropy(logits, labels)

    # ====== 第 2 步：反向传播，更新在线头 + 当前专家 prompt ======

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ====== 第 3 步：更新当前专家的 EMA 头（无梯度） ======

    model.update_ema_fc()   # 内部自动用 model.task_count 定位当前专家

    # ====== 第 4 步：收集路由统计量 G 和 Q（无梯度） ======

    with torch.no_grad():
        model.eval()
        model.collect(images, labels)

    return loss.item()
```

### 4 步操作对应论文的什么？

| 步骤 | 调用方法 | 论文对应 |
|------|---------|---------|
| 第 1 步 | `model(images)` + 掩码 | 公式 (6)：带 prompt 的前向传播 + logit mask + 交叉熵损失 |
| 第 2 步 | `loss.backward()` + `optimizer.step()` | 更新在线头 ψ 和当前专家 prompt p_t |
| 第 3 步 | `model.update_ema_fc()` | 公式 (7)：EMA 头的滑动平均更新 |
| 第 4 步 | `model.collect(images, labels)` | 公式 (2)：累积 Gram 矩阵 G 和原型矩阵 Q |

---

## 4. 任务切换时

当一个任务的所有 batch 训练完毕后，调用一次：

```python
model.process_task_count()
```

这一行代码内部完成了 3 件事：

| 操作 | 论文对应 |
|------|---------|
| `task_count += 1` | 任务计数器前进 |
| `rp_head.update()` | 公式 (4)：闭式求解路由矩阵 U |
| `experts.init_new_expert(task_count)` | 公式 (5)：用之前 prompt 的均值热启动新专家 |
| `init_fc(task_count)` | 把当前在线头参数克隆给新专家的 EMA 头作为初始值 |

---

## 5. 推理/评估阶段

推理时分 3 步：先路由选专家，再多头前向传播，最后集成输出。

```python
@torch.no_grad()
def evaluate(model, test_loader, device, num_classes):
    """
    注意：评估前需要确保路由矩阵已更新。
    如果是在任务结束后评估，process_task_count 已经调用过 update()。
    如果是在训练中途评估，需要手动调用一次 model.update()。
    """
    model.eval()
    model.update()  # 确保路由矩阵是最新的

    total_correct = 0
    total_samples = 0

    # 已见类别的全局掩码（推理时通常开放所有已见类别）
    # 你需要根据自己的框架维护已见类别列表
    global_mask = torch.full((num_classes,), float('-inf'), device=device)
    for c in seen_classes:
        global_mask[c] = 0.0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # ====== 第 1 步：路由——选专家 ======

        rp_logits = model.forward_with_rp(images)     # [B, T]
        expert_ids = torch.argmax(rp_logits, dim=-1)   # [B]

        # ====== 第 2 步：多头前向传播 ======

        logit_ls = model.forward_with_ema(images, expert_ids=expert_ids)
        # logit_ls 是一个列表：[在线头logits, EMA头1 logits, EMA头2 logits]
        # 每个元素形状为 [B, num_classes]

        # ====== 第 3 步：集成输出 ======

        # 加掩码
        logit_ls = [logit + global_mask for logit in logit_ls]

        # 每个头做 softmax，然后逐元素取 max（论文公式 10）
        prob_ls = [torch.softmax(logit, dim=-1) for logit in logit_ls]
        prob_stack = torch.stack(prob_ls, dim=-1)          # [B, C, n+1]
        ensemble_prob = prob_stack.max(dim=-1)[0]          # [B, C]
        preds = torch.argmax(ensemble_prob, dim=-1)        # [B]

        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return accuracy
```

### 推理 3 步对应论文的什么？

| 步骤 | 调用方法 | 论文对应 |
|------|---------|---------|
| 第 1 步 | `forward_with_rp` → `argmax` | 路由通路：完整 ViT → 随机投影 → 解析路由器 → 选专家 |
| 第 2 步 | `forward_with_ema` | 公式 (8)(9)：带 prompt 的 ViT → 在线头 + EMA 头分别计算 logits |
| 第 3 步 | `softmax` → `max` → `argmax` | 公式 (10)：时间集成，取逐元素最大概率后预测 |

---

## 6. 完整训练循环示例

```python
model = FlyPrompt(...).to(device)
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

for task_id in range(num_tasks):
    train_loader = get_task_dataloader(task_id)

    for images, labels in train_loader:
        train_one_batch(model, images, labels, optimizer, device)

    # 任务结束后：更新路由 + 热启动新专家 + 初始化新 EMA 头
    model.process_task_count()

    # 评估（可选）
    acc = evaluate(model, test_loader, device, num_classes)
    print(f"Task {task_id} done, accuracy: {acc:.4f}")
```

---

## 7. 关键注意事项

### 7.1 掩码的使用

- **训练时**：掩码只开放当前 batch 中出现的类别（按 batch 动态构造）
- **推理时**：掩码开放所有历史上见过的类别（全局维护）

### 7.2 collect 与 forward 的区别

| | `model(images)` （训练用） | `model.collect(images, labels)` （统计量收集） |
|---|---|---|
| ViT 使用方式 | 插入 prompt 的 ViT | 完整 ViT，不插 prompt |
| 标签含义 | 原始类别标签 | 被替换为当前任务 ID |
| 用途 | 分类，计算损失 | 累积 G 和 Q，供路由器使用 |
| 是否需要梯度 | 是 | 否 |

### 7.3 哪些参数在训练

| 组件 | 是否训练 | 说明 |
|------|---------|------|
| ViT backbone | ❌ 冻结 | 预训练参数不动 |
| 在线分类头 backbone.fc | ✅ 训练 | 唯一训练的分类头 |
| 专家 prompt | ✅ 训练 | 每个任务训练对应专家的 prompt |
| EMA 头 | ❌ 不训练 | 通过 EMA 滑动平均被动更新 |
| 随机投影矩阵 W_rand | ❌ 固定 | 初始化后永不改变 |
| 路由头 fc | ❌ 不训练 | 通过闭式解更新 |
| G 和 Q 矩阵 | ❌ 不训练 | 通过累积统计量更新 |

### 7.4 方法调用时机速查

| 方法 | 何时调用 | 频率 |
|------|---------|------|
| `model(images)` | 训练时前向传播 | 每个 batch |
| `model.update_ema_fc()` | 训练时，反向传播之后 | 每个 batch |
| `model.collect(images, labels)` | 训练时，收集路由统计量 | 每个 batch |
| `model.process_task_count()` | 一个任务训练结束后 | 每个任务结束时调用一次 |
| `model.update()` | 评估前确保路由矩阵最新 | 每次评估前 |
| `model.forward_with_rp(images)` | 推理时选专家 | 每个测试 batch |
| `model.forward_with_ema(images, expert_ids)` | 推理时多头集成 | 每个测试 batch |
