import logging
from typing import Iterable

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vit as vit


logger = logging.getLogger()


# 专家提示模块
class Prompt(nn.Module):
    def __init__(
        self, 
        num_experts: int,        # 专家总数，即任务数
        len_prompt: int = 20,    # 每个 prompt 包含多少个 token
        embed_dim: int = 768,    # 每个 token 的维度，和 Vit 的嵌入维度一致
        pos_prompt: Iterable[int] = (0, 1, 2, 3, 4)  # prompt 要插入到 ViT 的哪些 block 层 (默认第0到第4层)
    ):

        super().__init__()
        
        self.num_experts = num_experts
        self.len_prompt = len_prompt
        self.embed_dim = embed_dim

        print("=== debug ===")
        print("list(pos_prompt)：")
        print(list(pos_prompt))
        print("=============")

        # pos_prompt 正好需要 buffer 这个级别的管理
        self.register_buffer('pos_prompt', torch.tensor(list(pos_prompt), dtype=torch.int64))
        self.num_layers = int(self.pos_prompt.numel())

        ## === 核心训练参数
        ## 这个四维张量存储了 所有专家索引 在 所有层 的 prompt 
        ## 第一维是层索引，第二维是专家索引，第三维和第四维是具体的每个 prompt token
        self.prompts = nn.Parameter(
            torch.empty(
                self.num_layers, # 插入的总层数
                num_experts,     # 专家总数，即任务数
                len_prompt,      # prompt 的长度，即每个 prompt 包含多少个 token
                embed_dim        # 每个 token 的嵌入向量维度
            )
        )
        # === 

        print("=== debug ===")
        print("self.prompts 的形状：")
        print(self.prompts.size())
        print("=============")

        nn.init.uniform_(self.prompts)

    def forward(
        self, 
        backbone: nn.Module, 
        inputs: torch.Tensor, 
        expert_ids: torch.Tensor
    ) -> torch.Tensor:

        print("=== debug ===")
        print("backbone.blocks 的数量：")
        print(len(backbone.blocks))
        print("=============")

        print("=== debug ===")
        print("backbone.blocks[0]：")
        print(backbone.blocks[0])
        print("=============")

        print("=== debug ===")
        print("backbone.blocks：")
        print(backbone.blocks)
        print("=============")

        print("=== debug ===")
        print("inputs 的形状：")
        print(inputs.size())
        print("=============")

        # 通过 ViT 的 patch embedding 层，得到 token 序列
        # (框架图中 Input 的产生)
        x = backbone.patch_embed(inputs)
        # B: batch 中的图片数量 (一般为 64 )
        # N: patch token 的数量 (一般为 196 = 14x14 )
        # D: 每个 patch token 的维度 (一般为 768 )
        B, N, D = x.size()
        # x: [64, 196, 768]

        print("=== debug ===")
        print("inputs 经过预训练模型后，x 的形状：")
        print(x.size())
        print("=============")

        print("=== debug ===")
        print("backbone.cls_token 的形状：")
        print(backbone.cls_token.size())
        print("=============")

        # backbone.cls_token 是预训练模型初始化时创建的一个可学习参数
        # 先将 cls_token 扩展到 batch 大小，准备拼接
        cls_token = backbone.cls_token.expand(B, -1, -1)    # -1 表示这个维度不变

        print("=== debug ===")
        print("扩展到 batch 大小后 cls_token 的形状：")
        print(cls_token.size())
        print("=============")

        # 拼接 cls_token 到 x 的 patch token 序列最前面
        token_appended = torch.cat((cls_token, x), dim=1)

        print("=== debug ===")
        print("拼接 cls_token 后，token_appended 的形状：")
        print(token_appended.size())
        print("=============")

        # 此时 cls token 和其它 patch token 混在一起
        # 此时 cls token 还不携带任何其它具体 patch token 的信息
        # 后面经过多层 transformer ，cls token 会逐渐从所有 patch token 中“吸收”信息
        # 最终变成整张图的全局表示，供分类头使用

        # 并且，此时还没有加位置编码
        # 如果没有位置编码，把 patch token 打乱顺序送进去，模型的输出完全一样
        # 因为 transformer 的自注意力本身不区分顺序
        # 加了位置编码后，模型就能区分“左上角的 patch ”和“右下角的 patch ”，空间结构被保留

        print("=== debug ===")
        print("backbone.pos_embed 的形状：")
        print(backbone.pos_embed.size())
        print("=============")

        # 加位置编码 + dropout
        # backbone.pos_embed 也是一个可学习参数，告诉模型“每个 token 在序列中的位置”
        # 加法时会自动广播到 batch 维度
        # pos_drop 就是一个普通的 Dropout 层，训练时 随机把一些值置零防止过拟合，推理时 不做任何操作
        x = backbone.pos_drop(token_appended + backbone.pos_embed)

        print("=== debug ===")
        print("加位置编码后 x 的形状：")
        print(x.size())
        print("=============")

        # 记住 token 序列的原始长度(197)，这个数字在后面的 prompt 插入-丢弃的循环中会用到
        orig_N = x.size(1)

        prompts = self._build_batched_prompts(backbone, expert_ids)  # [B, num_layers, len_prompt, D]

        print("=== debug ===")
        print("prompts 的形状：")
        print(prompts.size())
        print("=============")

        for n, block in enumerate(backbone.blocks):
            # eq(n)：找出等于 n 的位置 -> 结果为布尔掩码
            # nonzero()：提取 True 的索引 -> 结果为 2D 索引张量
            # squeeze()：去掉多余维度 -> 结果为 1D 索引张量

            print("=== debug ===")
            print("self.pos_prompt.eq(n) 的内容：")
            print(self.pos_prompt.eq(n))
            print("=============")

            print("=== debug ===")
            print("(self.pos_prompt.eq(n)).nonzero(as_tuple=False) 的内容：")
            print((self.pos_prompt.eq(n)).nonzero(as_tuple=False))
            print("=============")

            print("=== debug ===")
            print("(self.pos_prompt.eq(n)).nonzero(as_tuple=True) 的内容：")
            print((self.pos_prompt.eq(n)).nonzero(as_tuple=True))
            print("=============")

            pos_n = (self.pos_prompt.eq(n)).nonzero(as_tuple=False).squeeze()
            # 结果：找出 pos_prompt 中所有值等于 n 的位置索引，存入 pos_n

            print("=== debug ===")
            print("pos_n 的内容：")
            print(pos_n)
            print("=============")

            if pos_n.numel() != 0:
                # 将 prompt token 追加到 cls+patch token 末尾
                x = torch.cat((x, prompts[:, pos_n]), dim=1)

                print("=== debug ===")
                print("x 的形状：")
                print(x.size())
                print("=============")

                print("=== debug ===")
                print("prompts[0, pos_n] 的内容：")
                print(prompts[0, pos_n])
                print("=============")

                print("=== debug ===")
                print("x[0, orig_N:] 的内容：")
                print(x[0, orig_N:])
                print("=============")

            x = block(x)

            print("=== debug ===")
            print("x[0, orig_N:] 的内容：")
            print(x[0, orig_N:])
            print("=============")

            # 此时 x 中所有 token 的嵌入向量内容已更新，融入了 prompt token 的信息

            # 裁剪后保证 x 长度一致，下一层可以正常拼接
            # x 带着 prompt token 影响的新向量，传给下一层
            x = x[:, :orig_N, :]

        # 对经过所有 block 处理后的 x 做最终的层归一化 ( Lyayer Normalization )
        # 作用是让输出的特征分布更稳定，这是标准 ViT 结构的最后一步
        x = backbone.norm(x)

        print("=== debug ===")
        print("x[0, 0, :5] 的内容：")
        print(x[0, 0, :5])
        print("=============")

        # 取出 cls token 作为整张图的全局语义表示
        # 传给后续分类头进行预测
        return x[:, 0]

    @torch.no_grad()
    def init_new_expert(self, expert_id: int):
        # 边界检查：
        # expert_id == 0 ：第 0 个专家是第一个，没有前驱专家可以参考，跳过
        # expert_id >= self.num_experts ：越界，非法 id，跳过
        if expert_id == 0 or expert_id >= self.num_experts:
            return

        # 取出所有前驱专家的 prompt 
        # .clone() 是为了防止后续操作影响原数据
        prev_experts = self.prompts[:, :expert_id].clone()  # [num_layers, expert_id, L, D]
        # 在 专家的维度 取平均
        prev_experts_mean = prev_experts.mean(dim=1)        # [num_layers, L, D]
        # .data: 作用是绕过梯度追踪直接修改参数值
        # 如果不绕过梯度追踪，而 prev_experts_mean 来自 prev_experts.mean(dim=1)
        # 也就是说，新专家的梯度 流回了 旧专家的参数
        # 旧专家 因为“初始化新专家”这个操作，被错误地更新了
        self.prompts.data[:, expert_id] = prev_experts_mean

    def _build_batched_prompts(
        self, 
        backbone: nn.Module, 
        expert_ids: torch.Tensor
    ) -> torch.Tensor:

        print("=== debug ===")
        print("expert_ids 的形状：")
        print(expert_ids.size())
        print("=============")

        # 获取 batch 大小
        B = expert_ids.size(0)

        print("=== debug ===")
        print("expert_ids 的值：")
        print(expert_ids)
        print("=============")

        prompts = []  # 此时 prompts 是一个 list 列表
        for l_idx in range(self.num_layers):
            # [expert_ids.long()] 是 pytorch 的高级索引(fancy indexing)
            # 用一组下标（这里是专家索引，数量为 batch 个）来从第一维（专家维度）中选择对应的 prompt token 内容
            p_l = self.prompts[l_idx][expert_ids.long()]  # [B, len_prompt, D]

            print("=== debug ===")
            print("p_l 的形状：")
            print(p_l.size())
            print("=============")

            prompts.append(p_l)

            print("=== debug ===")
            print("prompts 的内容：")
            print(prompts)
            print("=============")

        # 最后把 prompts 这个 list 列表在第一维 stack 成一个四维张量
        # prompts[i, j] ：第 i 张图片在第 j 层 应该使用的 20 个 prompts token 内容
        prompts = torch.stack(prompts, dim=1)  # [B, num_layers, len_prompt, D]

        print("=== debug ===")
        print("prompts 的形状：")
        print(prompts.size())
        print("=============")

        # 之前 x 里面的每个 token 都已经加了位置编码了，所以这里 prompts 也要加上位置编码

        D = prompts.size(-1)

        print("=== debug ===")
        print("backbone.pos_embed[:, :1, :] 的形状：")
        print(backbone.pos_embed[:, :1, :].size())
        print("=============")

        print("=== debug ===")
        print("backbone.pos_embed[:, :1, :].unsqueeze(1) 的形状：")
        print(backbone.pos_embed[:, :1, :].unsqueeze(1).size())
        print("=============")

        # unsqueeze(1)：插入一个新维度，作为第 1 维
        pos_bias = backbone.pos_embed[:, :1, :].unsqueeze(1).expand(B, self.num_layers, self.len_prompt, D)

        print("=== debug ===")
        print("pos_bias 的形状：")
        print(pos_bias.size())
        print("=============")

        prompts = prompts + pos_bias

        print("=== debug ===")
        print("prompts 的形状：")
        print(prompts.size())
        print("=============")

        return prompts


class RPFC(nn.Module):
    def __init__(
        self,
        M            : int,            # 随机投影的目标维度（论文中默认 10000）
        ridge        : float = 1e4,    # 正则化参数 λ 
        embed_dim    : int = 768,      # 嵌入向量维度
        num_classes  : int = 100,      # 任务数，专家数
        **kwargs
    ):

        super().__init__()
        
        self.ridge = ridge
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        if M == 0:
            self.M = embed_dim
            self.use_rp = False
            self.register_buffer('W_rand', torch.empty(0))
            self.register_buffer('Q', torch.zeros(embed_dim, num_classes))
            self.register_buffer('G', torch.zeros(embed_dim, embed_dim))
        else:
            self.M = M
            self.use_rp = True
            # 固定的随机投影矩阵，对应果蝇的 PN -> KC 稀疏随机连接
            # 形状一般默认为 [ 768, 10000 ]，从标准正态分布中采样，不训练
            self.register_buffer('W_rand', torch.randn(embed_dim, M))
            # 原型矩阵 Q，每列对应每个分类的特征总和
            self.register_buffer('Q', torch.zeros(M, num_classes))
            # Gram 矩阵 G，累积特征维度之间的二阶相关性
            self.register_buffer('G', torch.zeros(M, M))

        # 路由分类头，权重就是路由矩阵 U
        # 映射到 num_classes 个专家中的一个
        self.fc = nn.Linear(self.M, num_classes, bias=False)

        # 所有参数都不需要梯度，因为路由矩阵 U 是闭式解，不用反向传播
        for param in self.parameters():
            param.requires_grad = False

    def target2onehot(self, targets):
        device = targets.device

        print("=== debug ===")
        print("targets 的形状：")
        print(targets.size())
        print("=============")

        print("=== debug ===")
        print("targets[:5] 的内容：")
        print(targets[:5])
        print("=============")

        onehot = torch.zeros(targets.size(0), self.num_classes, device=device)

        print("=== debug ===")
        print("onehot 的形状：")
        print(onehot.size())
        print("=============")

        print("=== debug ===")
        t = targets.unsqueeze(1)
        print("t 的形状：")
        print(t.size())
        print("=============")

        # 语法：tensor.scatter_(dim, index, value)
        # dim = 1   ->   在列的方向操作
        # index     ->   每行填 1 的位置
        # value     ->   填入的值
        # 整体含义：沿着 dim=1 的方向，按照索引张量指定的位置，填入值1

        # onehot 的形状：[64, 5]
        # targets 的形状：[64]
        # 要先将 targets 变为 [64, 1]
        # 维度数要对应上
        onehot.scatter_(1, targets.unsqueeze(1), 1)
        
        return onehot

    def collect(self, features, labels):
        # features 是什么

        print("=== debug ===")
        print("features 的形状：")
        print(features.size())
        print("=============")

        # features 就是原始图像经过冻结 ViT 完整前向传播后得到的 cls token 特征向量
        # 形状为 [64, 768]

        # 把 features 和 labels 从计算图中分离出来
        # 后续操作不会产生梯度
        # 因为这里只是在收集统计信息
        features = features.detach()
        labels = labels.detach()

        if self.use_rp:
            # 随机投影
            features_h = F.relu(features @ self.W_rand)
        else:
            # 直接用原始特征
            features_h = features
        
        Y = self.target2onehot(labels)

        self.Q = self.Q + features_h.T @ Y
        self.G = self.G + features_h.T @ features_h

    def update(self):
        device = self.fc.weight.device
        # 闭式求解路由矩阵：
        # 构造正则化矩阵
        # self.G + self.ridge * torch.eye(self.M, device=device)
        # torch.eye(self.M, device=device) 是 M * M 的单位矩阵
        # tortch.linalg.solve(A, B) 求解线性方程组： A X = B  X = A_-1 B
        Wo = torch.linalg.solve(self.G + self.ridge * torch.eye(self.M, device=device), self.Q).T
        self.fc.weight.data = Wo.to(device)

    def forward(self, x):
        if self.use_rp:
            # 如果使用随机投影

            print("=== debug ===")
            print("self.W_rand 的形状：")
            print(self.W_rand.size())
            print("=============")

            x = F.relu(x @ self.W_rand)
        
        x = self.fc(x)
        
        return x


class FlyPrompt(nn.Module):
    def __init__(
        self,
        task_num       : int   = 10,
        num_classes    : int   = 100,
        backbone_name  : str   = None,
        len_prompt     : int   = 20,        # 每个专家的 prompts token 的个数(长度)
        pos_prompt     : Iterable[int] = (0, 1, 2, 3, 4),   # prompts token 插入到 backbone 的哪些层
        rp_dim         : int   = 10000,
        rp_ridge       : float = 1e4,
        ema_ratio      : Iterable[float] = (0.9, 0.99),
        **kwargs
    ):

        super().__init__()

        self.kwargs = kwargs
        self.task_num = task_num
        self.num_classes = num_classes
        self.len_prompt = len_prompt
        self.pos_prompt = pos_prompt
        self.rp_dim = rp_dim
        self.rp_ridge = rp_ridge
        self.ema_ratio = ema_ratio
        self.num_ema = len(ema_ratio)

        self.task_count = 0

        # Backbone

        # 断言检查，确保传入了 backbone_name，否则直接报错提示
        assert backbone_name is not None, 'backbone_name must be specified'

        # Use custom ViT model from models.vit to support local .npz loading
        # self.add_module('backbone', ...) 等价于 self.backbone = ...
        # 将 backbone 注册为子模块，pytorch 能追踪它的参数
        if hasattr(vit, backbone_name):
            # 用自定义的 models.vit 模块加载
            logger.info(f'Using custom ViT model: {backbone_name}')
            self.add_module('backbone', getattr(vit, backbone_name)(pretrained=True, num_classes=num_classes))
        else:
            # 用 timm 库加载
            logger.info(f'Using timm model: {backbone_name}')
            self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=num_classes))

        print("=== debug ===")
        print("self.backbone.num_features 的值：")
        print(self.backbone.num_features)
        print("=============")

        # 获取特征维度数
        # 后续 prompt、分类头等模块都依赖这个维度数
        self.embed_dim = self.backbone.num_features

        # 冻结 backbone 的全部参数
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

        # 单独解冻分类头 —— 在线头
        # 为什么只训练分类头？
        # 分类头 (任务特定的映射)，针对当前任务微调
        # prompt (任务特定的引导信号)，引导 backbone 提取任务相关特征
        # backbone 不动，只训练少量参数 (分类头+prompt)
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad   = True

        # Expert prompts

        # 专家 prompt 模块
        self.experts = Prompt(
            num_experts = self.task_num,
            len_prompt = self.len_prompt,
            embed_dim = self.embed_dim,
            pos_prompt = self.pos_prompt,
        )

        # Expert FCs

        # 每个专家的 EMA 头列表
        self.experts_fc = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.embed_dim, self.num_classes, bias=True) for _ in range(self.num_ema)
            ]) for _ in range(self.task_num)
        ])

        # EMA 头不参与梯度训练
        for expert_fc in self.experts_fc:
            for fc in expert_fc:
                for param in fc.parameters():
                    param.requires_grad = False
        
        # 初始化专家 0 的分类头
        self.init_fc(expert_id = 0)

        # Random projection head

        # 随机梯度投影头
        self.rp_head = RPFC(
            M = self.rp_dim,
            ridge = self.rp_ridge,
            embed_dim = self.embed_dim,
            num_classes = self.task_num,
        )

    # forward 训练时用
    def forward(
        self, 
        inputs: torch.Tensor, 
        expert_ids: torch.Tensor = None, 
        **kwargs
    ) -> torch.Tensor:
        print("=== debug ===")
        print("inputs 的形状：")
        print(inputs.size())
        print("=============")

        # 训练时不传 expert_ids，默认把整个 batch 分配给当前任务的专家
        # 比如当前在训练任务 3，所有样本的 expert_ids 都设为 3
        if expert_ids is None:
            expert_ids = torch.full((inputs.size(0),), self.task_count, device=inputs.device, dtype=torch.long)

        print("=== debug ===")
        print("expert_ids 的形状：")
        print(expert_ids.size())
        print("=============")

        # 带 prompt 的前向传播：图像 -> patch embeding -> 插入专家prompt -> block 层 -> x (cls token 嵌入向量)
        x = self.experts(self.backbone, inputs, expert_ids)

        print("=== debug ===")
        print("x 的形状：")
        print(x.size())
        print("=============")

        # 通过在线分类头得到 logits
        x = self.backbone.fc(x)
        
        return x
    
    def forward_with_rp(
        self, 
        inputs: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor:
        print("=== debug ===")
        print("inputs 的形状：")
        print(inputs.size())
        print("=============")

        # 通过冻结的完整 ViT (不插prompt)
        x = self.backbone.forward_features(inputs)

        print("=== debug ===")
        print("x 的形状：")
        print(x.size())
        print("=============")

        # 获取通用 cls token 的嵌入向量
        x = x[:, 0]

        print("=== debug ===")
        print("x 的形状：")
        print(x.size())
        print("=============")

        # 通过 RP + FC 
        x = self.rp_head(x)

        print("=== debug ===")
        print("x 的形状：")
        print(x.size())
        print("=============")

        print("=== debug ===")
        print("x[0, :] 的内容：")
        print(x[0, :])
        print("=============")

        return x
    
    def forward_with_ema(
        self, 
        inputs: torch.Tensor, 
        expert_ids: torch.Tensor = None, 
        **kwargs
    ) -> torch.Tensor:
        if expert_ids is None:
            expert_ids = torch.full((inputs.size(0),), self.task_count, device=inputs.device, dtype=torch.long)

        # expert_ids 由上一步 forward_with_rp 的结果决定

        print("=== debug ===")
        print("expert_ids 的形状：")
        print(expert_ids.size())
        print("=============")

        print("=== debug ===")
        print("expert_ids[:5] 的值：")
        print(expert_ids[:5])
        print("=============")

        # 带 prompt 的前向传播，得到 cls token 嵌入向量
        x = self.experts(self.backbone, inputs, expert_ids)

        print("=== debug ===")
        print("x 的形状：")
        print(x.size())
        print("=============")

        outputs_ls = []

        # online head

        print("=== debug ===")
        oho = self.backbone.fc(x)
        print("oho 的形状：")
        print(oho.size())
        print("=============")

        print("=== debug ===")
        print("oho[0] 的内容：")
        print(oho[0])
        print("=============")

        outputs_ls.append(self.backbone.fc(x))

        print("=== debug ===")
        print("outputs_ls 的内容：")
        print(outputs_ls)
        print("=============")

        print("=== debug ===")
        print("x 的形状：")
        print(x.size())
        print("=============")

        print("=== debug ===")
        print("expert_ids 的形状：")
        print(expert_ids.size())
        print("=============")

        # ema head
        for i in range(self.num_ema):
            outputs = []
            # 逐样本处理，因为不同样本可能分配到不同专家
            # 每个样本要用自己对应专家的 EMA 头
            for x_i, e_i in zip(x, expert_ids):
                # self.experts_fc[e_i.item()][i]
                # 为当前样本 x_i 选择专家 e_i.item() 的 EMA 头 i 
                # 将当前样本的 x_i 送入专家 e_i.item() 的 EMA 头 i
                outputs.append(self.experts_fc[e_i.item()][i](x_i))

                print("=== debug ===")
                print("outputs 的内容：")
                print(outputs)
                print("=============")

            outputs = torch.stack(outputs, dim=0)

            print("=== debug ===")
            print("outputs 的形状：")
            print(outputs.size())
            print("=============")

            outputs_ls.append(outputs)

            print("=== debug ===")
            print("outputs_ls 的内容：")
            print(outputs_ls)
            print("=============")

        return outputs_ls
    
    def collect(self, inputs: torch.Tensor, labels: torch.Tensor):
        print("=== debug ===")
        print("inputs 的形状：")
        print(inputs.size())
        print("=============")

        print("=== debug ===")
        print("labels 的形状：")
        print(labels.size())
        print("=============")

        print("=== debug ===")
        print("labels[:5] 的内容：")
        print(labels[:5])
        print("=============")

        # 取 cls token 嵌入向量
        features = self.backbone.forward_features(inputs)
        features = features[:, 0]

        # 把 原始的类别标签 全部替换成了当前 任务编号
        # 为什么这样做？
        # 因为 rp_head (随机投影路由头) 的作用不是分类具体类别
        # 而是区分“这个样本属于哪个任务”
        # 它不关心这张图是猫还是狗，只关心这张图属于任务3
        labels = torch.full((labels.size(0),), self.task_count, device=labels.device, dtype=torch.long)

        print("=== debug ===")
        print("features 的形状：")
        print(features.size())
        print("=============")

        print("=== debug ===")
        print("labels 的形状：")
        print(labels.size())
        print("=============")

        print("=== debug ===")
        print("labels[:5] 的内容：")
        print(labels[:5])
        print("=============")

        self.rp_head.collect(features, labels)

    def update(self):
        # 当一个任务的所有数据都 collect 完之后
        # 调用 update 让 rp_head 根据收集到的统计量信息更新路由权重
        self.rp_head.update()

    @torch.no_grad()
    def init_fc(self, expert_id: int = None):
        if expert_id is None:
            # 没指定就用当前任务编号
            expert_id = self.task_count
        if expert_id >= self.task_num:
            # 如果超出范围就直接返回
            return
        # 从 正常训练的分类头 中取出当前的 权重 和 偏置
        w, b = self.backbone.fc.weight.data, self.backbone.fc.bias.data
        # 把这组 权重 和 偏置 复制到指定专家的每一个 EMA 头中
        for i in range(self.num_ema):
            self.experts_fc[expert_id][i].weight.data.copy_(w)
            self.experts_fc[expert_id][i].bias.data.copy_(b)

    @torch.no_grad()
    def update_ema_fc(self, expert_id: int = None):
        if expert_id is None:
            expert_id = self.task_count

        for i in range(self.num_ema):
            ema_ratio = self.ema_ratio[i]
            # 当前 在线头 权重 和 偏置
            online_w = self.backbone.fc.weight.data
            online_b = self.backbone.fc.bias.data

            print("=== debug ===")
            print("online_w 的形状：")
            print(online_w.size())
            print("=============")

            print("=== debug ===")
            print("online_b 的形状：")
            print(online_b.size())
            print("=============")

            # EMA头 的 权重 和 偏置
            ema_w = self.experts_fc[expert_id][i].weight.data
            ema_b = self.experts_fc[expert_id][i].bias.data

            # 更新 EMA
            # ema_w = ema_ratio * ema_w + (1 - ema_ratio) * online_w
            # ema_b = ema_ratio * ema_b + (1 - ema_ratio) * online_b
            ema_w.mul_(ema_ratio).add_(online_w, alpha=1.0 - ema_ratio)
            ema_b.mul_(ema_ratio).add_(online_b, alpha=1.0 - ema_ratio)

    def loss_fn(self, output, target):
        print("=== debug ===")
        print("output 的形状：")
        print(output.size())
        print("=============")

        print("=== debug ===")
        print("output[:1] 的内容：")
        print(output[:1])
        print("=============")

        print("=== debug ===")
        print("target 的形状：")
        print(target.size())
        print("=============")

        print("=== debug ===")
        print("target[:1] 的内容：")
        print(target[:1])
        print("=============")

        return F.cross_entropy(output, target)

    def process_task_count(self):
        # 这个函数在每个任务训练结束后调用一次，做收尾和准备工作

        # 任务计数器 + 1
        self.task_count += 1
        # 更新路由权重
        self.rp_head.update()
        # 为新任务初始化一个新 专家prompt
        self.experts.init_new_expert(self.task_count)
        # 把当前 backbone.fc 的 权重 和 偏置 复制给新 专家的所有 EMA 头，作为它们的初始值
        self.init_fc(self.task_count)