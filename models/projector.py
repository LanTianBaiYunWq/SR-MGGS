"""
投影模块
用于将学生特征映射到教师特征空间
"""

from typing import Optional

import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """
    MLP投影器
    将学生特征投影到教师特征空间
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        output_dim: int = 2048,
        num_layers: int = 2,
        use_bn: bool = False,
        dropout: float = 0.0
    ):
        """
        Args:
            input_dim: 输入维度（学生特征维度）
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（教师特征维度）
            num_layers: 层数
            use_bn: 是否使用BatchNorm
            dropout: dropout率
        """
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            next_dim = output_dim if is_last else hidden_dim
            
            # 线性层
            layers.append(nn.Linear(current_dim, next_dim))
            
            if not is_last:
                # BatchNorm (可选)
                if use_bn:
                    layers.append(nn.BatchNorm1d(next_dim))
                
                # 激活函数
                layers.append(nn.GELU())
                
                # Dropout
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            
            current_dim = next_dim
        
        self.projector = nn.Sequential(*layers)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            
        Returns:
            投影后特征 [batch_size, output_dim]
        """
        return self.projector(x)


class SimCLRProjector(nn.Module):
    """
    SimCLR风格的投影头
    用于对比学习
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 128,
        num_layers: int = 2,
        use_layer_norm: bool = True
    ):
        """
        Args:
            input_dim: ????
            hidden_dim: ?????
            output_dim: ????
            num_layers: ??
            use_layer_norm: ???? LayerNorm ?? BatchNorm?
        """
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            next_dim = output_dim if is_last else hidden_dim

            layers.append(nn.Linear(current_dim, next_dim, bias=False))

            if not is_last:
                # ???????? batch ?????BatchNorm1d ? batch_size=1
                # ??????????????????? LayerNorm?
                if use_layer_norm:
                    layers.append(nn.LayerNorm(next_dim))
                else:
                    layers.append(nn.BatchNorm1d(next_dim))
                layers.append(nn.ReLU(inplace=True))

            current_dim = next_dim

        self.projector = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class DistillProjector(nn.Module):
    """
    蒸馏投影器
    专门用于知识蒸馏的投影模块
    """
    
    def __init__(
        self,
        student_dim: int = 512,
        teacher_dim: int = 2048,
        hidden_dim: Optional[int] = None,
        normalize: bool = True
    ):
        """
        Args:
            student_dim: 学生特征维度
            teacher_dim: 教师特征维度
            hidden_dim: 隐藏层维度（None表示自动计算）
            normalize: 是否L2归一化
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = (student_dim + teacher_dim) // 2
        
        self.projector = nn.Sequential(
            nn.Linear(student_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, teacher_dim)
        )
        
        self.normalize = normalize
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 学生特征
            
        Returns:
            投影后特征
        """
        x = self.projector(x)
        
        if self.normalize:
            x = nn.functional.normalize(x, dim=-1)
        
        return x


class MultiHeadProjector(nn.Module):
    """
    多头投影器
    输出多个签名头，用于不同用途
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        output_dims: list = [256, 256],
        hidden_dim: int = 512
    ):
        """
        Args:
            input_dim: 输入维度
            output_dims: 各头输出维度列表
            hidden_dim: 共享隐藏层维度
        """
        super().__init__()
        
        # 共享backbone
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 多个投影头
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in output_dims
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        head_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征
            head_idx: 头索引（None表示返回所有头）
            
        Returns:
            投影后特征
        """
        shared_feat = self.shared(x)
        
        if head_idx is not None:
            return self.heads[head_idx](shared_feat)
        
        return [head(shared_feat) for head in self.heads]


class GSDModel(nn.Module):
    """
    GSD完整模型
    整合几何编码器、投影器和可选的对比头
    """
    
    def __init__(
        self,
        geometry_encoder: nn.Module,
        projector: nn.Module,
        contrastive_head: Optional[nn.Module] = None
    ):
        """
        Args:
            geometry_encoder: 几何编码器
            projector: 蒸馏投影器
            contrastive_head: 对比学习投影头（可选）
        """
        super().__init__()
        
        self.encoder = geometry_encoder
        self.projector = projector
        self.contrastive_head = contrastive_head
    
    def forward(
        self,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_all: bool = False
    ):
        """
        前向传播
        
        Args:
            coords: 坐标
            mask: padding掩码
            return_all: 是否返回所有中间结果
            
        Returns:
            根据return_all返回不同内容
        """
        # 几何编码
        embedding = self.encoder(coords, mask)
        
        # 蒸馏投影
        projected = self.projector(embedding)
        
        if return_all:
            result = {
                'embedding': embedding,
                'projected': projected
            }
            
            # 对比特征
            if self.contrastive_head is not None:
                result['contrastive'] = self.contrastive_head(embedding)
            
            return result
        
        return projected
    
    def get_embedding(
        self,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取几何嵌入（用于签名生成）
        
        Args:
            coords: 坐标
            mask: padding掩码
            
        Returns:
            几何嵌入
        """
        return self.encoder(coords, mask)


def create_projector(config: dict) -> nn.Module:
    """
    根据配置创建投影器
    
    Args:
        config: 配置字典
        
    Returns:
        投影器模型
    """
    return MLPProjector(
        input_dim=config.get('input_dim', 512),
        hidden_dim=config.get('hidden_dim', 1024),
        output_dim=config.get('output_dim', 2048),
        num_layers=config.get('num_layers', 2)
    )

