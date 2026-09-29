"""
教师特征投影器
===============================

本模块实现将视觉教师特征投影到几何编码器特征空间的投影器。

设计要点：
    1. 投影器参与训练（可训练），教师本体保持冻结
    2. 使用 LayerNorm 而非 BatchNorm，避免小批次问题
    3. 输出 L2 归一化，便于余弦距离计算

主要组件：
    - TeacherProjector: 将 teacher_feat_dim 映射到 d_model

使用示例：
    >>> projector = TeacherProjector(teacher_dim=2048, student_dim=256)
    >>> teacher_feat = teacher(images)  # [B, 2048]
    >>> aligned_feat = projector(teacher_feat)  # [B, 256]

作者：GSD Team
版本：1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherProjector(nn.Module):
    """
    教师特征投影器
    
    将教师特征（如 ResNet50 的 2048 维）投影到学生特征空间（如 256 维）
    
    特点：
    - 两层 MLP 结构
    - 使用 LayerNorm（避免 BatchNorm 的小批次问题）
    - 输出 L2 归一化（用于余弦距离）
    """
    
    def __init__(
        self,
        teacher_dim: int = 2048,
        student_dim: int = 256,
        hidden_dim: int = None,
        normalize_output: bool = True,
        dropout: float = 0.1
    ):
        """
        Args:
            teacher_dim: 教师特征维度（如 ResNet50 为 2048）
            student_dim: 学生特征维度（几何编码器输出维度）
            hidden_dim: 隐藏层维度（None 则自动计算）
            normalize_output: 是否 L2 归一化输出
            dropout: Dropout 率
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = (teacher_dim + student_dim) // 2
        
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        self.normalize_output = normalize_output
        
        # 两层 MLP
        self.projector = nn.Sequential(
            nn.Linear(teacher_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, student_dim),
            nn.LayerNorm(student_dim)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, teacher_feat: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            teacher_feat: 教师特征 [B, teacher_dim]
            
        Returns:
            投影后的特征 [B, student_dim]，可选 L2 归一化
        """
        x = self.projector(teacher_feat)
        
        if self.normalize_output:
            x = F.normalize(x, dim=-1)
        
        return x


class BidirectionalProjector(nn.Module):
    """
    双向投影器
    
    同时支持：
    - 教师 -> 学生空间投影（用于对齐损失）
    - 学生 -> 教师空间投影（可选，用于双向对齐）
    """
    
    def __init__(
        self,
        teacher_dim: int = 2048,
        student_dim: int = 256,
        hidden_dim: int = None,
        normalize_output: bool = True
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = (teacher_dim + student_dim) // 2
        
        # 教师 -> 学生
        self.t2s = nn.Sequential(
            nn.Linear(teacher_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, student_dim)
        )
        
        # 学生 -> 教师
        self.s2t = nn.Sequential(
            nn.Linear(student_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, teacher_dim)
        )
        
        self.normalize_output = normalize_output
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def project_teacher(self, teacher_feat: torch.Tensor) -> torch.Tensor:
        """教师 -> 学生空间"""
        x = self.t2s(teacher_feat)
        if self.normalize_output:
            x = F.normalize(x, dim=-1)
        return x
    
    def project_student(self, student_feat: torch.Tensor) -> torch.Tensor:
        """学生 -> 教师空间"""
        x = self.s2t(student_feat)
        if self.normalize_output:
            x = F.normalize(x, dim=-1)
        return x
    
    def forward(
        self,
        teacher_feat: torch.Tensor = None,
        student_feat: torch.Tensor = None
    ):
        """
        前向传播
        
        Args:
            teacher_feat: 教师特征（可选）
            student_feat: 学生特征（可选）
            
        Returns:
            投影后的特征（根据输入返回对应结果）
        """
        result = {}
        
        if teacher_feat is not None:
            result['t2s'] = self.project_teacher(teacher_feat)
        
        if student_feat is not None:
            result['s2t'] = self.project_student(student_feat)
        
        return result


def create_teacher_projector(config: dict) -> TeacherProjector:
    """
    根据配置创建教师投影器
    
    Args:
        config: 配置字典
        
    Returns:
        TeacherProjector
    """
    return TeacherProjector(
        teacher_dim=config.get('teacher_dim', 2048),
        student_dim=config.get('student_dim', 256),
        hidden_dim=config.get('hidden_dim', None),
        normalize_output=config.get('normalize_output', True),
        dropout=config.get('dropout', 0.1)
    )
