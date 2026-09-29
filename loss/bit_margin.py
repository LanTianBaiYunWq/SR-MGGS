"""
符号边界损失（Bit Margin Loss）
===============================

本模块实现零水印签名稳定性的关键损失函数。

设计目的：
    让最终签名的投影值远离 0，减少符号翻转的风险。
    这是零水印系统稳定性的核心保障。

原理：
    零水印签名生成流程：E → R·E = z → sign(z) → 签名 bits
    如果 z 的某个分量接近 0，微小扰动就可能导致 sign 翻转。
    通过损失函数让 |z| > margin，提高符号稳定性。

损失公式：
    L_sign = mean(relu(margin - z_q * z_k))  # 符号一致性
    L_amp  = mean(relu(margin - |z_q|)) + mean(relu(margin - |z_k|))  # 幅度边界
    L_bit  = L_sign + beta * L_amp

关键设计：
    1. 投影矩阵 R 由 seed 生成，训练/推理一致（不学习）
    2. 仅在训练时使用，推理仍走现有的签名链路
    3. margin 默认 0.5，可调

使用示例：
    >>> loss_fn = BitMarginLoss(d_model=256, n_bits=256, margin=0.5)
    >>> loss = loss_fn(E_q, E_k)

作者：GSD Team
版本：1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BitMarginLoss(nn.Module):
    """
    符号边界损失
    
    通过固定随机投影矩阵，让几何特征投影到签名空间时远离 0。
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_bits: int = 256,
        margin: float = 0.5,
        beta: float = 0.5,
        seed: int = 42,
        reduction: str = "mean"
    ):
        """
        Args:
            d_model: 几何特征维度
            n_bits: 签名位数（投影维度）
            margin: 幅度边界阈值
            beta: L_amp 的权重系数
            seed: 投影矩阵的随机种子（确保训练/推理一致）
            reduction: 归约方式
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_bits = n_bits
        self.margin = margin
        self.beta = beta
        self.reduction = reduction
        
        # 生成固定的随机投影矩阵（不参与训练）
        # 使用与推理时相同的 seed
        rng = np.random.default_rng(seed)
        R = rng.standard_normal((d_model, n_bits)).astype(np.float32)
        # 正交化（提高投影质量）
        R = self._orthogonalize(R)
        
        # 注册为 buffer（不参与梯度更新）
        self.register_buffer('R', torch.from_numpy(R))
    
    def _orthogonalize(self, R: np.ndarray) -> np.ndarray:
        """
        对投影矩阵进行正交化
        
        使用 QR 分解，保留 Q 的前 n_bits 列
        """
        if R.shape[0] >= R.shape[1]:
            Q, _ = np.linalg.qr(R)
            return Q.astype(np.float32)
        else:
            # d_model < n_bits 的情况
            Q, _ = np.linalg.qr(R.T)
            return Q.T.astype(np.float32)
    
    def project(self, E: torch.Tensor) -> torch.Tensor:
        """
        将特征投影到签名空间
        
        Args:
            E: 几何特征 [B, d_model]
            
        Returns:
            投影值 [B, n_bits]
        """
        # 先归一化
        E_norm = F.normalize(E, dim=-1)
        # 投影
        z = E_norm @ self.R  # [B, n_bits]
        return z
    
    def forward(
        self,
        E_q: torch.Tensor,
        E_k: torch.Tensor
    ) -> torch.Tensor:
        """
        计算符号边界损失
        
        Args:
            E_q: 几何特征 view A [B, d]
            E_k: 几何特征 view B [B, d]
            
        Returns:
            损失值
        """
        # 投影
        z_q = self.project(E_q)  # [B, n_bits]
        z_k = self.project(E_k)  # [B, n_bits]
        
        # 1. 符号一致性损失
        # z_q * z_k > 0 表示符号相同
        # 我们希望 z_q * z_k > margin
        sign_product = z_q * z_k  # [B, n_bits]
        L_sign = F.relu(self.margin - sign_product)  # [B, n_bits]
        
        # 2. 幅度边界损失
        # 希望 |z| > margin
        L_amp_q = F.relu(self.margin - z_q.abs())  # [B, n_bits]
        L_amp_k = F.relu(self.margin - z_k.abs())  # [B, n_bits]
        L_amp = L_amp_q + L_amp_k
        
        # 总损失
        loss = L_sign + self.beta * L_amp
        
        # 归约
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
    
    def get_bit_flip_rate(
        self,
        E_q: torch.Tensor,
        E_k: torch.Tensor
    ) -> float:
        """
        计算两个视图之间的签名比特翻转率
        
        用于评估签名稳定性
        
        Args:
            E_q: 几何特征 view A
            E_k: 几何特征 view B
            
        Returns:
            翻转率 (0~1)
        """
        with torch.no_grad():
            z_q = self.project(E_q)
            z_k = self.project(E_k)
            
            # 签名
            sig_q = (z_q > 0).float()
            sig_k = (z_k > 0).float()
            
            # 翻转率
            flip_rate = (sig_q != sig_k).float().mean().item()
            
            return flip_rate
    
    def generate_signature(self, E: torch.Tensor) -> torch.Tensor:
        """
        生成签名（仅用于调试/验证）
        
        实际推理应使用 utils/signature.py 中的方法
        
        Args:
            E: 几何特征 [B, d]
            
        Returns:
            签名 bits [B, n_bits]
        """
        with torch.no_grad():
            z = self.project(E)
            return (z > 0).int()


class AdaptiveMarginLoss(nn.Module):
    """
    自适应边界损失
    
    根据训练进度动态调整 margin
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_bits: int = 256,
        min_margin: float = 0.2,
        max_margin: float = 0.8,
        warmup_steps: int = 1000,
        beta: float = 0.5,
        seed: int = 42
    ):
        """
        Args:
            d_model: 特征维度
            n_bits: 签名位数
            min_margin: 最小边界
            max_margin: 最大边界
            warmup_steps: 预热步数
            beta: L_amp 权重
            seed: 随机种子
        """
        super().__init__()
        
        self.min_margin = min_margin
        self.max_margin = max_margin
        self.warmup_steps = warmup_steps
        
        self.base_loss = BitMarginLoss(
            d_model=d_model,
            n_bits=n_bits,
            margin=min_margin,
            beta=beta,
            seed=seed
        )
        
        self.register_buffer('step', torch.tensor(0))
    
    @property
    def current_margin(self) -> float:
        """计算当前 margin"""
        progress = min(1.0, self.step.item() / self.warmup_steps)
        return self.min_margin + progress * (self.max_margin - self.min_margin)
    
    def forward(
        self,
        E_q: torch.Tensor,
        E_k: torch.Tensor
    ) -> torch.Tensor:
        """计算损失"""
        # 更新 margin
        self.base_loss.margin = self.current_margin
        
        # 更新步数
        if self.training:
            self.step += 1
        
        return self.base_loss(E_q, E_k)
    
    def get_bit_flip_rate(
        self,
        E_q: torch.Tensor,
        E_k: torch.Tensor
    ) -> float:
        return self.base_loss.get_bit_flip_rate(E_q, E_k)


def create_bit_margin_loss(config: dict) -> nn.Module:
    """
    根据配置创建符号边界损失
    
    Args:
        config: 配置字典
        
    Returns:
        损失模块
    """
    if config.get('adaptive', False):
        return AdaptiveMarginLoss(
            d_model=config.get('d_model', 256),
            n_bits=config.get('n_bits', 256),
            min_margin=config.get('min_margin', 0.2),
            max_margin=config.get('max_margin', 0.8),
            warmup_steps=config.get('warmup_steps', 1000),
            beta=config.get('beta', 0.5),
            seed=config.get('seed', 42)
        )
    else:
        return BitMarginLoss(
            d_model=config.get('d_model', 256),
            n_bits=config.get('n_bits', 256),
            margin=config.get('margin', 0.5),
            beta=config.get('beta', 0.5),
            seed=config.get('seed', 42)
        )
