"""
签名生成工具
包含随机投影、量化、BCH纠错、HMAC密钥绑定
"""

import hashlib
import hmac
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

try:
    import bchlib
    BCH_AVAILABLE = True
except ImportError:
    BCH_AVAILABLE = False
    print("Warning: bchlib not installed. BCH error correction will be disabled.")


class RandomProjection(nn.Module):
    """
    随机超平面投影
    将高维嵌入投影到低维空间用于签名生成
    """
    
    def __init__(
        self, 
        input_dim: int = 512, 
        output_dim: int = 256,
        seed: int = 42
    ):
        """
        Args:
            input_dim: 输入特征维度
            output_dim: 输出签名位数
            seed: 随机种子（用于生成固定的投影矩阵）
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 使用固定种子生成随机投影矩阵
        rng = np.random.RandomState(seed)
        projection_matrix = rng.randn(input_dim, output_dim).astype(np.float32)
        # 正交化以提高稳定性
        projection_matrix, _ = np.linalg.qr(projection_matrix)
        
        self.register_buffer(
            'projection', 
            torch.from_numpy(projection_matrix)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行随机投影
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            
        Returns:
            投影后特征 [batch_size, output_dim]
        """
        return torch.matmul(x, self.projection)


class SignQuantizer(nn.Module):
    """
    符号量化器
    将连续值转换为二值签名
    """
    
    def __init__(self, method: str = "sign"):
        """
        Args:
            method: 量化方法 ("sign" 或 "median")
        """
        super().__init__()
        self.method = method
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行量化
        
        Args:
            x: 输入特征 [batch_size, dim]
            
        Returns:
            二值签名 [batch_size, dim]，值为0或1
        """
        if self.method == "sign":
            # 符号量化：正数->1，负数->0
            return (x > 0).float()
        elif self.method == "median":
            # 中值量化：大于中值->1，小于中值->0
            median = x.median(dim=-1, keepdim=True).values
            return (x > median).float()
        else:
            raise ValueError(f"Unknown quantization method: {self.method}")
    
    def to_bits(self, x: torch.Tensor) -> np.ndarray:
        """
        将量化后的tensor转换为bit数组
        
        Args:
            x: 量化后的tensor [batch_size, dim]
            
        Returns:
            bit数组 [batch_size, dim]
        """
        return x.cpu().numpy().astype(np.uint8)


class BCHEncoder:
    """
    BCH纠错编码器
    提供前向纠错能力
    """
    
    def __init__(self, poly: int = 137, bits: int = 5):
        """
        Args:
            poly: BCH多项式
            bits: 纠错位数
        """
        if not BCH_AVAILABLE:
            raise RuntimeError("bchlib not installed. Please install with: pip install bchlib")
        
        self.bch = bchlib.BCH(poly, bits)
        self.ecc_bytes = self.bch.ecc_bytes
    
    def encode(self, data: bytes) -> bytes:
        """
        对数据进行BCH编码
        
        Args:
            data: 原始数据
            
        Returns:
            编码后的数据（原始数据 + ECC）
        """
        ecc = self.bch.encode(data)
        return data + ecc
    
    def decode(self, data: bytes) -> Tuple[bytes, bool]:
        """
        对数据进行BCH解码
        
        Args:
            data: 编码后的数据
            
        Returns:
            (解码后的数据, 是否成功)
        """
        # 分离数据和ECC
        ecc_start = len(data) - self.ecc_bytes
        original = bytearray(data[:ecc_start])
        ecc = bytearray(data[ecc_start:])
        
        # 尝试纠错
        try:
            nerrors = self.bch.decode(original, ecc)
            if nerrors >= 0:
                self.bch.correct(original, ecc)
                return bytes(original), True
            else:
                return bytes(original), False
        except Exception:
            return bytes(original), False


class HMACBinder:
    """
    HMAC密钥绑定
    将签名与密钥绑定，增强安全性
    """
    
    def __init__(self, key: str = "gsd_secret_key"):
        """
        Args:
            key: HMAC密钥
        """
        self.key = key.encode('utf-8')
    
    def bind(self, signature: bytes) -> bytes:
        """
        对签名进行HMAC绑定
        
        Args:
            signature: 原始签名
            
        Returns:
            绑定后的签名
        """
        h = hmac.new(self.key, signature, hashlib.sha256)
        return signature + h.digest()
    
    def verify(self, bound_signature: bytes) -> Tuple[bytes, bool]:
        """
        验证HMAC绑定的签名
        
        Args:
            bound_signature: 绑定后的签名
            
        Returns:
            (原始签名, 是否验证成功)
        """
        signature = bound_signature[:-32]
        mac = bound_signature[-32:]
        
        expected_mac = hmac.new(self.key, signature, hashlib.sha256).digest()
        is_valid = hmac.compare_digest(mac, expected_mac)
        
        return signature, is_valid


class SignatureGenerator:
    """
    完整的签名生成器
    整合随机投影、量化、BCH纠错、HMAC绑定
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        signature_bits: int = 256,
        quantization: str = "sign",
        use_bch: bool = True,
        bch_poly: int = 137,
        bch_bits: int = 5,
        use_hmac: bool = True,
        hmac_key: str = "gsd_secret_key",
        seed: int = 42,
        device: str = "cuda"
    ):
        """
        Args:
            embed_dim: 嵌入维度
            signature_bits: 签名位数
            quantization: 量化方法
            use_bch: 是否使用BCH纠错
            bch_poly: BCH多项式
            bch_bits: BCH纠错位数
            use_hmac: 是否使用HMAC
            hmac_key: HMAC密钥
            seed: 随机种子
            device: 设备
        """
        self.signature_bits = signature_bits
        self.use_bch = use_bch and BCH_AVAILABLE
        self.use_hmac = use_hmac
        self.device = device
        
        # 初始化各组件
        self.projector = RandomProjection(embed_dim, signature_bits, seed).to(device)
        self.quantizer = SignQuantizer(quantization)
        
        if self.use_bch:
            self.bch = BCHEncoder(bch_poly, bch_bits)
        
        if self.use_hmac:
            self.hmac = HMACBinder(hmac_key)
    
    def generate(
        self, 
        embedding: torch.Tensor,
        return_intermediate: bool = False
    ) -> Union[bytes, Tuple[bytes, dict]]:
        """
        从嵌入生成签名
        
        Args:
            embedding: 嵌入向量 [batch_size, embed_dim] 或 [embed_dim]
            return_intermediate: 是否返回中间结果
            
        Returns:
            签名bytes，或(签名, 中间结果字典)
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        
        intermediates = {}
        
        # 1. 随机投影
        projected = self.projector(embedding)
        intermediates['projected'] = projected.cpu().numpy()
        
        # 2. 量化
        quantized = self.quantizer(projected)
        bits = self.quantizer.to_bits(quantized)
        intermediates['bits'] = bits
        
        # 3. 转换为bytes
        # 将bit数组打包为bytes
        signature_bytes = self._bits_to_bytes(bits[0])
        intermediates['raw_signature'] = signature_bytes
        
        # 4. BCH编码
        if self.use_bch:
            signature_bytes = self.bch.encode(signature_bytes)
            intermediates['bch_encoded'] = signature_bytes
        
        # 5. HMAC绑定
        if self.use_hmac:
            signature_bytes = self.hmac.bind(signature_bytes)
            intermediates['hmac_bound'] = signature_bytes
        
        if return_intermediate:
            return signature_bytes, intermediates
        return signature_bytes
    
    def verify(
        self,
        signature1: bytes,
        signature2: bytes,
        threshold: float = 0.1
    ) -> Tuple[bool, float]:
        """
        验证两个签名是否匹配
        
        Args:
            signature1: 第一个签名
            signature2: 第二个签名
            threshold: BER阈值
            
        Returns:
            (是否匹配, BER值)
        """
        # 1. 验证HMAC
        if self.use_hmac:
            sig1, valid1 = self.hmac.verify(signature1)
            sig2, valid2 = self.hmac.verify(signature2)
            if not (valid1 and valid2):
                return False, 1.0
        else:
            sig1, sig2 = signature1, signature2
        
        # 2. BCH解码
        if self.use_bch:
            sig1, success1 = self.bch.decode(sig1)
            sig2, success2 = self.bch.decode(sig2)
        
        # 3. 计算BER
        bits1 = self._bytes_to_bits(sig1)
        bits2 = self._bytes_to_bits(sig2)
        
        # 确保长度一致
        min_len = min(len(bits1), len(bits2))
        bits1 = bits1[:min_len]
        bits2 = bits2[:min_len]
        
        ber = np.mean(bits1 != bits2)
        
        return ber <= threshold, ber
    
    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        """将bit数组转换为bytes"""
        # 填充到8的倍数
        padded_len = (len(bits) + 7) // 8 * 8
        padded = np.zeros(padded_len, dtype=np.uint8)
        padded[:len(bits)] = bits
        
        # 打包
        packed = np.packbits(padded)
        return bytes(packed)
    
    def _bytes_to_bits(self, data: bytes) -> np.ndarray:
        """将bytes转换为bit数组"""
        arr = np.frombuffer(data, dtype=np.uint8)
        return np.unpackbits(arr)


def compute_ber(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """
    计算两个签名的比特错误率(BER)
    
    Args:
        sig1: 签名1 (bit数组)
        sig2: 签名2 (bit数组)
        
    Returns:
        BER值 (0-1之间)
    """
    assert len(sig1) == len(sig2), "Signature lengths must match"
    return np.mean(sig1 != sig2)


def compute_hamming_distance(sig1: np.ndarray, sig2: np.ndarray) -> int:
    """
    计算两个签名的汉明距离
    
    Args:
        sig1: 签名1 (bit数组)
        sig2: 签名2 (bit数组)
        
    Returns:
        汉明距离
    """
    return int(np.sum(sig1 != sig2))

