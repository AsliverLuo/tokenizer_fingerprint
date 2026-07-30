"""
reference_bank.py — 参考库管理

构建、保存、加载参考模型指纹库。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .schema import BankStatistics, ModelFingerprint
from .similarity import compute_bank_statistics

logger = logging.getLogger(__name__)


class ReferenceBank:
    """
    参考库：存储已知模型的指纹。

    目录结构：
        reference_bank/
        ├── index.json              # 索引文件
        ├── openai/
        │   ├── gpt-4o.json
        │   └── gpt-4o-mini.json
        ├── anthropic/
        │   └── claude-sonnet-4-20250514.json
        └── deepseek/
            └── deepseek-chat.json
    """

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.fingerprints: dict[str, ModelFingerprint] = {}
        self._index: list[dict] = []
        self._statistics: Optional[BankStatistics] = None

    @property
    def statistics(self) -> Optional[BankStatistics]:
        return self._statistics

    def compute_statistics(self) -> BankStatistics:
        """从已加载指纹计算跨家族 BCS 统计量，并缓存。"""
        fps = self.all_fingerprints()
        self._statistics = compute_bank_statistics(fps)
        return self._statistics

    def add(self, fingerprint: ModelFingerprint):
        """添加一个模型指纹"""
        self.fingerprints[fingerprint.model_name] = fingerprint
        logger.info(f"Added fingerprint: {fingerprint.model_name} (family={fingerprint.family})")

    def get(self, model_name: str) -> Optional[ModelFingerprint]:
        return self.fingerprints.get(model_name)

    def list_models(self) -> list[str]:
        return list(self.fingerprints.keys())

    def list_families(self) -> list[str]:
        return list(set(fp.family for fp in self.fingerprints.values()))

    def get_by_family(self, family: str) -> list[ModelFingerprint]:
        return [fp for fp in self.fingerprints.values() if fp.family == family]

    def all_fingerprints(self) -> list[ModelFingerprint]:
        return list(self.fingerprints.values())

    def save(self):
        """保存参考库到磁盘"""
        self.base_dir.mkdir(parents=True, exist_ok=True)

        index = []
        for name, fp in self.fingerprints.items():
            family_dir = self.base_dir / fp.family
            family_dir.mkdir(parents=True, exist_ok=True)

            # 文件名安全处理
            safe_name = name.replace("/", "_").replace(":", "_")
            fp_path = family_dir / f"{safe_name}.json"
            fp.save(fp_path, include_raw_results=True)

            index.append({
                "model_name": name,
                "family": fp.family,
                "path": str(fp_path.relative_to(self.base_dir)),
                "n_probes": fp.n_probes,
            })

        # 写索引
        with open(self.base_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(index)} fingerprints to {self.base_dir}")

    @classmethod
    def load(
        cls,
        base_dir: Path,
        bank_compare_path: Optional[str | Path] = None,
    ) -> "ReferenceBank":
        """从磁盘加载参考库，可选从 compare-bank JSON 加载统计量。

        Args:
            base_dir: 参考库目录
            bank_compare_path: 可选，compare-bank 输出的 JSON 文件路径，
                              用于提取跨家族 BCS 统计量。
        """
        base_dir = Path(base_dir)
        bank = cls(base_dir)

        index_path = base_dir / "index.json"
        if not index_path.exists():
            logger.warning(f"No index.json found in {base_dir}")
            return bank

        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        for entry in index:
            fp_path = base_dir / entry["path"]
            if fp_path.exists():
                fp = ModelFingerprint.load(fp_path)
                bank.fingerprints[fp.model_name] = fp
            else:
                logger.warning(f"Fingerprint file not found: {fp_path}")

        logger.info(f"Loaded {len(bank.fingerprints)} fingerprints from {base_dir}")

        # 可选加载 bank statistics
        if bank_compare_path:
            bc_path = Path(bank_compare_path)
            if bc_path.exists():
                bc_data = json.loads(bc_path.read_text(encoding="utf-8"))
                pairs = bc_data.get("pairs", [])
                if pairs:
                    bank._statistics = compute_bank_statistics(
                        bank.all_fingerprints(), bank_compare_pairs=pairs
                    )
                    logger.info(
                        f"Bank statistics loaded from {bc_path}: "
                        f"cross-fam mean={bank._statistics.cross_family_mean:.4f}, "
                        f"p99={bank._statistics.cross_family_p99:.4f}"
                    )

        return bank
