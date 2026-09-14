"""
eCTD元数据缓存管理器

提供两级缓存策略（内存LRU + 磁盘持久化），提升元数据提取性能。

Stage 1 Task 1.3 - 缓存管理模块
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional
import json
import hashlib
from datetime import datetime
from collections import OrderedDict

from core.ectd_metadata_extractor import SequenceMetadataIndex


# 缓存配置
CACHE_DIR = Path(".ectd_cache")  # 默认缓存目录
MAX_MEMORY_CACHE_SIZE = 50  # 内存缓存最大条目数
CACHE_VERSION = "1.0"  # 缓存格式版本


@dataclass
class CachedMetadataIndex:
    """缓存的元数据索引（带元信息）"""
    index: SequenceMetadataIndex
    cache_key: str
    cached_at: str  # ISO format timestamp
    source_mtime: float  # index.xml的修改时间


class SequenceMetadataCache:
    """序列元数据缓存管理器"""

    def __init__(self, cache_dir: Optional[Path] = None, max_memory_size: int = MAX_MEMORY_CACHE_SIZE):
        """
        初始化缓存管理器

        参数:
            cache_dir: 磁盘缓存目录，默认为当前目录下的.ectd_cache
            max_memory_size: 内存缓存最大条目数
        """
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(exist_ok=True)

        self.max_memory_size = max_memory_size

        # 内存缓存 (LRU: OrderedDict)
        self._memory_cache: OrderedDict[str, CachedMetadataIndex] = OrderedDict()

    def get(self, sequence_path: str) -> Optional[SequenceMetadataIndex]:
        """
        从缓存获取元数据索引

        参数:
            sequence_path: 序列目录路径

        返回:
            SequenceMetadataIndex对象，如果缓存不存在或已失效则返回None
        """
        cache_key = self._generate_cache_key(sequence_path)

        # 1. 尝试从内存缓存获取
        cached = self._get_from_memory(cache_key)
        if cached:
            # 验证缓存是否失效
            if self._is_cache_valid(sequence_path, cached):
                return cached.index
            else:
                # 失效，移除内存缓存
                self._remove_from_memory(cache_key)

        # 2. 尝试从磁盘缓存获取
        cached = self._get_from_disk(cache_key)
        if cached:
            # 验证缓存是否失效
            if self._is_cache_valid(sequence_path, cached):
                # 加载到内存缓存
                self._put_to_memory(cache_key, cached)
                return cached.index
            else:
                # 失效，移除磁盘缓存
                self._remove_from_disk(cache_key)

        return None

    def put(self, sequence_path: str, index: SequenceMetadataIndex) -> None:
        """
        将元数据索引放入缓存

        参数:
            sequence_path: 序列目录路径
            index: 元数据索引对象
        """
        cache_key = self._generate_cache_key(sequence_path)

        # 获取index.xml的修改时间
        index_xml_path = Path(sequence_path) / "index.xml"
        if not index_xml_path.exists():
            return

        source_mtime = index_xml_path.stat().st_mtime

        # 创建缓存对象
        cached = CachedMetadataIndex(
            index=index,
            cache_key=cache_key,
            cached_at=datetime.now().isoformat(),
            source_mtime=source_mtime
        )

        # 1. 放入内存缓存
        self._put_to_memory(cache_key, cached)

        # 2. 持久化到磁盘缓存
        self._put_to_disk(cache_key, cached)

    def invalidate(self, sequence_path: str) -> None:
        """
        使指定序列的缓存失效

        参数:
            sequence_path: 序列目录路径
        """
        cache_key = self._generate_cache_key(sequence_path)
        self._remove_from_memory(cache_key)
        self._remove_from_disk(cache_key)

    def clear(self) -> None:
        """清空所有缓存（内存+磁盘）"""
        self._memory_cache.clear()

        # 删除所有磁盘缓存文件
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()

    def get_stats(self) -> Dict[str, any]:
        """
        获取缓存统计信息

        返回:
            包含缓存统计的字典
        """
        disk_cache_count = len(list(self.cache_dir.glob("*.json")))

        return {
            "memory_cache_size": len(self._memory_cache),
            "memory_cache_max": self.max_memory_size,
            "disk_cache_size": disk_cache_count,
            "cache_directory": str(self.cache_dir)
        }

    # --- 私有方法 ---

    def _generate_cache_key(self, sequence_path: str) -> str:
        """生成缓存键（基于序列路径的哈希）"""
        normalized_path = Path(sequence_path).resolve()
        path_hash = hashlib.sha256(str(normalized_path).encode()).hexdigest()[:16]
        return f"seq_{path_hash}"

    def _get_from_memory(self, cache_key: str) -> Optional[CachedMetadataIndex]:
        """从内存缓存获取"""
        if cache_key in self._memory_cache:
            # LRU: 移到末尾（最近使用）
            self._memory_cache.move_to_end(cache_key)
            return self._memory_cache[cache_key]
        return None

    def _put_to_memory(self, cache_key: str, cached: CachedMetadataIndex) -> None:
        """放入内存缓存"""
        # 如果已存在，先删除（会更新到末尾）
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]

        self._memory_cache[cache_key] = cached

        # LRU淘汰：如果超过最大大小，删除最旧的
        if len(self._memory_cache) > self.max_memory_size:
            # popitem(last=False) 删除最早的项
            self._memory_cache.popitem(last=False)

    def _remove_from_memory(self, cache_key: str) -> None:
        """从内存缓存移除"""
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]

    def _get_from_disk(self, cache_key: str) -> Optional[CachedMetadataIndex]:
        """从磁盘缓存获取"""
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 验证缓存版本
            if data.get("cache_version") != CACHE_VERSION:
                return None

            # 重建对象（简化版，实际需要完整反序列化）
            # 这里假设存储了完整的JSON表示
            index_data = data.get("index")
            if not index_data:
                return None

            # 反序列化SequenceMetadataIndex
            index = self._deserialize_index(index_data)

            cached = CachedMetadataIndex(
                index=index,
                cache_key=data["cache_key"],
                cached_at=data["cached_at"],
                source_mtime=data["source_mtime"]
            )

            return cached

        except (json.JSONDecodeError, KeyError, Exception):
            # 缓存文件损坏，删除
            cache_file.unlink(missing_ok=True)
            return None

    def _put_to_disk(self, cache_key: str, cached: CachedMetadataIndex) -> None:
        """持久化到磁盘缓存"""
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            # 序列化
            data = {
                "cache_version": CACHE_VERSION,
                "cache_key": cached.cache_key,
                "cached_at": cached.cached_at,
                "source_mtime": cached.source_mtime,
                "index": self._serialize_index(cached.index)
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception:
            # 序列化失败，忽略（不阻塞主流程）
            pass

    def _remove_from_disk(self, cache_key: str) -> None:
        """从磁盘缓存移除"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        cache_file.unlink(missing_ok=True)

    def _is_cache_valid(self, sequence_path: str, cached: CachedMetadataIndex) -> bool:
        """
        验证缓存是否有效（基于mtime）

        参数:
            sequence_path: 序列目录路径
            cached: 缓存对象

        返回:
            True表示缓存有效，False表示已失效
        """
        index_xml_path = Path(sequence_path) / "index.xml"

        if not index_xml_path.exists():
            return False

        current_mtime = index_xml_path.stat().st_mtime

        # 如果文件修改时间变化，缓存失效
        return current_mtime == cached.source_mtime

    def _serialize_index(self, index: SequenceMetadataIndex) -> Dict:
        """序列化SequenceMetadataIndex为JSON兼容字典"""
        return {
            "sequence_number": index.sequence_number,
            "sequence_path": index.sequence_path,
            "envelope_version": index.envelope_version,
            "submission_description": index.submission_description,
            "total_sections": index.total_sections,
            "total_leafs": index.total_leafs,
            "sections": {
                key: self._serialize_section(section)
                for key, section in index.sections.items()
            }
        }

    def _serialize_section(self, section) -> Dict:
        """序列化SectionMetadataSnapshot"""
        from core.ectd_metadata_extractor import SectionMetadataSnapshot

        return {
            "sequence_number": section.sequence_number,
            "section_path": section.section_path,
            "xml_node_id": section.xml_node_id,
            "identifier": {
                "element_name": section.identifier.element_name,
                "attributes": section.identifier.attributes,
                "section_path": section.identifier.section_path,
            },
            "leaf_metadata": [
                {
                    "leaf_id": leaf.leaf_id,
                    "operation": leaf.operation,
                    "title": leaf.title,
                    "xlink_href": leaf.xlink_href,
                    "modified_file": leaf.modified_file,
                    "checksum": leaf.checksum,
                    "checksum_type": leaf.checksum_type
                }
                for leaf in section.leaf_metadata
            ]
        }

    def _deserialize_index(self, data: Dict) -> SequenceMetadataIndex:
        """反序列化JSON为SequenceMetadataIndex"""
        from core.ectd_metadata_extractor import SequenceMetadataIndex

        index = SequenceMetadataIndex(
            sequence_number=data["sequence_number"],
            sequence_path=data["sequence_path"],
            envelope_version=data.get("envelope_version"),
            submission_description=data.get("submission_description")
        )

        # 恢复sections
        for key, section_data in data.get("sections", {}).items():
            section = self._deserialize_section(section_data)
            index.sections[key] = section

        index.total_sections = data.get("total_sections", len(index.sections))
        index.total_leafs = data.get("total_leafs", 0)

        return index

    def _deserialize_section(self, data: Dict):
        """反序列化JSON为SectionMetadataSnapshot"""
        from core.ectd_metadata_extractor import SectionMetadataSnapshot, LeafMetadata
        from core.ectd_section_identifier import SectionIdentifier

        identifier = SectionIdentifier(
            element_name=data["identifier"]["element_name"],
            attributes=data["identifier"]["attributes"],
            section_path=data["identifier"]["section_path"]
        )

        leaf_metadata = [
            LeafMetadata(
                leaf_id=leaf_data["leaf_id"],
                operation=leaf_data["operation"],
                title=leaf_data["title"],
                xlink_href=leaf_data.get("xlink_href"),
                modified_file=leaf_data.get("modified_file"),
                checksum=leaf_data.get("checksum"),
                checksum_type=leaf_data.get("checksum_type")
            )
            for leaf_data in data.get("leaf_metadata", [])
        ]

        return SectionMetadataSnapshot(
            sequence_number=data["sequence_number"],
            identifier=identifier,
            leaf_metadata=leaf_metadata,
            section_path=data["section_path"],
            xml_node_id=data.get("xml_node_id")
        )


# 全局缓存实例（可选）
_global_cache: Optional[SequenceMetadataCache] = None


def get_global_cache() -> SequenceMetadataCache:
    """获取全局缓存实例（单例模式）"""
    global _global_cache
    if _global_cache is None:
        _global_cache = SequenceMetadataCache()
    return _global_cache


def extract_with_cache(sequence_path: str, use_cache: bool = True) -> SequenceMetadataIndex:
    """
    便捷函数：带缓存的元数据提取

    参数:
        sequence_path: 序列目录路径
        use_cache: 是否使用缓存

    返回:
        SequenceMetadataIndex对象

    示例:
        # 首次提取（会缓存）
        index = extract_with_cache("/path/to/0005")

        # 再次提取（从缓存读取）
        index = extract_with_cache("/path/to/0005")  # 快速返回

        # 强制重新提取
        index = extract_with_cache("/path/to/0005", use_cache=False)
    """
    cache = get_global_cache()

    if use_cache:
        # 尝试从缓存获取
        cached_index = cache.get(sequence_path)
        if cached_index:
            return cached_index

    # 缓存未命中或禁用，执行提取
    from core.ectd_metadata_extractor import extract_sequence_metadata_index
    index = extract_sequence_metadata_index(sequence_path)

    # 存入缓存
    if use_cache:
        cache.put(sequence_path, index)

    return index
