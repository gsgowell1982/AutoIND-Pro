"""
eCTD元数据提取器

从index.xml提取序列的元数据信息，构建用于跨序列对比的数据结构。

Stage 1 Task 1.1 - 元数据提取核心模块
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Set, List, Optional, Any
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

from core.ectd_section_identifier import SectionIdentifier, SECTION_MATCHING_RULES


# eCTD命名空间定义
ECTD_NAMESPACES = {
    'ectd': 'http://www.ich.org/ectd',
    'xlink': 'http://www.w3.org/1999/xlink'
}


@dataclass
class LeafMetadata:
    """单个leaf的元数据"""
    leaf_id: str
    operation: str  # new, delete, replace, append
    title: str
    xlink_href: Optional[str] = None
    modified_file: Optional[str] = None  # replace/delete/append的目标leaf ID
    checksum: Optional[str] = None
    checksum_type: Optional[str] = None


@dataclass
class SectionMetadataSnapshot:
    """单个section的元数据快照"""
    sequence_number: str
    identifier: SectionIdentifier
    leaf_metadata: List[LeafMetadata]  # 该section下所有leaf的详细信息
    section_path: str  # 在XML树中的路径
    xml_node_id: Optional[str] = None

    @property
    def leaf_ids(self) -> Set[str]:
        """返回leaf ID集合（向后兼容）"""
        return {leaf.leaf_id for leaf in self.leaf_metadata}

    def get_matching_key(self) -> str:
        """生成用于跨序列匹配的键"""
        return self.identifier.get_matching_key()

    def get_display_path(self) -> str:
        """生成人类可读的路径"""
        return self.identifier.to_display_path()


@dataclass
class SequenceMetadataIndex:
    """序列的完整元数据索引"""
    sequence_number: str
    sequence_path: str
    envelope_version: Optional[str] = None
    submission_description: Optional[str] = None

    # 核心数据：section快照字典 {matching_key: SectionMetadataSnapshot}
    sections: Dict[str, SectionMetadataSnapshot] = field(default_factory=dict)

    # 统计信息
    total_sections: int = 0
    total_leafs: int = 0

    def add_section(self, snapshot: SectionMetadataSnapshot) -> None:
        """添加section快照到索引"""
        key = snapshot.get_matching_key()
        self.sections[key] = snapshot
        self.total_sections += 1
        self.total_leafs += len(snapshot.leaf_metadata)

    def get_section_by_key(self, matching_key: str) -> Optional[SectionMetadataSnapshot]:
        """通过匹配键获取section快照"""
        return self.sections.get(matching_key)

    def get_all_sections(self) -> List[SectionMetadataSnapshot]:
        """获取所有section快照列表"""
        return list(self.sections.values())


class ECTDMetadataExtractor:
    """eCTD元数据提取器"""

    def __init__(self, sequence_path: str):
        """
        初始化提取器

        参数:
            sequence_path: 序列目录的绝对路径
        """
        self.sequence_path = Path(sequence_path)
        self.index_xml_path = self.sequence_path / "index.xml"

        if not self.index_xml_path.exists():
            raise FileNotFoundError(f"index.xml not found: {self.index_xml_path}")

    def extract_metadata_index(self) -> SequenceMetadataIndex:
        """
        提取序列的完整元数据索引

        返回:
            SequenceMetadataIndex对象，包含所有section和leaf信息

        异常:
            FileNotFoundError: index.xml不存在
            ET.ParseError: XML解析失败
        """
        # 解析XML
        tree = ET.parse(str(self.index_xml_path))
        root = tree.getroot()

        # 提取序列基本信息
        sequence_number = self._extract_sequence_number(root)
        envelope_version = self._extract_envelope_version(root)
        submission_description = self._extract_submission_description(root)

        # 创建索引对象
        index = SequenceMetadataIndex(
            sequence_number=sequence_number,
            sequence_path=str(self.sequence_path),
            envelope_version=envelope_version,
            submission_description=submission_description
        )

        # 遍历提取所有需要生命周期追踪的section
        self._extract_sections_recursive(root, index, section_path="")

        return index

    def _extract_sequence_number(self, root: Element) -> str:
        """从XML根元素提取序列号"""
        # 尝试多个可能的位置
        # 1. ectd:ectd/ectd:admin/ectd:sequence-number
        sequence_elem = root.find('.//ectd:sequence-number', ECTD_NAMESPACES)
        if sequence_elem is not None and sequence_elem.text:
            return sequence_elem.text.strip()

        # 2. 从目录名提取
        dir_name = self.sequence_path.name
        if dir_name.isdigit():
            return str(int(dir_name))

        # 3. 尝试从目录名中提取数字部分
        import re
        match = re.search(r'\d{4}', dir_name)
        if match:
            return str(int(match.group()))

        raise ValueError(f"Cannot extract sequence number from {self.index_xml_path}")

    def _extract_envelope_version(self, root: Element) -> Optional[str]:
        """提取eCTD envelope版本"""
        # 通常在根元素的属性中
        return root.get('dtd-version') or root.get('version')

    def _extract_submission_description(self, root: Element) -> Optional[str]:
        """提取提交描述"""
        desc_elem = root.find('.//ectd:submission-description', ECTD_NAMESPACES)
        if desc_elem is not None and desc_elem.text:
            return desc_elem.text.strip()
        return None

    def _extract_sections_recursive(
        self,
        element: Element,
        index: SequenceMetadataIndex,
        section_path: str
    ) -> None:
        """
        递归提取section元数据

        参数:
            element: 当前XML元素
            index: 要填充的索引对象
            section_path: 当前路径（用于调试）
        """
        # 获取元素的本地名称（去除命名空间前缀）
        tag_name = element.tag
        if '}' in tag_name:
            tag_name = tag_name.split('}')[1]

        # 检查是否是需要追踪的section类型
        if self._should_track_section(tag_name):
            # 提取section的元数据快照
            snapshot = self._extract_section_snapshot(element, tag_name, section_path, index.sequence_number)
            if snapshot:
                index.add_section(snapshot)

        # 递归处理子元素
        for child in element:
            child_path = f"{section_path}/{tag_name}" if section_path else tag_name
            self._extract_sections_recursive(child, index, child_path)

    def _should_track_section(self, tag_name: str) -> bool:
        """判断是否需要追踪此section类型"""
        return tag_name in SECTION_MATCHING_RULES

    def _extract_section_snapshot(
        self,
        element: Element,
        tag_name: str,
        section_path: str,
        sequence_number: str
    ) -> Optional[SectionMetadataSnapshot]:
        """
        从XML元素提取section快照

        参数:
            element: XML元素
            tag_name: 元素标签名
            section_path: 元素路径
            sequence_number: 序列号

        返回:
            SectionMetadataSnapshot对象，如果提取失败则返回None
        """
        # 提取所有属性
        attributes = dict(element.attrib)

        # 移除XML命名空间相关的属性
        attributes = {k: v for k, v in attributes.items() if not k.startswith('{')}

        # 创建SectionIdentifier
        identifier = SectionIdentifier(
            element_name=tag_name,
            attributes=attributes
        )

        # 提取该section下的所有leaf元数据
        leaf_metadata = self._extract_leaf_metadata(element)

        # 提取XML节点ID（如果有）
        xml_node_id = element.get('ID') or element.get('id')

        snapshot = SectionMetadataSnapshot(
            sequence_number=sequence_number,
            identifier=identifier,
            leaf_metadata=leaf_metadata,
            section_path=section_path,
            xml_node_id=xml_node_id
        )

        return snapshot

    def _extract_leaf_metadata(self, section_element: Element) -> List[LeafMetadata]:
        """
        提取section下所有leaf的元数据

        参数:
            section_element: section的XML元素

        返回:
            LeafMetadata对象列表
        """
        leafs = []

        # 查找所有leaf元素（可能是<leaf>或<ectd:leaf>）
        for leaf_elem in section_element.iter():
            tag_name = leaf_elem.tag
            if '}' in tag_name:
                tag_name = tag_name.split('}')[1]

            if tag_name != 'leaf':
                continue

            # 提取leaf属性
            leaf_id = leaf_elem.get('ID') or leaf_elem.get('id', '')
            operation = leaf_elem.get('operation', 'new')

            # 提取title
            title_elem = leaf_elem.find('ectd:title', ECTD_NAMESPACES)
            if title_elem is None:
                title_elem = leaf_elem.find('title')
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else ''

            # 提取xlink:href
            xlink_href = leaf_elem.get('{http://www.w3.org/1999/xlink}href')
            if not xlink_href:
                xlink_href = leaf_elem.get('href')

            # 提取modified-file (用于replace/delete/append操作)
            modified_file = leaf_elem.get('modified-file')

            # 提取checksum
            checksum_elem = leaf_elem.find('ectd:checksum', ECTD_NAMESPACES)
            if checksum_elem is None:
                checksum_elem = leaf_elem.find('checksum')

            checksum = None
            checksum_type = None
            if checksum_elem is not None:
                checksum = checksum_elem.text.strip() if checksum_elem.text else None
                checksum_type = checksum_elem.get('type', 'md5')

            leaf_meta = LeafMetadata(
                leaf_id=leaf_id,
                operation=operation,
                title=title,
                xlink_href=xlink_href,
                modified_file=modified_file,
                checksum=checksum,
                checksum_type=checksum_type
            )

            leafs.append(leaf_meta)

        return leafs


def extract_sequence_metadata_index(sequence_path: str) -> SequenceMetadataIndex:
    """
    便捷函数：提取序列的元数据索引

    参数:
        sequence_path: 序列目录路径

    返回:
        SequenceMetadataIndex对象

    示例:
        index = extract_sequence_metadata_index("/path/to/0005")
        print(f"序列 {index.sequence_number}")
        print(f"包含 {index.total_sections} 个sections")
        print(f"包含 {index.total_leafs} 个leafs")
    """
    extractor = ECTDMetadataExtractor(sequence_path)
    return extractor.extract_metadata_index()
