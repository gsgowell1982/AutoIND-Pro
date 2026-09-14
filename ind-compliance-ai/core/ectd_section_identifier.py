"""
eCTD Section标识和匹配模块

负责：
1. Section的唯一标识和匹配键生成
2. 复合主键策略
3. 跨模块配对关系
4. 生命周期追踪标记

版本: v1.0
创建日期: 2026-09-11
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal
from urllib.parse import quote, unquote
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Section匹配规则配置
# ============================================================================

SECTION_MATCHING_RULES = {
    # 模块2.3.S - 原料药（药学部分摘要）
    "m2-3-s-drug-substance": {
        "primary_keys": ["substance"],
        "match_strategy": "composite",
        "lifecycle_tracking": True,
        "cross_module_pair": "m3-2-s-drug-substance",
        "description": "原料药药学信息摘要"
    },

    # 模块3.2.S - 原料药（药学部分详细信息）
    "m3-2-s-drug-substance": {
        "primary_keys": ["substance"],
        "match_strategy": "composite",
        "lifecycle_tracking": True,
        "cross_module_pair": "m2-3-s-drug-substance",
        "description": "原料药药学信息详情"
    },

    # 模块2.3.P - 制剂（药学部分摘要）
    "m2-3-p-drug-product": {
        "primary_keys": ["product-name", "dosageform", "manufacturer"],
        "match_strategy": "composite",
        "lifecycle_tracking": True,
        "cross_module_pair": "m3-2-p-drug-product",
        "description": "制剂药学信息摘要"
    },

    # 模块3.2.P - 制剂（药学部分详细信息）
    "m3-2-p-drug-product": {
        "primary_keys": ["product-name", "dosageform", "manufacturer"],
        "match_strategy": "composite",
        "lifecycle_tracking": True,
        "cross_module_pair": "m2-3-p-drug-product",
        "description": "制剂药学信息详情"
    },

    # 模块2.7.3 - 临床有效性总结
    "m2-7-3-summary-of-clinical-efficacy": {
        "primary_keys": ["indication"],
        "match_strategy": "composite",
        "lifecycle_tracking": True,
        "cross_module_pair": "m5-3-5-reports-of-efficacy-and-safety-studies",
        "description": "临床有效性总结"
    },

    # 模块5.3.5 - 有效性和安全性研究报告
    "m5-3-5-reports-of-efficacy-and-safety-studies": {
        "primary_keys": ["indication"],
        "match_strategy": "composite",
        "lifecycle_tracking": True,
        "cross_module_pair": "m2-7-3-summary-of-clinical-efficacy",
        "description": "有效性和安全性研究报告"
    },

    # 模块3.2.P.4 - 辅料控制（可重复section）
    "m3-2-p-4-control-of-excipients": {
        "primary_keys": ["excipient"],
        "match_strategy": "composite",
        "lifecycle_tracking": False,  # 辅料不强制要求元数据生命周期耦合
        "cross_module_pair": None,
        "description": "辅料控制"
    }
}


# ============================================================================
# Section标识类
# ============================================================================

@dataclass
class SectionIdentifier:
    """
    Section唯一标识符

    用于：
    - 跨序列Section匹配
    - 跨模块Section配对
    - 元数据变更检测

    Attributes:
        element_name: Section元素名称（如"m2-3-s-drug-substance"）
        attributes: Section的所有属性（包括主键和非主键）

    Examples:
        >>> identifier = SectionIdentifier(
        ...     element_name="m2-3-s-drug-substance",
        ...     attributes={"substance": "API-A", "manufacturer": "MFR-X"}
        ... )
        >>> identifier.get_matching_key()
        'm2-3-s-drug-substance?manufacturer=MFR-X&substance=API-A'
        >>> identifier.get_cross_module_pair()
        'm3-2-s-drug-substance'
    """
    element_name: str
    attributes: Dict[str, str]

    def get_primary_keys(self) -> List[str]:
        """
        获取此element的主键列表

        Returns:
            主键属性名称列表。如果element未配置，返回所有属性。

        Notes:
            主键用于跨序列匹配同一个section。
            例如：m2-3-s的主键是["substance", "manufacturer"]
        """
        rule = SECTION_MATCHING_RULES.get(self.element_name, {})
        primary_keys = rule.get("primary_keys")

        if primary_keys:
            return primary_keys
        else:
            # 未配置的element，使用所有属性作为主键
            return list(self.attributes.keys())

    def get_matching_key(self) -> str:
        """
        生成用于跨序列匹配的键

        匹配键格式：element_name?key1=value1&key2=value2
        - 只包含主键属性
        - 键值对按字母序排序（保证一致性）
        - 值使用URL编码（处理特殊字符）

        Returns:
            匹配键字符串

        Examples:
            >>> identifier = SectionIdentifier(
            ...     "m2-3-s-drug-substance",
            ...     {"substance": "API-A", "manufacturer": "MFR-X"}
            ... )
            >>> identifier.get_matching_key()
            'm2-3-s-drug-substance?manufacturer=MFR-X&substance=API-A'

            >>> # 特殊字符自动编码
            >>> identifier2 = SectionIdentifier(
            ...     "m2-3-p-drug-product",
            ...     {"product-name": "Product A (Tablet)", "dosageform": "片剂"}
            ... )
            >>> key = identifier2.get_matching_key()
            >>> "Product+A+%28Tablet%29" in key
            True
        """
        primary_keys = self.get_primary_keys()

        # 构建键值对列表（只包含主键）
        key_value_pairs = []
        for key in sorted(primary_keys):  # 排序保证一致性
            value = self.attributes.get(key, "")
            # URL编码处理特殊字符（空格、引号、括号等）
            encoded_value = quote(value, safe="")
            key_value_pairs.append(f"{key}={encoded_value}")

        # 格式：element_name?key1=val1&key2=val2
        if key_value_pairs:
            return f"{self.element_name}?{'&'.join(key_value_pairs)}"
        else:
            return self.element_name

    def to_display_path(self) -> str:
        """
        生成人类可读的路径表示

        Returns:
            显示路径，格式：element_name[attr1='val1', attr2='val2']

        Examples:
            >>> identifier = SectionIdentifier(
            ...     "m2-3-s-drug-substance",
            ...     {"substance": "API-A", "manufacturer": "MFR-X"}
            ... )
            >>> identifier.to_display_path()
            "m2-3-s-drug-substance[substance='API-A', manufacturer='MFR-X']"
        """
        if not self.attributes:
            return self.element_name

        attr_strs = [f"{k}='{v}'" for k, v in self.attributes.items()]
        return f"{self.element_name}[{', '.join(attr_strs)}]"

    def get_cross_module_pair(self) -> Optional[str]:
        """
        获取跨模块配对的element名称

        Returns:
            配对的element名称，如果没有配对则返回None

        Examples:
            >>> identifier = SectionIdentifier("m2-3-s-drug-substance", {})
            >>> identifier.get_cross_module_pair()
            'm3-2-s-drug-substance'

            >>> identifier2 = SectionIdentifier("m3-2-p-4-control-of-excipients", {})
            >>> identifier2.get_cross_module_pair()
            None

        Notes:
            用于跨模块一致性验证：
            - 2.3.S ↔ 3.2.S
            - 2.3.P ↔ 3.2.P
            - 2.7.3 ↔ 5.3.5
        """
        rule = SECTION_MATCHING_RULES.get(self.element_name, {})
        return rule.get("cross_module_pair")

    def should_track_lifecycle(self) -> bool:
        """
        是否需要追踪元数据生命周期

        Returns:
            True如果需要追踪，False否则

        Notes:
            lifecycle_tracking=True的section需要执行元数据生命周期耦合验证：
            - 元数据变更时必须完整更新内容
            - 跨模块属性必须一致

        Examples:
            >>> identifier = SectionIdentifier("m2-3-s-drug-substance", {})
            >>> identifier.should_track_lifecycle()
            True

            >>> identifier2 = SectionIdentifier("m3-2-p-4-control-of-excipients", {})
            >>> identifier2.should_track_lifecycle()
            False
        """
        rule = SECTION_MATCHING_RULES.get(self.element_name, {})
        return rule.get("lifecycle_tracking", False)

    def get_description(self) -> str:
        """
        获取section的描述

        Returns:
            描述字符串
        """
        rule = SECTION_MATCHING_RULES.get(self.element_name, {})
        return rule.get("description", self.element_name)

    def matches(self, other: 'SectionIdentifier') -> bool:
        """
        判断是否与另一个SectionIdentifier匹配

        Args:
            other: 另一个SectionIdentifier

        Returns:
            True如果匹配（element名称相同且主键属性相同）

        Examples:
            >>> id1 = SectionIdentifier(
            ...     "m2-3-s-drug-substance",
            ...     {"substance": "API-A", "manufacturer": "MFR-X"}
            ... )
            >>> id2 = SectionIdentifier(
            ...     "m2-3-s-drug-substance",
            ...     {"substance": "API-A", "manufacturer": "MFR-Y"}  # 不同manufacturer
            ... )
            >>> id1.matches(id2)
            False

            >>> id3 = SectionIdentifier(
            ...     "m2-3-s-drug-substance",
            ...     {"substance": "API-A", "manufacturer": "MFR-X"}
            ... )
            >>> id1.matches(id3)
            True
        """
        return self.get_matching_key() == other.get_matching_key()

    def get_primary_attribute_values(self) -> Dict[str, str]:
        """
        获取主键属性的值

        Returns:
            主键属性字典

        Examples:
            >>> identifier = SectionIdentifier(
            ...     "m2-3-s-drug-substance",
            ...     {"substance": "API-A", "manufacturer": "MFR-X", "other": "value"}
            ... )
            >>> identifier.get_primary_attribute_values()
            {'substance': 'API-A', 'manufacturer': 'MFR-X'}
        """
        primary_keys = self.get_primary_keys()
        return {k: self.attributes.get(k, "") for k in primary_keys}

    def __eq__(self, other) -> bool:
        """相等性比较（基于匹配键）"""
        if not isinstance(other, SectionIdentifier):
            return False
        return self.matches(other)

    def __hash__(self) -> int:
        """哈希值（基于匹配键）"""
        return hash(self.get_matching_key())

    def __repr__(self) -> str:
        """字符串表示"""
        return f"SectionIdentifier({self.to_display_path()})"


# ============================================================================
# 工具函数
# ============================================================================

def create_section_identifier_from_xml_element(
    element_name: str,
    xml_attributes: Dict[str, str]
) -> SectionIdentifier:
    """
    从XML元素创建SectionIdentifier

    Args:
        element_name: XML元素名称（可能包含命名空间前缀）
        xml_attributes: XML元素的属性字典

    Returns:
        SectionIdentifier实例

    Examples:
        >>> identifier = create_section_identifier_from_xml_element(
        ...     "{http://www.ich.org/ectd}m2-3-s-drug-substance",
        ...     {"substance": "API-A", "manufacturer": "MFR-X"}
        ... )
        >>> identifier.element_name
        'm2-3-s-drug-substance'

    Notes:
        自动移除XML命名空间前缀
    """
    # 移除命名空间前缀
    clean_name = element_name.split("}")[-1] if "}" in element_name else element_name

    return SectionIdentifier(
        element_name=clean_name,
        attributes=xml_attributes
    )


def find_cross_module_pair(
    identifier: SectionIdentifier,
    candidate_sections: List[SectionIdentifier]
) -> Optional[SectionIdentifier]:
    """
    在候选列表中查找跨模块配对的section

    Args:
        identifier: 源section标识符
        candidate_sections: 候选section列表

    Returns:
        找到的配对section，如果没有则返回None

    Examples:
        >>> m2_section = SectionIdentifier(
        ...     "m2-3-s-drug-substance",
        ...     {"substance": "API-A", "manufacturer": "MFR-X"}
        ... )
        >>> m3_section = SectionIdentifier(
        ...     "m3-2-s-drug-substance",
        ...     {"substance": "API-A", "manufacturer": "MFR-X"}
        ... )
        >>> candidates = [m3_section]
        >>> pair = find_cross_module_pair(m2_section, candidates)
        >>> pair == m3_section
        True

    Notes:
        匹配条件：
        1. element_name是配对的模块
        2. 主键属性值相同
    """
    pair_element_name = identifier.get_cross_module_pair()
    if not pair_element_name:
        return None

    # 获取源section的主键值
    source_primary_values = identifier.get_primary_attribute_values()

    # 在候选中查找
    for candidate in candidate_sections:
        if candidate.element_name == pair_element_name:
            candidate_primary_values = candidate.get_primary_attribute_values()
            if source_primary_values == candidate_primary_values:
                return candidate

    return None


def get_configured_section_elements() -> List[str]:
    """
    获取所有已配置的section元素名称列表

    Returns:
        元素名称列表

    Examples:
        >>> elements = get_configured_section_elements()
        >>> "m2-3-s-drug-substance" in elements
        True
    """
    return list(SECTION_MATCHING_RULES.keys())


def is_section_configured(element_name: str) -> bool:
    """
    检查section元素是否已配置

    Args:
        element_name: 元素名称

    Returns:
        True如果已配置
    """
    return element_name in SECTION_MATCHING_RULES


def get_section_rule(element_name: str) -> Optional[Dict]:
    """
    获取section的匹配规则配置

    Args:
        element_name: 元素名称

    Returns:
        规则配置字典，如果未配置则返回None
    """
    return SECTION_MATCHING_RULES.get(element_name)
