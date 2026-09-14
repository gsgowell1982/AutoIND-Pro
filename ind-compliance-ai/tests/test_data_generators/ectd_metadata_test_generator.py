"""
eCTD测试数据生成器

用于生成mock的eCTD序列样本，用于测试元数据生命周期耦合规则。

版本: v1.0
创建日期: 2026-09-11
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class MockLeaf:
    """Mock的Leaf元素"""
    leaf_id: str
    title: str
    operation: str  # new, delete, replace, append
    xlink_href: Optional[str] = None
    modified_file: Optional[str] = None
    checksum: str = "abc123"


@dataclass
class MockSection:
    """Mock的Section元素"""
    element_name: str
    attributes: Dict[str, str]
    leafs: List[MockLeaf]


@dataclass
class MockSequence:
    """Mock的eCTD序列"""
    sequence_number: str
    sections: List[MockSection]
    dtd_version: str = "3.2"


def generate_mock_index_xml(
    sequence: MockSequence,
    output_path: str
) -> str:
    """
    生成mock的index.xml文件

    Args:
        sequence: 序列定义
        output_path: 输出路径

    Returns:
        生成的index.xml路径
    """
    # 创建根元素
    root = ET.Element(
        "{http://www.ich.org/ectd}ectd",
        attrib={
            "dtd-version": sequence.dtd_version,
            "{http://www.w3c.org/1999/xlink}xlink": "http://www.w3c.org/1999/xlink"
        }
    )

    # 注册命名空间
    ET.register_namespace("ectd", "http://www.ich.org/ectd")
    ET.register_namespace("xlink", "http://www.w3c.org/1999/xlink")

    # 创建m2元素（示例）
    m2_element = ET.SubElement(root, "{http://www.ich.org/ectd}m2-common-technical-document-summaries")

    # 添加sections
    for section in sequence.sections:
        section_elem = ET.SubElement(m2_element, f"{{http://www.ich.org/ectd}}{section.element_name}")

        # 添加属性
        for attr_name, attr_value in section.attributes.items():
            section_elem.set(attr_name, attr_value)

        # 添加leafs
        for leaf in section.leafs:
            leaf_elem = ET.SubElement(section_elem, "{http://www.ich.org/ectd}leaf")
            leaf_elem.set("ID", leaf.leaf_id)
            leaf_elem.set("operation", leaf.operation)
            leaf_elem.set("checksum", leaf.checksum)
            leaf_elem.set("checksum-type", "md5")

            if leaf.xlink_href:
                leaf_elem.set("{http://www.w3c.org/1999/xlink}href", leaf.xlink_href)

            if leaf.modified_file:
                leaf_elem.set("modified-file", leaf.modified_file)

            # 添加title子元素
            title_elem = ET.SubElement(leaf_elem, "{http://www.ich.org/ectd}title")
            title_elem.text = leaf.title

    # 创建目录
    output_path_obj = Path(output_path)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    # 写入文件
    tree = ET.ElementTree(root)
    index_xml_path = output_path_obj / "index.xml"

    # 格式化输出
    _indent(root)
    tree.write(
        str(index_xml_path),
        encoding="utf-8",
        xml_declaration=True
    )

    return str(index_xml_path)


def _indent(elem, level=0):
    """美化XML缩进"""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


# ============================================================================
# 预定义的测试场景
# ============================================================================

def create_compliant_metadata_update_scenario() -> Dict[str, MockSequence]:
    """
    场景1：合规的元数据更新

    序列0004: substance="API-A", manufacturer="MFR-X"
      - leaf s001, s002

    序列0005: substance="API-A", manufacturer="MFR-Y" (元数据变更)
      - delete s001, s002
      - new s003, s004 (完整更新)

    预期：HR-ECTD-060 PASS
    """
    seq_0004 = MockSequence(
        sequence_number="0004",
        sections=[
            MockSection(
                element_name="m2-3-s-drug-substance",
                attributes={"substance": "API-A", "manufacturer": "MFR-X"},
                leafs=[
                    MockLeaf("s001", "质量标准.pdf", "new", "m2/23s/spec.pdf"),
                    MockLeaf("s002", "检验方法.pdf", "new", "m2/23s/method.pdf")
                ]
            )
        ]
    )

    seq_0005 = MockSequence(
        sequence_number="0005",
        sections=[
            MockSection(
                element_name="m2-3-s-drug-substance",
                attributes={"substance": "API-A", "manufacturer": "MFR-Y"},
                leafs=[
                    MockLeaf("s001", "质量标准.pdf", "delete", modified_file="s001"),
                    MockLeaf("s002", "检验方法.pdf", "delete", modified_file="s002"),
                    MockLeaf("s003", "质量标准-更新.pdf", "new", "m2/23s/spec-new.pdf"),
                    MockLeaf("s004", "检验方法-更新.pdf", "new", "m2/23s/method-new.pdf")
                ]
            )
        ]
    )

    return {"0004": seq_0004, "0005": seq_0005}


def create_metadata_only_change_violation_scenario() -> Dict[str, MockSequence]:
    """
    场景2：违规 - 仅更新元数据，不更新内容

    序列0004: substance="API-A", manufacturer="MFR-X"
      - leaf s001, s002

    序列0005: substance="API-A", manufacturer="MFR-Y" (元数据变更)
      - 没有任何leaf操作！

    预期：HR-ECTD-060 FAIL（元数据变更但没有内容更新）
    """
    seq_0004 = MockSequence(
        sequence_number="0004",
        sections=[
            MockSection(
                element_name="m2-3-s-drug-substance",
                attributes={"substance": "API-A", "manufacturer": "MFR-X"},
                leafs=[
                    MockLeaf("s001", "质量标准.pdf", "new", "m2/23s/spec.pdf"),
                    MockLeaf("s002", "检验方法.pdf", "new", "m2/23s/method.pdf")
                ]
            )
        ]
    )

    seq_0005 = MockSequence(
        sequence_number="0005",
        sections=[
            MockSection(
                element_name="m2-3-s-drug-substance",
                attributes={"substance": "API-A", "manufacturer": "MFR-Y"},
                leafs=[]  # 违规：元数据变更但没有leaf操作
            )
        ]
    )

    return {"0004": seq_0004, "0005": seq_0005}


def create_partial_update_violation_scenario() -> Dict[str, MockSequence]:
    """
    场景3：违规 - 部分更新内容

    序列0004: substance="API-A", manufacturer="MFR-X"
      - leaf s001, s002, s003 (3个文件)

    序列0005: substance="API-A", manufacturer="MFR-Y" (元数据变更)
      - delete s001
      - new s004
      - 违规：s002和s003没有处理

    预期：HR-ECTD-060 FAIL（部分更新）
    """
    seq_0004 = MockSequence(
        sequence_number="0004",
        sections=[
            MockSection(
                element_name="m3-2-s-drug-substance",
                attributes={"substance": "API-A", "manufacturer": "MFR-X"},
                leafs=[
                    MockLeaf("s001", "质量标准.pdf", "new", "m3/32s/spec.pdf"),
                    MockLeaf("s002", "检验方法.pdf", "new", "m3/32s/method.pdf"),
                    MockLeaf("s003", "稳定性数据.pdf", "new", "m3/32s/stability.pdf")
                ]
            )
        ]
    )

    seq_0005 = MockSequence(
        sequence_number="0005",
        sections=[
            MockSection(
                element_name="m3-2-s-drug-substance",
                attributes={"substance": "API-A", "manufacturer": "MFR-Y"},
                leafs=[
                    MockLeaf("s001", "质量标准.pdf", "delete", modified_file="s001"),
                    MockLeaf("s004", "质量标准-更新.pdf", "new", "m3/32s/spec-new.pdf")
                    # 违规：s002和s003未处理
                ]
            )
        ]
    )

    return {"0004": seq_0004, "0005": seq_0005}


def create_cross_module_inconsistency_violation_scenario() -> Dict[str, MockSequence]:
    """
    场景4：违规 - 跨模块属性不一致

    序列0005中：
      - 2.3.S: substance="API-A", manufacturer="MFR-Y"
      - 3.2.S: substance="API-A", manufacturer="MFR-X"
      - 违规：同一substance但manufacturer不一致

    预期：HR-ECTD-061 FAIL
    """
    seq_0005 = MockSequence(
        sequence_number="0005",
        sections=[
            MockSection(
                element_name="m2-3-s-drug-substance",
                attributes={"substance": "API-A", "manufacturer": "MFR-Y"},
                leafs=[
                    MockLeaf("s001", "质量标准.pdf", "new", "m2/23s/spec.pdf")
                ]
            ),
            MockSection(
                element_name="m3-2-s-drug-substance",
                attributes={"substance": "API-A", "manufacturer": "MFR-X"},  # 不一致！
                leafs=[
                    MockLeaf("s101", "质量标准.pdf", "new", "m3/32s/spec.pdf")
                ]
            )
        ]
    )

    return {"0005": seq_0005}


def create_replace_operation_scenario() -> Dict[str, MockSequence]:
    """
    场景5：使用Replace操作的合规更新

    序列0004: substance="API-A", manufacturer="MFR-X"
      - leaf s001, s002

    序列0005: substance="API-A", manufacturer="MFR-Y" (元数据变更)
      - replace s001 (modified-file="s001")
      - replace s002 (modified-file="s002")

    预期：HR-ECTD-060 PASS（replace算作删除+新增）
    """
    seq_0004 = MockSequence(
        sequence_number="0004",
        sections=[
            MockSection(
                element_name="m2-3-s-drug-substance",
                attributes={"substance": "API-A", "manufacturer": "MFR-X"},
                leafs=[
                    MockLeaf("s001", "质量标准.pdf", "new", "m2/23s/spec.pdf"),
                    MockLeaf("s002", "检验方法.pdf", "new", "m2/23s/method.pdf")
                ]
            )
        ]
    )

    seq_0005 = MockSequence(
        sequence_number="0005",
        sections=[
            MockSection(
                element_name="m2-3-s-drug-substance",
                attributes={"substance": "API-A", "manufacturer": "MFR-Y"},
                leafs=[
                    MockLeaf("s003", "质量标准.pdf", "replace", "m2/23s/spec-new.pdf", modified_file="s001"),
                    MockLeaf("s004", "检验方法.pdf", "replace", "m2/23s/method-new.pdf", modified_file="s002")
                ]
            )
        ]
    )

    return {"0004": seq_0004, "0005": seq_0005}


def create_complex_multi_substance_scenario() -> Dict[str, MockSequence]:
    """
    场景6：复杂场景 - 多个substance

    测试Section匹配的复合主键策略

    序列0004:
      - substance="API-A", manufacturer="MFR-X"
      - substance="API-A", manufacturer="MFR-Y"
      - substance="API-B", manufacturer="MFR-Z"

    序列0005:
      - substance="API-A", manufacturer="MFR-X" (保持不变)
      - substance="API-A", manufacturer="MFR-Y" → "MFR-Y2" (元数据变更)
      - substance="API-B" 被删除
      - substance="API-C", manufacturer="MFR-W" (新增)

    预期：
    - API-A/MFR-X: 匹配，无变更
    - API-A/MFR-Y: 匹配，元数据变更（需完整更新）
    - API-B/MFR-Z: 删除
    - API-C/MFR-W: 新增
    """
    seq_0004 = MockSequence(
        sequence_number="0004",
        sections=[
            MockSection(
                element_name="m2-3-s-drug-substance",
                attributes={"substance": "API-A", "manufacturer": "MFR-X"},
                leafs=[MockLeaf("s001", "API-A-X.pdf", "new", "m2/23s/a-x.pdf")]
            ),
            MockSection(
                element_name="m2-3-s-drug-substance",
                attributes={"substance": "API-A", "manufacturer": "MFR-Y"},
                leafs=[MockLeaf("s002", "API-A-Y.pdf", "new", "m2/23s/a-y.pdf")]
            ),
            MockSection(
                element_name="m2-3-s-drug-substance",
                attributes={"substance": "API-B", "manufacturer": "MFR-Z"},
                leafs=[MockLeaf("s003", "API-B-Z.pdf", "new", "m2/23s/b-z.pdf")]
            )
        ]
    )

    seq_0005 = MockSequence(
        sequence_number="0005",
        sections=[
            MockSection(
                element_name="m2-3-s-drug-substance",
                attributes={"substance": "API-A", "manufacturer": "MFR-X"},
                leafs=[]  # 保持不变，无leaf操作
            ),
            MockSection(
                element_name="m2-3-s-drug-substance",
                attributes={"substance": "API-A", "manufacturer": "MFR-Y2"},  # 元数据变更
                leafs=[
                    MockLeaf("s002", "API-A-Y.pdf", "delete", modified_file="s002"),
                    MockLeaf("s004", "API-A-Y2.pdf", "new", "m2/23s/a-y2.pdf")
                ]
            ),
            # API-B被删除（不出现在序列0005中）
            MockSection(
                element_name="m2-3-s-drug-substance",
                attributes={"substance": "API-C", "manufacturer": "MFR-W"},  # 新增
                leafs=[MockLeaf("s005", "API-C-W.pdf", "new", "m2/23s/c-w.pdf")]
            )
        ]
    )

    return {"0004": seq_0004, "0005": seq_0005}


# ============================================================================
# 测试数据生成主函数
# ============================================================================

def generate_all_test_scenarios(base_output_dir: str) -> Dict[str, Dict[str, str]]:
    """
    生成所有测试场景的mock数据

    Args:
        base_output_dir: 输出根目录

    Returns:
        场景映射，格式：
        {
            "scenario_1_compliant": {
                "0004": "/path/to/0004",
                "0005": "/path/to/0005"
            },
            ...
        }
    """
    base_path = Path(base_output_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    scenarios = {
        "scenario_1_compliant": create_compliant_metadata_update_scenario(),
        "scenario_2_metadata_only": create_metadata_only_change_violation_scenario(),
        "scenario_3_partial_update": create_partial_update_violation_scenario(),
        "scenario_4_cross_module": create_cross_module_inconsistency_violation_scenario(),
        "scenario_5_replace": create_replace_operation_scenario(),
        "scenario_6_complex": create_complex_multi_substance_scenario()
    }

    result = {}

    for scenario_name, sequences in scenarios.items():
        scenario_dir = base_path / scenario_name
        scenario_dir.mkdir(exist_ok=True)

        scenario_paths = {}
        for seq_num, seq_data in sequences.items():
            seq_path = scenario_dir / seq_num
            generate_mock_index_xml(seq_data, str(seq_path))
            scenario_paths[seq_num] = str(seq_path)

        result[scenario_name] = scenario_paths

    return result


if __name__ == "__main__":
    # 示例：生成测试数据
    output_dir = "D:/AutoIND-Pro/test_data/ectd_metadata_lifecycle"
    scenarios = generate_all_test_scenarios(output_dir)

    print("生成的测试场景：")
    for scenario_name, paths in scenarios.items():
        print(f"\n{scenario_name}:")
        for seq_num, path in sorted(paths.items()):
            print(f"  序列 {seq_num}: {path}")
