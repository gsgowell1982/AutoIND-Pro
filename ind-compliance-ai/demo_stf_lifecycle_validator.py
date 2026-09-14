"""
eCTD STF生命周期管理验证器 - 演示脚本

演示如何使用STF生命周期验证器检查跨序列的STF文件合规性。

Phase 2.10 - STF Lifecycle Management Validation
创建日期: 2026-09-14
"""

import sys
import io
from pathlib import Path
from typing import List
import tempfile
import xml.etree.ElementTree as ET

# 设置stdout为UTF-8编码（Windows兼容）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.ectd_stf_lifecycle_validator import (
    STFLifecycleValidator,
    STFLifecycleSnapshot,
    ViolationDetail,
    ViolationSeverity,
    validate_stf_lifecycle_pair,
    validate_stf_lifecycle_application
)


def create_demo_stf_file(
    file_path: Path,
    study_id: str,
    operation: str,
    modified_file: str = None,
    leaf_ids: List[str] = None
) -> str:
    """创建演示用的STF XML文件"""
    ET.register_namespace('ectd', 'http://www.ich.org/ectd')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')

    root = ET.Element("{http://www.ich.org/ectd}study")
    root.set("operation", operation)

    if modified_file:
        root.set("modified-file", modified_file)

    study_identifier = ET.SubElement(root, "{http://www.ich.org/ectd}study-identifier")
    study_title = ET.SubElement(study_identifier, "{http://www.ich.org/ectd}study-title")
    study_title.text = f"Study {study_id}"

    study_document = ET.SubElement(root, "{http://www.ich.org/ectd}study-document")

    if leaf_ids:
        for leaf_id in leaf_ids:
            leaf_ref = ET.SubElement(study_document, "{http://www.ich.org/ectd}leaf-reference")
            leaf_ref.set("{http://www.w3.org/1999/xlink}href", f"#{leaf_id}")

    tree = ET.ElementTree(root)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(file_path), encoding="utf-8", xml_declaration=True)

    return str(file_path)


def print_section_header(title: str):
    """打印章节标题"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_violations(violations: List[ViolationDetail]):
    """打印违规详情"""
    if not violations:
        print("✅ 未发现违规")
        return

    print(f"❌ 发现 {len(violations)} 个违规:")
    for i, violation in enumerate(violations, 1):
        severity_icon = {
            ViolationSeverity.CRITICAL: "🔴",
            ViolationSeverity.ERROR: "🔴",
            ViolationSeverity.WARNING: "🟡",
            ViolationSeverity.INFO: "🔵"
        }.get(violation.severity, "⚪")

        print(f"\n  {i}. {severity_icon} [{violation.rule_id}] {violation.severity.value}")
        print(f"     消息: {violation.message}")
        print(f"     位置: {violation.location}")
        print(f"     详情: {violation.details}")
        if violation.suggestion:
            print(f"     建议: {violation.suggestion}")


def demo_scenario_1_compliant():
    """场景1: 完全合规的序列对"""
    print_section_header("场景1: 完全合规的序列对")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        # 创建序列0000 (首次提交)
        seq_0000 = temp_dir / "0000"
        stf_0000 = seq_0000 / "m5" / "stf-study001.xml"
        create_demo_stf_file(
            stf_0000,
            study_id="study001",
            operation="new",
            leaf_ids=["leaf-001", "leaf-002"]
        )
        print(f"\n创建序列0000: {stf_0000}")
        print("  - operation='new' (首次提交)")
        print("  - leaf引用: leaf-001, leaf-002")

        # 创建序列0001 (后续提交)
        seq_0001 = temp_dir / "0001"
        stf_0001 = seq_0001 / "m5" / "stf-study001.xml"
        create_demo_stf_file(
            stf_0001,
            study_id="study001",
            operation="append",
            modified_file="../0000/m5/stf-study001.xml",
            leaf_ids=["leaf-003", "leaf-004"]
        )
        print(f"\n创建序列0001: {stf_0001}")
        print("  - operation='append' (后续提交)")
        print("  - modified-file='../0000/m5/stf-study001.xml'")
        print("  - leaf引用: leaf-003, leaf-004 (无重复)")

        # 验证
        print("\n执行验证...")
        violations = validate_stf_lifecycle_pair(str(seq_0001), str(seq_0000))

        print_violations(violations)


def demo_scenario_2_operation_violation():
    """场景2: 操作类型违规"""
    print_section_header("场景2: 操作类型违规 (后续提交错误使用'new')")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        # 创建序列0000
        seq_0000 = temp_dir / "0000"
        stf_0000 = seq_0000 / "m5" / "stf-study002.xml"
        create_demo_stf_file(
            stf_0000,
            study_id="study002",
            operation="new",
            leaf_ids=["leaf-001"]
        )
        print(f"\n创建序列0000: operation='new'")

        # 创建序列0001 (错误：应该用append)
        seq_0001 = temp_dir / "0001"
        stf_0001 = seq_0001 / "m5" / "stf-study002.xml"
        create_demo_stf_file(
            stf_0001,
            study_id="study002",
            operation="new",  # ❌ 错误：应该是append
            leaf_ids=["leaf-002"]
        )
        print(f"创建序列0001: operation='new' ❌ (应该是'append')")

        # 验证
        print("\n执行验证...")
        violations = validate_stf_lifecycle_pair(str(seq_0001), str(seq_0000))

        print_violations(violations)


def demo_scenario_3_modified_file_violation():
    """场景3: modified-file引用违规"""
    print_section_header("场景3: modified-file引用违规")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        # 创建序列0000
        seq_0000 = temp_dir / "0000"
        stf_0000 = seq_0000 / "m5" / "stf-study003.xml"
        create_demo_stf_file(
            stf_0000,
            study_id="study003",
            operation="new",
            leaf_ids=["leaf-001"]
        )

        # 创建序列0001 (缺少modified-file)
        seq_0001 = temp_dir / "0001"
        stf_0001 = seq_0001 / "m5" / "stf-study003.xml"
        create_demo_stf_file(
            stf_0001,
            study_id="study003",
            operation="append",
            # ❌ 缺少modified_file参数
            leaf_ids=["leaf-002"]
        )
        print(f"\n创建序列0001:")
        print("  - operation='append'")
        print("  - ❌ 缺少modified-file属性")

        # 验证
        print("\n执行验证...")
        violations = validate_stf_lifecycle_pair(str(seq_0001), str(seq_0000))

        print_violations(violations)


def demo_scenario_4_study_id_violation():
    """场景4: study-identifier不一致违规"""
    print_section_header("场景4: study-identifier不一致违规")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        # 创建序列0000
        seq_0000 = temp_dir / "0000"
        stf_0000 = seq_0000 / "m5" / "stf-study004.xml"
        create_demo_stf_file(
            stf_0000,
            study_id="study004",
            operation="new",
            leaf_ids=["leaf-001"]
        )
        print(f"\n创建序列0000: study-id='study004'")

        # 创建序列0001 (study-id变化了)
        seq_0001 = temp_dir / "0001"
        stf_0001 = seq_0001 / "m5" / "stf-study004-modified.xml"
        create_demo_stf_file(
            stf_0001,
            study_id="study004-modified",  # ❌ 错误：study-id变化了
            operation="append",
            modified_file="../0000/m5/stf-study004.xml",
            leaf_ids=["leaf-002"]
        )
        print(f"创建序列0001: study-id='study004-modified' ❌ (不应变化)")

        # 验证
        print("\n执行验证...")
        violations = validate_stf_lifecycle_pair(str(seq_0001), str(seq_0000))

        print_violations(violations)


def demo_scenario_5_duplicate_leaves():
    """场景5: 重复leaf引用警告"""
    print_section_header("场景5: 重复leaf引用警告 (累积方式验证)")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        # 创建序列0000
        seq_0000 = temp_dir / "0000"
        stf_0000 = seq_0000 / "m5" / "stf-study005.xml"
        create_demo_stf_file(
            stf_0000,
            study_id="study005",
            operation="new",
            leaf_ids=["leaf-001", "leaf-002"]
        )
        print(f"\n创建序列0000: leaf引用 = ['leaf-001', 'leaf-002']")

        # 创建序列0001 (包含重复的leaf)
        seq_0001 = temp_dir / "0001"
        stf_0001 = seq_0001 / "m5" / "stf-study005.xml"
        create_demo_stf_file(
            stf_0001,
            study_id="study005",
            operation="append",
            modified_file="../0000/m5/stf-study005.xml",
            leaf_ids=["leaf-002", "leaf-003"]  # ⚠️ leaf-002重复了
        )
        print(f"创建序列0001: leaf引用 = ['leaf-002', 'leaf-003']")
        print("  ⚠️ leaf-002在前序列已存在（违反累积方式）")

        # 验证
        print("\n执行验证...")
        violations = validate_stf_lifecycle_pair(str(seq_0001), str(seq_0000))

        print_violations(violations)


def demo_scenario_6_batch_validation():
    """场景6: 批量验证多个序列"""
    print_section_header("场景6: 批量验证整个申请的所有序列")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        # 创建多个序列 (0000-0003)
        for seq_num in range(4):
            seq_id = f"{seq_num:04d}"
            seq_dir = temp_dir / seq_id
            stf_path = seq_dir / "m5" / "stf-study006.xml"

            if seq_num == 0:
                create_demo_stf_file(
                    stf_path,
                    study_id="study006",
                    operation="new",
                    leaf_ids=[f"leaf-{seq_num:03d}"]
                )
                print(f"\n创建序列{seq_id}: operation='new'")
            else:
                prev_seq = f"{seq_num - 1:04d}"
                create_demo_stf_file(
                    stf_path,
                    study_id="study006",
                    operation="append",
                    modified_file=f"../{prev_seq}/m5/stf-study006.xml",
                    leaf_ids=[f"leaf-{seq_num:03d}"]
                )
                print(f"创建序列{seq_id}: operation='append', modified-file='../{prev_seq}/...'")

        # 批量验证
        print("\n执行批量验证...")
        results = validate_stf_lifecycle_application(str(temp_dir))

        if not results:
            print("\n✅ 所有序列对均合规，未发现违规")
        else:
            print(f"\n发现 {len(results)} 个序列对存在违规:")
            for seq_pair, violations in results.items():
                print(f"\n  序列对: {seq_pair}")
                print_violations(violations)


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print(" eCTD STF生命周期管理验证器 - 演示脚本")
    print(" Phase 2.10 - STF Lifecycle Management Validation")
    print("=" * 80)

    print("\n本演示展示STF跨序列验证的主要功能:")
    print("  1. 操作类型验证 (new vs append)")
    print("  2. modified-file引用验证")
    print("  3. study-identifier一致性验证")
    print("  4. 累积方式验证 (无重复leaf)")
    print("  5. 批量序列验证")

    try:
        demo_scenario_1_compliant()
        demo_scenario_2_operation_violation()
        demo_scenario_3_modified_file_violation()
        demo_scenario_4_study_id_violation()
        demo_scenario_5_duplicate_leaves()
        demo_scenario_6_batch_validation()

        print("\n" + "=" * 80)
        print(" 演示完成")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
