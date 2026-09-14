"""
eCTD STF生命周期管理验证器

根据ICH STF Specification V2.6.1规范，验证STF文件在跨序列提交时的生命周期管理规则。

验证内容:
1. STF累积方式验证 (Accumulative Approach)
2. STF操作类型验证 (operation="new" vs "append")
3. Study-identifier一致性验证 (跨序列study-id不变)
4. Modified-file引用验证 (引用最近一次提交的STF)
5. STF删除操作验证 (不应使用operation="delete")

Stage: Phase 2.10
创建日期: 2026-09-14
"""

import re
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


class ViolationSeverity(Enum):
    """违规严重程度"""
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ViolationDetail:
    """违规详情"""
    rule_id: str
    severity: ViolationSeverity
    message: str
    location: str
    details: str
    suggestion: str


@dataclass
class STFLifecycleSnapshot:
    """STF生命周期快照"""
    sequence_number: str
    stf_file_path: str
    study_id: Optional[str] = None
    operation: Optional[str] = None  # "new" or "append"
    modified_file: Optional[str] = None  # 引用的前一个STF路径
    leaf_ids: Set[str] = field(default_factory=set)  # 该STF引用的leaf ID集合

    def __post_init__(self):
        if isinstance(self.leaf_ids, list):
            self.leaf_ids = set(self.leaf_ids)


@dataclass
class STFLifecycleHistory:
    """STF生命周期历史记录"""
    application_path: str  # 申请文件夹路径
    snapshots: List[STFLifecycleSnapshot] = field(default_factory=list)

    def add_snapshot(self, snapshot: STFLifecycleSnapshot):
        """添加快照"""
        self.snapshots.append(snapshot)

    def get_snapshot_by_sequence(self, sequence_number: str) -> Optional[STFLifecycleSnapshot]:
        """根据序列号获取快照"""
        for snapshot in self.snapshots:
            if snapshot.sequence_number == sequence_number:
                return snapshot
        return None

    def get_previous_snapshot(self, sequence_number: str) -> Optional[STFLifecycleSnapshot]:
        """获取指定序列的前一个快照"""
        try:
            current_seq_num = int(sequence_number)
        except ValueError:
            return None

        # 找到序列号小于当前序列的最大序列号
        previous_snapshots = [
            s for s in self.snapshots
            if s.sequence_number.isdigit() and int(s.sequence_number) < current_seq_num
        ]

        if not previous_snapshots:
            return None

        # 返回序列号最大的那个
        return max(previous_snapshots, key=lambda s: int(s.sequence_number))


class STFLifecycleValidator:
    """STF生命周期管理验证器"""

    # STF文件名模式: stf-{study-id}.xml
    STF_FILENAME_PATTERN = re.compile(r'^stf-([a-zA-Z0-9_-]+)\.xml$', re.IGNORECASE)

    def __init__(self):
        self.history = STFLifecycleHistory(application_path="")

    def extract_stf_snapshot(
        self,
        sequence_path: str,
        stf_file_path: str
    ) -> Optional[STFLifecycleSnapshot]:
        """
        从STF文件中提取生命周期快照

        参数:
            sequence_path: 序列文件夹路径 (如 /path/to/0005)
            stf_file_path: STF文件路径

        返回:
            STFLifecycleSnapshot对象，如果解析失败则返回None
        """
        sequence_number = Path(sequence_path).name

        try:
            # 解析XML
            tree = ET.parse(stf_file_path)
            root = tree.getroot()

            # 提取study-id (从文件名)
            filename = Path(stf_file_path).name
            match = self.STF_FILENAME_PATTERN.match(filename)
            study_id = match.group(1) if match else None

            # 提取operation属性
            # 根元素是 <ectd:study> 或 <study>
            operation = None
            for elem in root.iter():
                if 'operation' in elem.attrib:
                    operation = elem.attrib['operation']
                    break

            # 提取modified-file属性
            modified_file = root.attrib.get('modified-file')
            if not modified_file:
                # 也可能在子元素中
                for elem in root.iter():
                    if 'modified-file' in elem.attrib:
                        modified_file = elem.attrib['modified-file']
                        break

            # 提取引用的leaf IDs
            leaf_ids = set()
            for elem in root.iter():
                # 查找所有带 xlink:href 的元素
                for attr_name, attr_value in elem.attrib.items():
                    if 'href' in attr_name.lower():
                        # 从href中提取leaf ID (假设格式为 #leaf-id)
                        if attr_value.startswith('#'):
                            leaf_ids.add(attr_value[1:])

            return STFLifecycleSnapshot(
                sequence_number=sequence_number,
                stf_file_path=stf_file_path,
                study_id=study_id,
                operation=operation,
                modified_file=modified_file,
                leaf_ids=leaf_ids
            )

        except Exception as e:
            print(f"Error parsing STF file {stf_file_path}: {e}")
            return None

    def validate_stf_operation_type(
        self,
        current_snapshot: STFLifecycleSnapshot,
        previous_snapshot: Optional[STFLifecycleSnapshot]
    ) -> List[ViolationDetail]:
        """
        验证STF操作类型

        规则:
        - 首次提交的STF: operation必须为"new"
        - 后续提交的STF: operation必须为"append"
        - 不应使用operation="delete"或"replace"

        参数:
            current_snapshot: 当前序列的STF快照
            previous_snapshot: 前一个序列的STF快照（如果存在）

        返回:
            违规详情列表
        """
        violations = []

        # 检查operation属性是否存在
        if not current_snapshot.operation:
            violations.append(ViolationDetail(
                rule_id="STF-LC-001",
                severity=ViolationSeverity.ERROR,
                message="STF文件缺少operation属性",
                location=current_snapshot.stf_file_path,
                details=f"序列{current_snapshot.sequence_number}的STF文件未指定operation属性",
                suggestion="在STF根元素上添加operation属性: operation=\"new\"（首次）或operation=\"append\"（后续）"
            ))
            return violations

        operation = current_snapshot.operation.lower()

        # 规则1: 首次提交必须是"new"
        if previous_snapshot is None:
            if operation != "new":
                violations.append(ViolationDetail(
                    rule_id="STF-LC-002",
                    severity=ViolationSeverity.ERROR,
                    message=f"首次提交的STF operation应为'new'，实际为'{current_snapshot.operation}'",
                    location=current_snapshot.stf_file_path,
                    details=(
                        f"序列{current_snapshot.sequence_number}是首次提交STF，"
                        f"但operation属性为'{current_snapshot.operation}'而非'new'"
                    ),
                    suggestion="将operation属性改为'new': <ectd:study operation=\"new\" ...>"
                ))

        # 规则2: 后续提交必须是"append"
        else:
            if operation != "append":
                violations.append(ViolationDetail(
                    rule_id="STF-LC-003",
                    severity=ViolationSeverity.ERROR,
                    message=f"后续提交的STF operation应为'append'，实际为'{current_snapshot.operation}'",
                    location=current_snapshot.stf_file_path,
                    details=(
                        f"序列{current_snapshot.sequence_number}是后续提交STF "
                        f"(前序列{previous_snapshot.sequence_number}已有STF)，"
                        f"但operation属性为'{current_snapshot.operation}'而非'append'"
                    ),
                    suggestion="将operation属性改为'append': <ectd:study operation=\"append\" ...>"
                ))

        # 规则3: 不应使用delete或replace
        if operation in ("delete", "replace"):
            violations.append(ViolationDetail(
                rule_id="STF-LC-004",
                severity=ViolationSeverity.WARNING,
                message=f"STF operation不应使用'{current_snapshot.operation}'",
                location=current_snapshot.stf_file_path,
                details=(
                    f"STF生命周期管理应使用累积方式，不建议使用operation=\"{current_snapshot.operation}\"。"
                    "文件删除应在index.xml中操作，而非提交新的STF。"
                ),
                suggestion=(
                    "如需删除文件：在index.xml中将leaf的operation设为'delete'，不提交新STF。"
                    "如需修改文件标签：提交operation='append'的新STF，引用新的leaf。"
                )
            ))

        return violations

    def validate_modified_file_reference(
        self,
        current_snapshot: STFLifecycleSnapshot,
        previous_snapshot: Optional[STFLifecycleSnapshot]
    ) -> List[ViolationDetail]:
        """
        验证modified-file引用

        规则:
        - 后续提交的STF (operation="append") 必须有modified-file属性
        - modified-file必须引用前一个序列的STF文件
        - 引用路径应使用相对路径

        参数:
            current_snapshot: 当前序列的STF快照
            previous_snapshot: 前一个序列的STF快照

        返回:
            违规详情列表
        """
        violations = []

        # 只有当operation="append"时才需要验证
        if not current_snapshot.operation or current_snapshot.operation.lower() != "append":
            return violations

        # 规则1: append操作必须有modified-file属性
        if not current_snapshot.modified_file:
            violations.append(ViolationDetail(
                rule_id="STF-LC-005",
                severity=ViolationSeverity.ERROR,
                message="后续提交的STF缺少modified-file属性",
                location=current_snapshot.stf_file_path,
                details=(
                    f"序列{current_snapshot.sequence_number}的STF使用operation='append'，"
                    "但未指定modified-file属性引用前一个STF"
                ),
                suggestion=(
                    f"添加modified-file属性，引用前序列的STF: "
                    f"modified-file=\"../{previous_snapshot.sequence_number if previous_snapshot else 'XXXX'}/{Path(previous_snapshot.stf_file_path).name if previous_snapshot else 'stf-xxx.xml'}\""
                )
            ))
            return violations

        # 规则2: 验证引用的STF是否存在
        if previous_snapshot is None:
            violations.append(ViolationDetail(
                rule_id="STF-LC-006",
                severity=ViolationSeverity.ERROR,
                message="modified-file引用了不存在的前序列STF",
                location=current_snapshot.stf_file_path,
                details=(
                    f"序列{current_snapshot.sequence_number}的STF引用了'{current_snapshot.modified_file}'，"
                    "但未找到对应的前序列STF"
                ),
                suggestion="检查modified-file路径是否正确，或确保前序列包含STF文件"
            ))
        else:
            # 规则3: 验证引用的是最近的前序列STF
            # 从路径中提取序列号进行比较
            import re

            # 从previous_snapshot路径提取序列号
            prev_path_match = re.search(r'[\\/](\d{4})[\\/]', previous_snapshot.stf_file_path)
            expected_seq = prev_path_match.group(1) if prev_path_match else None

            # 从modified-file引用中提取序列号
            modified_match = re.search(r'[\\/](\d{4})[\\/]', current_snapshot.modified_file)
            actual_seq = modified_match.group(1) if modified_match else None

            # 如果能提取到序列号，则比较序列号；否则回退到文件名比较
            if expected_seq and actual_seq:
                if actual_seq != expected_seq:
                    violations.append(ViolationDetail(
                        rule_id="STF-LC-007",
                        severity=ViolationSeverity.WARNING,
                        message="modified-file可能未引用最近一次提交的STF",
                        location=current_snapshot.stf_file_path,
                        details=(
                            f"序列{current_snapshot.sequence_number}的modified-file引用了序列'{actual_seq}'的STF，"
                            f"但前序列是序列'{expected_seq}'"
                        ),
                        suggestion=f"建议修改modified-file为: '../{expected_seq}/{Path(previous_snapshot.stf_file_path).name}'"
                    ))
            else:
                # 回退到文件名比较
                expected_reference = Path(previous_snapshot.stf_file_path).name
                actual_reference = Path(current_snapshot.modified_file).name

                if actual_reference != expected_reference:
                    violations.append(ViolationDetail(
                        rule_id="STF-LC-007",
                        severity=ViolationSeverity.WARNING,
                        message="modified-file可能未引用最近一次提交的STF",
                        location=current_snapshot.stf_file_path,
                        details=(
                            f"序列{current_snapshot.sequence_number}的modified-file引用了'{actual_reference}'，"
                            f"但前序列{previous_snapshot.sequence_number}的STF文件是'{expected_reference}'"
                        ),
                        suggestion=f"建议修改modified-file为: '../{previous_snapshot.sequence_number}/{expected_reference}'"
                    ))

        return violations

    def validate_study_identifier_consistency(
        self,
        current_snapshot: STFLifecycleSnapshot,
        previous_snapshot: Optional[STFLifecycleSnapshot]
    ) -> List[ViolationDetail]:
        """
        验证study-identifier一致性

        规则:
        - 同一个研究的STF，其study-id应保持一致
        - 如果study-id发生变化，应有明确说明

        参数:
            current_snapshot: 当前序列的STF快照
            previous_snapshot: 前一个序列的STF快照

        返回:
            违规详情列表
        """
        violations = []

        # 只有当前后序列都有study-id时才验证
        if not current_snapshot.study_id or not previous_snapshot or not previous_snapshot.study_id:
            return violations

        # 规则: study-id应保持一致
        if current_snapshot.study_id != previous_snapshot.study_id:
            violations.append(ViolationDetail(
                rule_id="STF-LC-008",
                severity=ViolationSeverity.ERROR,
                message="study-identifier在不同序列间发生了变化",
                location=current_snapshot.stf_file_path,
                details=(
                    f"前序列{previous_snapshot.sequence_number}的study-id为'{previous_snapshot.study_id}'，"
                    f"但当前序列{current_snapshot.sequence_number}的study-id变为'{current_snapshot.study_id}'"
                ),
                suggestion=(
                    "同一个研究的STF文件应使用相同的study-id。"
                    "如果确实需要修改study-id，应提交新的STF (operation='append') 包含完整的study-identifier块，"
                    "并在提交文档中说明变更原因。"
                )
            ))

        return violations

    def validate_cumulative_approach(
        self,
        current_snapshot: STFLifecycleSnapshot,
        previous_snapshot: Optional[STFLifecycleSnapshot]
    ) -> List[ViolationDetail]:
        """
        验证累积方式 (Accumulative Approach)

        规则:
        - 后续STF应只包含新增或修改的leaf引用
        - 不应重复前序列已有的完整leaf列表

        参数:
            current_snapshot: 当前序列的STF快照
            previous_snapshot: 前一个序列的STF快照

        返回:
            违规详情列表
        """
        violations = []

        # 只有后续提交才需要验证累积方式
        if not previous_snapshot:
            return violations

        if not current_snapshot.operation or current_snapshot.operation.lower() != "append":
            return violations

        # 检查是否有重复的leaf引用
        if current_snapshot.leaf_ids and previous_snapshot.leaf_ids:
            duplicated_leaves = current_snapshot.leaf_ids & previous_snapshot.leaf_ids

            if duplicated_leaves:
                violations.append(ViolationDetail(
                    rule_id="STF-LC-009",
                    severity=ViolationSeverity.WARNING,
                    message="STF包含前序列已存在的leaf引用",
                    location=current_snapshot.stf_file_path,
                    details=(
                        f"序列{current_snapshot.sequence_number}的STF引用了 {len(duplicated_leaves)} 个"
                        f"在前序列{previous_snapshot.sequence_number}中已存在的leaf: "
                        f"{', '.join(list(duplicated_leaves)[:5])}{'...' if len(duplicated_leaves) > 5 else ''}"
                    ),
                    suggestion=(
                        "根据累积方式 (Accumulative Approach)，后续STF应只包含新增或修改的leaf引用，"
                        "无需重复前序列已有的leaf。请移除重复的leaf引用。"
                    )
                ))

        return violations

    def validate_sequence_pair(
        self,
        current_sequence_path: str,
        previous_sequence_path: Optional[str] = None
    ) -> List[ViolationDetail]:
        """
        验证一对序列的STF生命周期管理

        参数:
            current_sequence_path: 当前序列路径
            previous_sequence_path: 前序列路径（如果是首次提交则为None）

        返回:
            违规详情列表
        """
        violations = []

        # 查找当前序列的STF文件
        current_stf_files = list(Path(current_sequence_path).rglob("stf-*.xml"))
        if not current_stf_files:
            # 没有STF不算违规（可能该序列没有研究报告更新）
            return violations

        # 暂时只处理第一个STF文件
        current_stf_path = str(current_stf_files[0])
        current_snapshot = self.extract_stf_snapshot(current_sequence_path, current_stf_path)

        if not current_snapshot:
            violations.append(ViolationDetail(
                rule_id="STF-LC-010",
                severity=ViolationSeverity.ERROR,
                message="无法解析当前序列的STF文件",
                location=current_stf_path,
                details="STF文件格式错误或不是有效的XML",
                suggestion="检查STF文件的XML格式是否正确"
            ))
            return violations

        # 查找前序列的STF文件
        previous_snapshot = None
        if previous_sequence_path:
            previous_stf_files = list(Path(previous_sequence_path).rglob("stf-*.xml"))
            if previous_stf_files:
                previous_stf_path = str(previous_stf_files[0])
                previous_snapshot = self.extract_stf_snapshot(previous_sequence_path, previous_stf_path)

        # 执行各项验证
        violations.extend(self.validate_stf_operation_type(current_snapshot, previous_snapshot))
        violations.extend(self.validate_modified_file_reference(current_snapshot, previous_snapshot))
        violations.extend(self.validate_study_identifier_consistency(current_snapshot, previous_snapshot))
        violations.extend(self.validate_cumulative_approach(current_snapshot, previous_snapshot))

        return violations

    def validate_application_sequences(
        self,
        application_path: str
    ) -> Dict[str, List[ViolationDetail]]:
        """
        验证整个申请文件夹下所有序列的STF生命周期管理

        参数:
            application_path: 申请文件夹路径（包含0000, 0001, ... 序列文件夹）

        返回:
            按序列号组织的违规详情字典 {sequence_number: [violations]}
        """
        self.history = STFLifecycleHistory(application_path=application_path)
        results = {}

        # 查找所有序列文件夹
        app_path = Path(application_path)
        sequence_dirs = sorted([d for d in app_path.iterdir() if d.is_dir() and d.name.isdigit()])

        if not sequence_dirs:
            return results

        # 逐个序列验证
        for i, seq_dir in enumerate(sequence_dirs):
            sequence_number = seq_dir.name
            current_path = str(seq_dir)

            # 确定前序列路径
            previous_path = None
            if i > 0:
                previous_path = str(sequence_dirs[i - 1])

            # 验证该序列对
            violations = self.validate_sequence_pair(current_path, previous_path)

            if violations:
                results[sequence_number] = violations

        return results


def validate_stf_lifecycle_pair(
    current_sequence_path: str,
    previous_sequence_path: Optional[str] = None
) -> List[ViolationDetail]:
    """
    便捷函数：验证一对序列的STF生命周期管理

    参数:
        current_sequence_path: 当前序列路径
        previous_sequence_path: 前序列路径（如果是首次提交则为None）

    返回:
        违规详情列表

    示例:
        violations = validate_stf_lifecycle_pair("/path/to/0005", "/path/to/0004")
        for v in violations:
            print(f"{v.severity.value}: {v.message}")
    """
    validator = STFLifecycleValidator()
    return validator.validate_sequence_pair(current_sequence_path, previous_sequence_path)


def validate_stf_lifecycle_application(
    application_path: str
) -> Dict[str, List[ViolationDetail]]:
    """
    便捷函数：验证整个申请的STF生命周期管理

    参数:
        application_path: 申请文件夹路径（包含0000, 0001, ... 序列文件夹）

    返回:
        按序列号组织的违规详情字典

    示例:
        results = validate_stf_lifecycle_application("/path/to/application")
        for seq, violations in results.items():
            print(f"Sequence {seq}: {len(violations)} violations")
    """
    validator = STFLifecycleValidator()
    return validator.validate_application_sequences(application_path)
