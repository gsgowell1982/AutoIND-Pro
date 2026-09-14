"""
eCTD序列关系解析模块

负责：
1. 解析前序列路径
2. 提取序列元数据
3. 验证序列路径有效性

版本: v1.0
创建日期: 2026-09-11
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Literal
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class SequenceResolverError(Exception):
    """序列解析相关错误"""
    pass


def resolve_previous_sequence(
    current_sequence_path: str,
    strategy: Literal["auto", "explicit", "user_provided"] = "auto",
    explicit_path: Optional[str] = None,
    user_provided_mapping: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """
    解析前序列路径

    Args:
        current_sequence_path: 当前序列的路径
        strategy: 解析策略
            - "auto": 从index.xml提取 + 同目录搜索
            - "explicit": 使用explicit_path参数
            - "user_provided": 使用user_provided_mapping
        explicit_path: 显式指定的前序列路径（strategy="explicit"时使用）
        user_provided_mapping: 用户提供的序列映射 {序列号: 路径}

    Returns:
        前序列路径，或None（初始序列）

    Raises:
        SequenceResolverError: 解析失败

    Examples:
        >>> # 自动解析
        >>> prev = resolve_previous_sequence("/data/ectd/0005", strategy="auto")
        >>> # 结果: "/data/ectd/0004" 或 None

        >>> # 显式指定
        >>> prev = resolve_previous_sequence(
        ...     "/data/ectd/0005",
        ...     strategy="explicit",
        ...     explicit_path="/data/ectd/0004"
        ... )

        >>> # 用户提供映射
        >>> prev = resolve_previous_sequence(
        ...     "/data/ectd/0005",
        ...     strategy="user_provided",
        ...     user_provided_mapping={"0004": "/archive/ectd_0004"}
        ... )
    """
    logger.info(f"Resolving previous sequence for: {current_sequence_path}, strategy: {strategy}")

    # 策略1: 显式路径
    if strategy == "explicit":
        if not explicit_path:
            raise SequenceResolverError("explicit_path is required when strategy='explicit'")

        if not _validate_sequence_path(explicit_path):
            raise SequenceResolverError(f"Invalid explicit previous sequence path: {explicit_path}")

        logger.info(f"Using explicit previous sequence: {explicit_path}")
        return explicit_path

    # 验证当前序列路径
    if not _validate_sequence_path(current_sequence_path):
        raise SequenceResolverError(f"Invalid current sequence path: {current_sequence_path}")

    # 提取当前序列的元数据
    try:
        current_metadata = extract_sequence_metadata(current_sequence_path)
    except Exception as e:
        raise SequenceResolverError(f"Failed to extract metadata from current sequence: {e}")

    current_seq_num = current_metadata.get("sequence_number")
    if not current_seq_num:
        raise SequenceResolverError("Cannot extract sequence_number from current sequence")

    # 检查是否是初始序列
    if current_seq_num in ("0000", "0001"):
        logger.info(f"Sequence {current_seq_num} is initial sequence, no previous sequence")
        return None

    # 计算前序列号
    try:
        prev_seq_num = f"{int(current_seq_num) - 1:04d}"
    except ValueError:
        raise SequenceResolverError(f"Invalid sequence number format: {current_seq_num}")

    # 策略2: 用户提供映射
    if strategy == "user_provided":
        if not user_provided_mapping:
            raise SequenceResolverError("user_provided_mapping is required when strategy='user_provided'")

        if prev_seq_num in user_provided_mapping:
            prev_path = user_provided_mapping[prev_seq_num]
            if _validate_sequence_path(prev_path):
                logger.info(f"Using user-provided previous sequence: {prev_path}")
                return prev_path
            else:
                raise SequenceResolverError(f"User-provided path is invalid: {prev_path}")
        else:
            logger.warning(f"Previous sequence {prev_seq_num} not found in user-provided mapping")
            return None

    # 策略3: 自动搜索
    if strategy == "auto":
        parent_dir = Path(current_sequence_path).parent

        # 搜索策略1: 直接子目录名匹配
        candidate_path = parent_dir / prev_seq_num
        if candidate_path.exists() and _validate_sequence_path(str(candidate_path)):
            logger.info(f"Found previous sequence by direct name match: {candidate_path}")
            return str(candidate_path)

        # 搜索策略2: 遍历同目录下所有子目录，查找序列号匹配
        logger.info(f"Direct name match failed, searching in parent directory: {parent_dir}")
        for entry in parent_dir.iterdir():
            if entry.is_dir() and entry != Path(current_sequence_path):
                try:
                    entry_metadata = extract_sequence_metadata(str(entry))
                    if entry_metadata.get("sequence_number") == prev_seq_num:
                        logger.info(f"Found previous sequence by metadata match: {entry}")
                        return str(entry)
                except Exception as e:
                    # 跳过无法解析的目录
                    logger.debug(f"Skip directory {entry}: {e}")
                    continue

        # 未找到前序列
        logger.warning(f"Previous sequence {prev_seq_num} not found for current sequence {current_seq_num}")
        return None

    raise SequenceResolverError(f"Unknown strategy: {strategy}")


def extract_sequence_metadata(sequence_path: str) -> Dict[str, Any]:
    """
    从eCTD序列目录提取元数据

    Args:
        sequence_path: 序列目录路径（包含index.xml的目录）

    Returns:
        元数据字典，包含：
        {
            "sequence_number": "0005",          # 必需
            "sequence_type": "type-2-variation", # 可选
            "submission_date": "2026-09-01",    # 可选
            "dtd_version": "3.2",               # 可选
        }

    Raises:
        SequenceResolverError: 解析失败

    Notes:
        - 优先从index.xml的根元素或envelope提取
        - 如果有cn-regional.xml，也会尝试提取
        - 当前实现基于ICH eCTD DTD 3.2规范
    """
    sequence_path_obj = Path(sequence_path)
    index_xml_path = sequence_path_obj / "index.xml"

    if not index_xml_path.exists():
        raise SequenceResolverError(f"index.xml not found in: {sequence_path}")

    try:
        tree = ET.parse(str(index_xml_path))
        root = tree.getroot()
    except ET.ParseError as e:
        raise SequenceResolverError(f"Failed to parse index.xml: {e}")

    metadata = {}

    # 提取DTD版本（从根元素）
    dtd_version = root.get("dtd-version")
    if dtd_version:
        metadata["dtd_version"] = dtd_version

    # 尝试从目录名推断序列号（作为fallback）
    # 常见命名模式：0005, seq-0005, sequence-0005, old_submission等
    # 优先使用此方法，因为它最可靠
    dir_name = sequence_path_obj.name
    for potential_seq_num in dir_name.split("-"):
        if potential_seq_num.isdigit() and len(potential_seq_num) == 4:
            metadata["sequence_number"] = potential_seq_num
            logger.debug(f"Extracted sequence_number from directory name: {potential_seq_num}")
            break

    # TODO: 尝试从cn-regional.xml或envelope元素提取更精确的元数据
    # 这需要理解具体的XML结构，当前作为占位实现
    # 真实实施时需要根据实际的eCTD结构完善

    # 尝试查找sequence相关元素（可能在不同命名空间下）
    # 这是一个通用搜索，实际结构可能需要调整
    for elem in root.iter():
        tag_local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

        if tag_local in ("sequence-number", "sequence", "sequenceNumber"):
            if elem.text and elem.text.strip():
                metadata["sequence_number"] = elem.text.strip()

        if tag_local in ("sequence-type", "sequenceType"):
            if elem.text and elem.text.strip():
                metadata["sequence_type"] = elem.text.strip()

        if tag_local in ("submission-date", "submissionDate"):
            if elem.text and elem.text.strip():
                metadata["submission_date"] = elem.text.strip()

    # 验证必需字段
    if "sequence_number" not in metadata:
        raise SequenceResolverError(
            f"Cannot extract sequence_number from index.xml. "
            f"Please check XML structure or use explicit path strategy."
        )

    logger.debug(f"Extracted metadata from {sequence_path}: {metadata}")
    return metadata


def _validate_sequence_path(path: str) -> bool:
    """
    验证路径是否是有效的eCTD序列目录

    Args:
        path: 待验证的路径

    Returns:
        True如果路径有效，False否则

    Notes:
        有效序列目录的标准：
        1. 路径存在且是目录
        2. 包含index.xml文件
        3. index.xml可以被解析（格式正确）
    """
    path_obj = Path(path)

    # 检查路径存在且是目录
    if not path_obj.exists():
        logger.debug(f"Path does not exist: {path}")
        return False

    if not path_obj.is_dir():
        logger.debug(f"Path is not a directory: {path}")
        return False

    # 检查index.xml存在
    index_xml_path = path_obj / "index.xml"
    if not index_xml_path.exists():
        logger.debug(f"index.xml not found in: {path}")
        return False

    # 尝试解析index.xml（验证格式）
    try:
        tree = ET.parse(str(index_xml_path))
        root = tree.getroot()
        # 基本验证：根元素应该是ectd命名空间的元素
        if "ectd" not in root.tag.lower():
            logger.debug(f"index.xml root element is not ectd: {root.tag}")
            return False
    except ET.ParseError as e:
        logger.debug(f"index.xml parse error: {e}")
        return False

    return True


# ============================================================================
# 接口预留：真实样本验证和优化
# ============================================================================

def validate_with_real_sample(
    sample_path: str,
    expected_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    使用真实eCTD样本验证解析能力

    这是一个接口函数，用于在获得真实样本后进行验证和优化。

    Args:
        sample_path: 真实eCTD样本路径
        expected_metadata: 预期的元数据（用于验证准确性）

    Returns:
        验证报告，包含：
        {
            "success": bool,
            "extracted_metadata": {...},
            "issues": [list of issues found],
            "suggestions": [list of optimization suggestions]
        }

    Usage:
        当获得真实样本后：
        1. 运行此函数验证当前实现
        2. 根据报告中的issues调整extract_sequence_metadata()
        3. 根据suggestions优化解析逻辑

    Example:
        >>> report = validate_with_real_sample(
        ...     "/real_samples/product_a/0005",
        ...     expected_metadata={"sequence_number": "0005"}
        ... )
        >>> if not report["success"]:
        ...     print("Issues found:", report["issues"])
    """
    report = {
        "success": False,
        "extracted_metadata": {},
        "issues": [],
        "suggestions": []
    }

    # 验证路径
    if not _validate_sequence_path(sample_path):
        report["issues"].append("Invalid sequence path")
        return report

    # 提取元数据
    try:
        metadata = extract_sequence_metadata(sample_path)
        report["extracted_metadata"] = metadata
    except Exception as e:
        report["issues"].append(f"Metadata extraction failed: {e}")
        return report

    # 如果提供了预期值，进行对比
    if expected_metadata:
        for key, expected_value in expected_metadata.items():
            extracted_value = metadata.get(key)
            if extracted_value != expected_value:
                report["issues"].append(
                    f"Metadata mismatch for '{key}': "
                    f"expected='{expected_value}', extracted='{extracted_value}'"
                )

    # 检查前序列解析
    try:
        prev_path = resolve_previous_sequence(sample_path, strategy="auto")
        if prev_path:
            if not _validate_sequence_path(prev_path):
                report["issues"].append(f"Resolved previous sequence is invalid: {prev_path}")
        # 如果是初始序列，prev_path为None是正常的
    except Exception as e:
        report["issues"].append(f"Previous sequence resolution failed: {e}")

    # 生成优化建议
    if "sequence_type" not in metadata:
        report["suggestions"].append(
            "sequence_type not extracted. Consider adding extraction logic if available in XML."
        )

    if "submission_date" not in metadata:
        report["suggestions"].append(
            "submission_date not extracted. Consider adding extraction logic if available in XML."
        )

    # 判定成功
    report["success"] = len(report["issues"]) == 0

    return report


def optimize_with_real_samples(sample_paths: list[str]) -> Dict[str, Any]:
    """
    批量使用真实样本优化解析器

    Args:
        sample_paths: 真实样本路径列表

    Returns:
        优化报告

    Usage:
        当积累了多个真实样本后，运行此函数：
        1. 识别常见的XML结构模式
        2. 发现边界情况
        3. 生成优化建议
    """
    report = {
        "total_samples": len(sample_paths),
        "successful": 0,
        "failed": 0,
        "common_issues": {},
        "xml_structures_observed": [],
        "optimization_recommendations": []
    }

    for sample_path in sample_paths:
        validation = validate_with_real_sample(sample_path)

        if validation["success"]:
            report["successful"] += 1
        else:
            report["failed"] += 1
            # 统计常见问题
            for issue in validation["issues"]:
                if issue not in report["common_issues"]:
                    report["common_issues"][issue] = 0
                report["common_issues"][issue] += 1

    # 生成优化建议
    if report["failed"] > 0:
        report["optimization_recommendations"].append(
            f"{report['failed']}/{report['total_samples']} samples failed validation. "
            "Review common_issues for patterns."
        )

    if report["common_issues"]:
        most_common = max(report["common_issues"].items(), key=lambda x: x[1])
        report["optimization_recommendations"].append(
            f"Most common issue: '{most_common[0]}' ({most_common[1]} occurrences). "
            "Prioritize fixing this."
        )

    return report
