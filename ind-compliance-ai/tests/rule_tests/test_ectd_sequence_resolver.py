"""
eCTD序列解析器单元测试

测试 core/ectd_sequence_resolver.py 的功能

版本: v1.0
创建日期: 2026-09-11
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.ectd_sequence_resolver import (
    resolve_previous_sequence,
    extract_sequence_metadata,
    _validate_sequence_path,
    SequenceResolverError
)
from tests.test_data_generators.ectd_metadata_test_generator import (
    generate_mock_index_xml,
    MockSequence,
    MockSection,
    MockLeaf
)


class TestSequenceResolver(unittest.TestCase):
    """测试序列解析器"""

    def setUp(self):
        """创建临时测试目录"""
        self.test_dir = tempfile.mkdtemp(prefix="ectd_test_")
        self.test_path = Path(self.test_dir)

    def tearDown(self):
        """清理临时目录"""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)

    def _create_mock_sequence(self, seq_num: str) -> str:
        """创建一个mock序列"""
        seq = MockSequence(
            sequence_number=seq_num,
            sections=[
                MockSection(
                    element_name="m2-3-s-drug-substance",
                    attributes={"substance": "API-A", "manufacturer": "MFR-X"},
                    leafs=[
                        MockLeaf("s001", "test.pdf", "new", "m2/23s/test.pdf")
                    ]
                )
            ]
        )
        seq_path = self.test_path / seq_num
        generate_mock_index_xml(seq, str(seq_path))
        return str(seq_path)

    # ========================================================================
    # 测试：_validate_sequence_path
    # ========================================================================

    def test_validate_sequence_path_valid(self):
        """测试：有效的序列路径"""
        seq_path = self._create_mock_sequence("0005")
        self.assertTrue(_validate_sequence_path(seq_path))

    def test_validate_sequence_path_not_exists(self):
        """测试：路径不存在"""
        fake_path = str(self.test_path / "nonexistent")
        self.assertFalse(_validate_sequence_path(fake_path))

    def test_validate_sequence_path_not_directory(self):
        """测试：路径不是目录"""
        file_path = self.test_path / "file.txt"
        file_path.write_text("test")
        self.assertFalse(_validate_sequence_path(str(file_path)))

    def test_validate_sequence_path_no_index_xml(self):
        """测试：缺少index.xml"""
        dir_path = self.test_path / "no_index"
        dir_path.mkdir()
        self.assertFalse(_validate_sequence_path(str(dir_path)))

    # ========================================================================
    # 测试：extract_sequence_metadata
    # ========================================================================

    def test_extract_sequence_metadata_basic(self):
        """测试：提取基本元数据"""
        seq_path = self._create_mock_sequence("0005")
        metadata = extract_sequence_metadata(seq_path)

        self.assertIn("sequence_number", metadata)
        self.assertEqual(metadata["sequence_number"], "0005")
        self.assertIn("dtd_version", metadata)
        self.assertEqual(metadata["dtd_version"], "3.2")

    def test_extract_sequence_metadata_from_dirname(self):
        """测试：从目录名推断序列号"""
        seq = MockSequence(
            sequence_number="0007",
            sections=[]
        )
        # 目录名包含序列号
        seq_path = self.test_path / "sequence-0007"
        generate_mock_index_xml(seq, str(seq_path))

        metadata = extract_sequence_metadata(str(seq_path))
        self.assertEqual(metadata["sequence_number"], "0007")

    def test_extract_sequence_metadata_invalid_xml(self):
        """测试：无效的XML"""
        seq_path = self.test_path / "invalid"
        seq_path.mkdir()
        index_xml = seq_path / "index.xml"
        index_xml.write_text("<invalid>xml</notclosed>")

        with self.assertRaises(SequenceResolverError):
            extract_sequence_metadata(str(seq_path))

    def test_extract_sequence_metadata_no_index_xml(self):
        """测试：缺少index.xml"""
        seq_path = self.test_path / "no_index"
        seq_path.mkdir()

        with self.assertRaises(SequenceResolverError):
            extract_sequence_metadata(str(seq_path))

    # ========================================================================
    # 测试：resolve_previous_sequence - 策略explicit
    # ========================================================================

    def test_resolve_previous_sequence_explicit_valid(self):
        """测试：显式指定有效的前序列路径"""
        current_path = self._create_mock_sequence("0005")
        previous_path = self._create_mock_sequence("0004")

        result = resolve_previous_sequence(
            current_path,
            strategy="explicit",
            explicit_path=previous_path
        )

        self.assertEqual(result, previous_path)

    def test_resolve_previous_sequence_explicit_invalid(self):
        """测试：显式指定无效的前序列路径"""
        current_path = self._create_mock_sequence("0005")
        fake_path = str(self.test_path / "nonexistent")

        with self.assertRaises(SequenceResolverError):
            resolve_previous_sequence(
                current_path,
                strategy="explicit",
                explicit_path=fake_path
            )

    def test_resolve_previous_sequence_explicit_no_path(self):
        """测试：显式策略但未提供路径"""
        current_path = self._create_mock_sequence("0005")

        with self.assertRaises(SequenceResolverError):
            resolve_previous_sequence(
                current_path,
                strategy="explicit",
                explicit_path=None
            )

    # ========================================================================
    # 测试：resolve_previous_sequence - 策略auto
    # ========================================================================

    def test_resolve_previous_sequence_auto_direct_match(self):
        """测试：自动搜索 - 直接目录名匹配"""
        # 创建序列0004和0005在同一父目录下
        self._create_mock_sequence("0004")
        current_path = self._create_mock_sequence("0005")

        result = resolve_previous_sequence(current_path, strategy="auto")

        expected_path = str(self.test_path / "0004")
        self.assertEqual(result, expected_path)

    def test_resolve_previous_sequence_auto_metadata_match(self):
        """测试：自动搜索 - 通过元数据匹配"""
        # 创建目录名不是标准格式但包含序列号的情况
        seq_0004 = MockSequence(sequence_number="0004", sections=[])
        seq_path_0004 = self.test_path / "old-submission-0004"  # 包含序列号
        generate_mock_index_xml(seq_0004, str(seq_path_0004))

        current_path = self._create_mock_sequence("0005")

        result = resolve_previous_sequence(current_path, strategy="auto")

        # 应该能通过元数据找到
        self.assertEqual(result, str(seq_path_0004))

    def test_resolve_previous_sequence_auto_initial_sequence(self):
        """测试：自动搜索 - 初始序列"""
        current_path = self._create_mock_sequence("0000")

        result = resolve_previous_sequence(current_path, strategy="auto")

        # 初始序列应返回None
        self.assertIsNone(result)

    def test_resolve_previous_sequence_auto_seq_0001(self):
        """测试：自动搜索 - 序列0001也视为初始序列"""
        current_path = self._create_mock_sequence("0001")

        result = resolve_previous_sequence(current_path, strategy="auto")

        self.assertIsNone(result)

    def test_resolve_previous_sequence_auto_not_found(self):
        """测试：自动搜索 - 找不到前序列"""
        current_path = self._create_mock_sequence("0005")
        # 没有创建0004

        result = resolve_previous_sequence(current_path, strategy="auto")

        # 找不到应返回None（记录警告）
        self.assertIsNone(result)

    def test_resolve_previous_sequence_auto_skip_invalid_dirs(self):
        """测试：自动搜索 - 跳过无效目录"""
        # 创建一些干扰目录
        (self.test_path / "invalid1").mkdir()
        (self.test_path / "invalid2").mkdir()
        invalid_with_xml = self.test_path / "invalid3"
        invalid_with_xml.mkdir()
        (invalid_with_xml / "index.xml").write_text("not xml")

        # 创建有效的前序列
        self._create_mock_sequence("0004")
        current_path = self._create_mock_sequence("0005")

        result = resolve_previous_sequence(current_path, strategy="auto")

        # 应该能跳过无效目录，找到有效的0004
        expected_path = str(self.test_path / "0004")
        self.assertEqual(result, expected_path)

    # ========================================================================
    # 测试：resolve_previous_sequence - 策略user_provided
    # ========================================================================

    def test_resolve_previous_sequence_user_provided_valid(self):
        """测试：用户提供映射 - 有效"""
        current_path = self._create_mock_sequence("0005")
        previous_path = self._create_mock_sequence("0004")

        mapping = {"0004": previous_path}

        result = resolve_previous_sequence(
            current_path,
            strategy="user_provided",
            user_provided_mapping=mapping
        )

        self.assertEqual(result, previous_path)

    def test_resolve_previous_sequence_user_provided_not_in_mapping(self):
        """测试：用户提供映射 - 序列号不在映射中"""
        current_path = self._create_mock_sequence("0005")

        mapping = {"0003": "/some/path"}  # 0004不在映射中

        result = resolve_previous_sequence(
            current_path,
            strategy="user_provided",
            user_provided_mapping=mapping
        )

        # 找不到应返回None
        self.assertIsNone(result)

    def test_resolve_previous_sequence_user_provided_invalid_path(self):
        """测试：用户提供映射 - 路径无效"""
        current_path = self._create_mock_sequence("0005")

        mapping = {"0004": "/nonexistent/path"}

        with self.assertRaises(SequenceResolverError):
            resolve_previous_sequence(
                current_path,
                strategy="user_provided",
                user_provided_mapping=mapping
            )

    def test_resolve_previous_sequence_user_provided_no_mapping(self):
        """测试：用户提供策略但未提供映射"""
        current_path = self._create_mock_sequence("0005")

        with self.assertRaises(SequenceResolverError):
            resolve_previous_sequence(
                current_path,
                strategy="user_provided",
                user_provided_mapping=None
            )

    # ========================================================================
    # 测试：错误处理
    # ========================================================================

    def test_resolve_previous_sequence_invalid_current_path(self):
        """测试：当前序列路径无效"""
        fake_path = str(self.test_path / "nonexistent")

        with self.assertRaises(SequenceResolverError):
            resolve_previous_sequence(fake_path, strategy="auto")

    def test_resolve_previous_sequence_invalid_sequence_number_format(self):
        """测试：序列号格式无效"""
        # 创建一个序列号不是数字的序列
        seq = MockSequence(sequence_number="ABCD", sections=[])
        seq_path = self.test_path / "invalid_seq"
        generate_mock_index_xml(seq, str(seq_path))

        # 手动修改index.xml，确保sequence_number是ABCD
        # (测试数据生成器可能已经写入了)

        with self.assertRaises(SequenceResolverError):
            resolve_previous_sequence(str(seq_path), strategy="auto")

    def test_resolve_previous_sequence_unknown_strategy(self):
        """测试：未知策略"""
        current_path = self._create_mock_sequence("0005")

        with self.assertRaises(SequenceResolverError):
            resolve_previous_sequence(current_path, strategy="unknown")  # type: ignore

    # ========================================================================
    # 测试：复杂场景
    # ========================================================================

    def test_resolve_previous_sequence_multiple_sequences(self):
        """测试：多个序列的情况"""
        # 创建序列0003, 0004, 0005
        self._create_mock_sequence("0003")
        self._create_mock_sequence("0004")
        current_path = self._create_mock_sequence("0005")

        result = resolve_previous_sequence(current_path, strategy="auto")

        # 应该找到0004，而不是0003
        expected_path = str(self.test_path / "0004")
        self.assertEqual(result, expected_path)

    def test_resolve_previous_sequence_non_sequential_directories(self):
        """测试：目录名不连续的情况"""
        # 创建0003和0005，跳过0004
        self._create_mock_sequence("0003")
        current_path = self._create_mock_sequence("0005")

        result = resolve_previous_sequence(current_path, strategy="auto")

        # 0004不存在，应返回None
        self.assertIsNone(result)


class TestSequenceResolverIntegration(unittest.TestCase):
    """集成测试：使用测试数据生成器的场景"""

    def setUp(self):
        """创建临时测试目录"""
        self.test_dir = tempfile.mkdtemp(prefix="ectd_integration_test_")
        self.test_path = Path(self.test_dir)

    def tearDown(self):
        """清理临时目录"""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)

    def test_compliant_scenario_sequence_resolution(self):
        """测试：合规场景的序列解析"""
        from tests.test_data_generators.ectd_metadata_test_generator import (
            create_compliant_metadata_update_scenario,
            generate_mock_index_xml
        )

        scenarios = create_compliant_metadata_update_scenario()

        # 生成序列0004和0005
        seq_0004_path = self.test_path / "0004"
        seq_0005_path = self.test_path / "0005"

        generate_mock_index_xml(scenarios["0004"], str(seq_0004_path))
        generate_mock_index_xml(scenarios["0005"], str(seq_0005_path))

        # 解析前序列
        prev_path = resolve_previous_sequence(str(seq_0005_path), strategy="auto")

        self.assertEqual(prev_path, str(seq_0004_path))

        # 提取元数据验证
        metadata_0004 = extract_sequence_metadata(str(seq_0004_path))
        metadata_0005 = extract_sequence_metadata(str(seq_0005_path))

        self.assertEqual(metadata_0004["sequence_number"], "0004")
        self.assertEqual(metadata_0005["sequence_number"], "0005")


if __name__ == "__main__":
    unittest.main()
