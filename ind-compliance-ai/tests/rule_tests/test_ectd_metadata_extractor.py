"""
测试 eCTD元数据提取器

测试从index.xml提取section和leaf元数据的核心功能。
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any

from core.ectd_metadata_extractor import (
    ECTDMetadataExtractor,
    extract_sequence_metadata_index,
    SequenceMetadataIndex,
    SectionMetadataSnapshot,
    LeafMetadata,
)
from core.ectd_section_identifier import SectionIdentifier


class TestECTDMetadataExtractor:
    """测试ECTDMetadataExtractor类"""

    @pytest.fixture
    def temp_sequence_dir(self, tmp_path):
        """创建临时序列目录"""
        seq_dir = tmp_path / "0005"
        seq_dir.mkdir()
        return seq_dir

    @pytest.fixture
    def minimal_index_xml(self, temp_sequence_dir):
        """创建最小化的index.xml"""
        index_content = """<?xml version="1.0" encoding="UTF-8"?>
<ectd:ectd xmlns:ectd="http://www.ich.org/ectd"
           xmlns:xlink="http://www.w3.org/1999/xlink"
           dtd-version="3.2.2">
    <ectd:admin>
        <ectd:sequence-number>0005</ectd:sequence-number>
        <ectd:submission-description>补充资料</ectd:submission-description>
    </ectd:admin>
</ectd:ectd>"""
        index_path = temp_sequence_dir / "index.xml"
        index_path.write_text(index_content, encoding='utf-8')
        return index_path

    @pytest.fixture
    def index_with_sections(self, temp_sequence_dir):
        """创建包含section的index.xml"""
        index_content = """<?xml version="1.0" encoding="UTF-8"?>
<ectd:ectd xmlns:ectd="http://www.ich.org/ectd"
           xmlns:xlink="http://www.w3.org/1999/xlink"
           dtd-version="3.2.2">
    <ectd:admin>
        <ectd:sequence-number>0005</ectd:sequence-number>
    </ectd:admin>
    <ectd:m2>
        <ectd:m2-3-body-data>
            <ectd:m2-3-s-drug-substance substance="API-A" manufacturer="Manufacturer-X">
                <ectd:leaf ID="s001" operation="new" xlink:href="m2/s/api-a/summary.pdf">
                    <ectd:title>API-A Summary</ectd:title>
                    <ectd:checksum type="md5">abc123def456</ectd:checksum>
                </ectd:leaf>
            </ectd:m2-3-s-drug-substance>
        </ectd:m2-3-body-data>
    </ectd:m2>
</ectd:ectd>"""
        index_path = temp_sequence_dir / "index.xml"
        index_path.write_text(index_content, encoding='utf-8')
        return index_path

    def test_extractor_init_success(self, temp_sequence_dir, minimal_index_xml):
        """测试提取器初始化成功"""
        extractor = ECTDMetadataExtractor(str(temp_sequence_dir))
        assert extractor.sequence_path == temp_sequence_dir
        assert extractor.index_xml_path == minimal_index_xml

    def test_extractor_init_no_index_xml(self, temp_sequence_dir):
        """测试index.xml不存在时抛出异常"""
        with pytest.raises(FileNotFoundError, match="index.xml not found"):
            ECTDMetadataExtractor(str(temp_sequence_dir))

    def test_extract_sequence_number_from_xml(self, temp_sequence_dir, minimal_index_xml):
        """测试从XML提取序列号"""
        extractor = ECTDMetadataExtractor(str(temp_sequence_dir))
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(minimal_index_xml))
        root = tree.getroot()
        seq_num = extractor._extract_sequence_number(root)
        assert seq_num == "0005"

    def test_extract_sequence_number_from_dirname(self, tmp_path):
        """测试从目录名提取序列号"""
        seq_dir = tmp_path / "0007"
        seq_dir.mkdir()
        # 创建空index.xml（无sequence-number元素）
        index_content = """<?xml version="1.0" encoding="UTF-8"?>
<ectd:ectd xmlns:ectd="http://www.ich.org/ectd"></ectd:ectd>"""
        (seq_dir / "index.xml").write_text(index_content, encoding='utf-8')

        extractor = ECTDMetadataExtractor(str(seq_dir))
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(seq_dir / "index.xml"))
        root = tree.getroot()
        seq_num = extractor._extract_sequence_number(root)
        assert seq_num == "7"

    def test_extract_envelope_version(self, temp_sequence_dir, minimal_index_xml):
        """测试提取envelope版本"""
        extractor = ECTDMetadataExtractor(str(temp_sequence_dir))
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(minimal_index_xml))
        root = tree.getroot()
        version = extractor._extract_envelope_version(root)
        assert version == "3.2.2"

    def test_extract_submission_description(self, temp_sequence_dir, minimal_index_xml):
        """测试提取提交描述"""
        extractor = ECTDMetadataExtractor(str(temp_sequence_dir))
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(minimal_index_xml))
        root = tree.getroot()
        desc = extractor._extract_submission_description(root)
        assert desc == "补充资料"

    def test_extract_metadata_index_minimal(self, temp_sequence_dir, minimal_index_xml):
        """测试提取最小化的元数据索引"""
        extractor = ECTDMetadataExtractor(str(temp_sequence_dir))
        index = extractor.extract_metadata_index()

        assert isinstance(index, SequenceMetadataIndex)
        assert index.sequence_number == "0005"
        assert index.envelope_version == "3.2.2"
        assert index.submission_description == "补充资料"
        assert index.total_sections == 0
        assert index.total_leafs == 0

    def test_extract_metadata_index_with_section(self, temp_sequence_dir, index_with_sections):
        """测试提取包含section的元数据索引"""
        extractor = ECTDMetadataExtractor(str(temp_sequence_dir))
        index = extractor.extract_metadata_index()

        assert index.sequence_number == "0005"
        assert index.total_sections == 1
        assert index.total_leafs == 1

        # 验证section提取正确
        sections = index.get_all_sections()
        assert len(sections) == 1

        section = sections[0]
        assert isinstance(section, SectionMetadataSnapshot)
        assert section.identifier.element_name == "m2-3-s-drug-substance"
        assert section.identifier.attributes["substance"] == "API-A"
        assert section.identifier.attributes["manufacturer"] == "Manufacturer-X"

    def test_extract_leaf_metadata(self, temp_sequence_dir, index_with_sections):
        """测试提取leaf元数据"""
        extractor = ECTDMetadataExtractor(str(temp_sequence_dir))
        index = extractor.extract_metadata_index()

        section = index.get_all_sections()[0]
        assert len(section.leaf_metadata) == 1

        leaf = section.leaf_metadata[0]
        assert isinstance(leaf, LeafMetadata)
        assert leaf.leaf_id == "s001"
        assert leaf.operation == "new"
        assert leaf.title == "API-A Summary"
        assert leaf.xlink_href == "m2/s/api-a/summary.pdf"
        assert leaf.checksum == "abc123def456"
        assert leaf.checksum_type == "md5"

    def test_section_matching_key_generation(self, temp_sequence_dir, index_with_sections):
        """测试section匹配键生成"""
        extractor = ECTDMetadataExtractor(str(temp_sequence_dir))
        index = extractor.extract_metadata_index()

        section = index.get_all_sections()[0]
        matching_key = section.get_matching_key()

        # 验证匹配键格式
        # Note: 只包含主键属性（substance），不包含可变的元数据（manufacturer）
        assert "m2-3-s-drug-substance" in matching_key
        assert "API-A" in matching_key
        assert "Manufacturer-X" not in matching_key  # manufacturer不是主键，不在匹配键中

    def test_convenience_function(self, temp_sequence_dir, index_with_sections):
        """测试便捷函数extract_sequence_metadata_index"""
        index = extract_sequence_metadata_index(str(temp_sequence_dir))

        assert isinstance(index, SequenceMetadataIndex)
        assert index.sequence_number == "0005"
        assert index.total_sections == 1


class TestComplexSections:
    """测试复杂的section场景"""

    @pytest.fixture
    def complex_index_xml(self, tmp_path):
        """创建包含多个section的复杂index.xml"""
        seq_dir = tmp_path / "0006"
        seq_dir.mkdir()

        index_content = """<?xml version="1.0" encoding="UTF-8"?>
<ectd:ectd xmlns:ectd="http://www.ich.org/ectd"
           xmlns:xlink="http://www.w3.org/1999/xlink"
           dtd-version="3.2.2">
    <ectd:admin>
        <ectd:sequence-number>0006</ectd:sequence-number>
    </ectd:admin>
    <ectd:m2>
        <ectd:m2-3-body-data>
            <ectd:m2-3-s-drug-substance substance="API-A" manufacturer="Mfr-X">
                <ectd:leaf ID="s001" operation="new" xlink:href="m2/s1.pdf">
                    <ectd:title>API-A Doc</ectd:title>
                </ectd:leaf>
            </ectd:m2-3-s-drug-substance>
            <ectd:m2-3-s-drug-substance substance="API-B" manufacturer="Mfr-Y">
                <ectd:leaf ID="s002" operation="new" xlink:href="m2/s2.pdf">
                    <ectd:title>API-B Doc</ectd:title>
                </ectd:leaf>
                <ectd:leaf ID="s003" operation="new" xlink:href="m2/s3.pdf">
                    <ectd:title>API-B Specs</ectd:title>
                </ectd:leaf>
            </ectd:m2-3-s-drug-substance>
            <ectd:m2-3-p-drug-product product-name="Tablet-A" dosageform="tablet" manufacturer="Mfr-Z">
                <ectd:leaf ID="p001" operation="replace" modified-file="p000" xlink:href="m2/p1.pdf">
                    <ectd:title>Product Summary Updated</ectd:title>
                </ectd:leaf>
            </ectd:m2-3-p-drug-product>
        </ectd:m2-3-body-data>
    </ectd:m2>
</ectd:ectd>"""
        index_path = seq_dir / "index.xml"
        index_path.write_text(index_content, encoding='utf-8')
        return seq_dir

    def test_extract_multiple_sections(self, complex_index_xml):
        """测试提取多个section"""
        index = extract_sequence_metadata_index(str(complex_index_xml))

        assert index.total_sections == 3
        assert index.total_leafs == 4

    def test_extract_different_section_types(self, complex_index_xml):
        """测试提取不同类型的section"""
        index = extract_sequence_metadata_index(str(complex_index_xml))

        sections = index.get_all_sections()
        element_names = {s.identifier.element_name for s in sections}

        assert "m2-3-s-drug-substance" in element_names
        assert "m2-3-p-drug-product" in element_names

    def test_extract_multiple_leafs_per_section(self, complex_index_xml):
        """测试section包含多个leaf"""
        index = extract_sequence_metadata_index(str(complex_index_xml))

        # 查找API-B section（有2个leafs）
        for section in index.get_all_sections():
            if section.identifier.attributes.get("substance") == "API-B":
                assert len(section.leaf_metadata) == 2
                leaf_ids = {leaf.leaf_id for leaf in section.leaf_metadata}
                assert "s002" in leaf_ids
                assert "s003" in leaf_ids
                break
        else:
            pytest.fail("API-B section not found")

    def test_extract_replace_operation(self, complex_index_xml):
        """测试提取replace操作的leaf"""
        index = extract_sequence_metadata_index(str(complex_index_xml))

        # 查找drug-product section
        for section in index.get_all_sections():
            if section.identifier.element_name == "m2-3-p-drug-product":
                leaf = section.leaf_metadata[0]
                assert leaf.operation == "replace"
                assert leaf.modified_file == "p000"
                break
        else:
            pytest.fail("drug-product section not found")


class TestSectionIdentification:
    """测试section识别逻辑"""

    @pytest.fixture
    def extractor_instance(self, tmp_path):
        """创建提取器实例"""
        seq_dir = tmp_path / "0001"
        seq_dir.mkdir()
        index_content = """<?xml version="1.0" encoding="UTF-8"?>
<ectd:ectd xmlns:ectd="http://www.ich.org/ectd">
    <ectd:admin><ectd:sequence-number>0001</ectd:sequence-number></ectd:admin>
</ectd:ectd>"""
        (seq_dir / "index.xml").write_text(index_content, encoding='utf-8')
        return ECTDMetadataExtractor(str(seq_dir))

    def test_should_track_known_sections(self, extractor_instance):
        """测试识别需要追踪的section类型"""
        assert extractor_instance._should_track_section("m2-3-s-drug-substance") is True
        assert extractor_instance._should_track_section("m3-2-s-drug-substance") is True
        assert extractor_instance._should_track_section("m2-3-p-drug-product") is True
        assert extractor_instance._should_track_section("m3-2-p-drug-product") is True

    def test_should_not_track_unknown_sections(self, extractor_instance):
        """测试不追踪未知section类型"""
        assert extractor_instance._should_track_section("m1-cover-letter") is False
        assert extractor_instance._should_track_section("m4-study") is False
        assert extractor_instance._should_track_section("unknown-element") is False


class TestSequenceMetadataIndex:
    """测试SequenceMetadataIndex数据结构"""

    def test_add_section(self):
        """测试添加section"""
        index = SequenceMetadataIndex(sequence_number="0005", sequence_path="/path/to/0005")

        identifier = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A"}
        )

        snapshot = SectionMetadataSnapshot(
            sequence_number="0005",
            identifier=identifier,
            leaf_metadata=[],
            section_path="m2/m2-3/s"
        )

        index.add_section(snapshot)

        assert index.total_sections == 1
        assert index.total_leafs == 0

    def test_get_section_by_key(self):
        """测试通过匹配键获取section"""
        index = SequenceMetadataIndex(sequence_number="0005", sequence_path="/path/to/0005")

        identifier = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A", "manufacturer": "Mfr-X"}
        )

        snapshot = SectionMetadataSnapshot(
            sequence_number="0005",
            identifier=identifier,
            leaf_metadata=[],
            section_path="m2/m2-3/s"
        )

        index.add_section(snapshot)

        # 使用相同的匹配键检索
        key = identifier.get_matching_key()
        retrieved = index.get_section_by_key(key)

        assert retrieved is not None
        assert retrieved.identifier.element_name == "m2-3-s-drug-substance"

    def test_leaf_ids_property(self):
        """测试leaf_ids属性（向后兼容）"""
        leaf1 = LeafMetadata(leaf_id="l001", operation="new", title="Doc 1")
        leaf2 = LeafMetadata(leaf_id="l002", operation="new", title="Doc 2")

        identifier = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A"}
        )

        snapshot = SectionMetadataSnapshot(
            sequence_number="0005",
            identifier=identifier,
            leaf_metadata=[leaf1, leaf2],
            section_path="m2/m2-3/s"
        )

        leaf_ids = snapshot.leaf_ids
        assert isinstance(leaf_ids, set)
        assert len(leaf_ids) == 2
        assert "l001" in leaf_ids
        assert "l002" in leaf_ids
