"""
测试eCTD Section标识和匹配模块

测试覆盖：
1. SectionIdentifier基本功能
2. 匹配键生成和比对
3. 跨模块配对查找
4. 主键提取和属性值获取
5. 生命周期追踪判断
6. 工具函数

版本: v1.0
创建日期: 2026-09-11
"""

import unittest
from core.ectd_section_identifier import (
    SectionIdentifier,
    create_section_identifier_from_xml_element,
    find_cross_module_pair,
    get_configured_section_elements,
    is_section_configured,
    get_section_rule,
    SECTION_MATCHING_RULES
)


# ============================================================================
# 测试组1: SectionIdentifier基本功能
# ============================================================================

class TestSectionIdentifierBasics(unittest.TestCase):
    """测试SectionIdentifier的基本功能"""

    def test_create_simple_identifier(self):
        """测试创建简单的标识符"""
        identifier = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A", "manufacturer": "MFR-X"}
        )

        assert identifier.element_name == "m2-3-s-drug-substance"
        assert identifier.attributes["substance"] == "API-A"
        assert identifier.attributes["manufacturer"] == "MFR-X"

    def test_get_primary_keys_configured_element(self):
        """测试获取已配置元素的主键"""
        identifier = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A", "manufacturer": "MFR-X"}
        )

        primary_keys = identifier.get_primary_keys()
        # Note: 只有substance是主键，manufacturer是可变元数据
        assert set(primary_keys) == {"substance"}

    def test_get_primary_keys_unconfigured_element(self):
        """测试获取未配置元素的主键（返回所有属性）"""
        identifier = SectionIdentifier(
            element_name="unknown-element",
            attributes={"attr1": "val1", "attr2": "val2"}
        )

        primary_keys = identifier.get_primary_keys()
        assert set(primary_keys) == {"attr1", "attr2"}

    def test_to_display_path(self):
        """测试生成显示路径"""
        identifier = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A", "manufacturer": "MFR-X"}
        )

        display_path = identifier.to_display_path()
        assert "m2-3-s-drug-substance" in display_path
        assert "substance='API-A'" in display_path
        assert "manufacturer='MFR-X'" in display_path

    def test_to_display_path_no_attributes(self):
        """测试没有属性的显示路径"""
        identifier = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={}
        )

        display_path = identifier.to_display_path()
        assert display_path == "m2-3-s-drug-substance"

    def test_get_description(self):
        """测试获取描述"""
        identifier = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={}
        )

        description = identifier.get_description()
        assert description == "原料药药学信息摘要"


# ============================================================================
# 测试组2: 匹配键生成和比对
# ============================================================================

class TestMatchingKeyGeneration(unittest.TestCase):
    """测试匹配键生成和比对逻辑"""

    def test_get_matching_key_simple(self):
        """测试简单的匹配键生成"""
        identifier = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A", "manufacturer": "MFR-X"}
        )

        key = identifier.get_matching_key()
        # Note: 只包含主键属性（substance），不包含manufacturer
        assert key == "m2-3-s-drug-substance?substance=API-A"

    def test_get_matching_key_special_characters(self):
        """测试特殊字符的URL编码"""
        identifier = SectionIdentifier(
            element_name="m2-3-p-drug-product",
            attributes={
                "product-name": "Product A (Tablet)",
                "dosageform": "片剂"
            }
        )

        key = identifier.get_matching_key()
        # 特殊字符应该被URL编码
        assert "Product+A+%28Tablet%29" in key or "Product%20A%20%28Tablet%29" in key
        # 中文也应该被编码
        assert "%E7%89%87%E5%89%82" in key

    def test_get_matching_key_consistency(self):
        """测试匹配键的一致性（相同属性生成相同键）"""
        id1 = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A", "manufacturer": "MFR-X"}
        )
        id2 = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"manufacturer": "MFR-X", "substance": "API-A"}  # 顺序不同
        )

        # 即使属性添加顺序不同，匹配键应该相同（因为排序）
        assert id1.get_matching_key() == id2.get_matching_key()

    def test_matches_same_identifier(self):
        """测试相同标识符的匹配"""
        id1 = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A", "manufacturer": "MFR-X"}
        )
        id2 = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A", "manufacturer": "MFR-X"}
        )

        assert id1.matches(id2)
        assert id1 == id2  # 测试__eq__
        assert hash(id1) == hash(id2)  # 测试__hash__

    def test_matches_different_identifier(self):
        """测试不同标识符不匹配"""
        id1 = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A", "manufacturer": "MFR-X"}
        )
        id2 = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-B", "manufacturer": "MFR-Y"}  # 不同substance（主键）
        )

        assert not id1.matches(id2)
        assert id1 != id2

    def test_matches_ignores_non_primary_keys(self):
        """测试匹配时忽略非主键属性"""
        # m2-3-s的主键只有substance（manufacturer不是主键）
        id1 = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={
                "substance": "API-A",
                "manufacturer": "MFR-X",
                "extra-attr": "value1"  # 非主键
            }
        )
        id2 = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={
                "substance": "API-A",
                "manufacturer": "MFR-Y",  # 非主键，不同值
                "extra-attr": "value2"  # 非主键，不同值
            }
        )

        # 应该匹配（因为主键substance相同）
        assert id1.matches(id2)


# ============================================================================
# 测试组3: 跨模块配对
# ============================================================================

class TestCrossModulePairing(unittest.TestCase):
    """测试跨模块配对功能"""

    def test_get_cross_module_pair_m2_to_m3(self):
        """测试从m2-3-s到m3-2-s的配对"""
        identifier = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A"}
        )

        pair_name = identifier.get_cross_module_pair()
        assert pair_name == "m3-2-s-drug-substance"

    def test_get_cross_module_pair_m3_to_m2(self):
        """测试从m3-2-s到m2-3-s的配对"""
        identifier = SectionIdentifier(
            element_name="m3-2-s-drug-substance",
            attributes={"substance": "API-A"}
        )

        pair_name = identifier.get_cross_module_pair()
        assert pair_name == "m2-3-s-drug-substance"

    def test_get_cross_module_pair_no_pair(self):
        """测试没有配对的元素"""
        identifier = SectionIdentifier(
            element_name="m3-2-p-4-control-of-excipients",
            attributes={"excipient": "Lactose"}
        )

        pair_name = identifier.get_cross_module_pair()
        assert pair_name is None

    def test_find_cross_module_pair_success(self):
        """测试成功查找跨模块配对"""
        m2_section = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A", "manufacturer": "MFR-X"}
        )
        m3_section = SectionIdentifier(
            element_name="m3-2-s-drug-substance",
            attributes={"substance": "API-A", "manufacturer": "MFR-X"}
        )
        other_section = SectionIdentifier(
            element_name="m3-2-s-drug-substance",
            attributes={"substance": "API-B", "manufacturer": "MFR-Y"}
        )

        candidates = [m3_section, other_section]
        pair = find_cross_module_pair(m2_section, candidates)

        assert pair is not None
        assert pair == m3_section

    def test_find_cross_module_pair_not_found(self):
        """测试找不到跨模块配对"""
        m2_section = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A", "manufacturer": "MFR-X"}
        )
        wrong_section = SectionIdentifier(
            element_name="m3-2-s-drug-substance",
            attributes={"substance": "API-B", "manufacturer": "MFR-Y"}  # 不匹配
        )

        candidates = [wrong_section]
        pair = find_cross_module_pair(m2_section, candidates)

        assert pair is None

    def test_find_cross_module_pair_no_pair_element(self):
        """测试元素本身没有配对时查找返回None"""
        section = SectionIdentifier(
            element_name="m3-2-p-4-control-of-excipients",
            attributes={"excipient": "Lactose"}
        )

        candidates = []
        pair = find_cross_module_pair(section, candidates)

        assert pair is None


# ============================================================================
# 测试组4: 主键和属性值提取
# ============================================================================

class TestPrimaryAttributeExtraction(unittest.TestCase):
    """测试主键和属性值提取"""

    def test_get_primary_attribute_values(self):
        """测试获取主键属性值"""
        identifier = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={
                "substance": "API-A",
                "manufacturer": "MFR-X",
                "other-attr": "value"
            }
        )

        primary_values = identifier.get_primary_attribute_values()
        # Note: 只有substance是主键
        assert primary_values == {"substance": "API-A"}
        assert "other-attr" not in primary_values
        assert "manufacturer" not in primary_values

    def test_get_primary_attribute_values_missing_key(self):
        """测试缺少主键属性时返回空字符串"""
        identifier = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"manufacturer": "MFR-X"}  # 缺少substance（主键）
        )

        primary_values = identifier.get_primary_attribute_values()
        assert primary_values["substance"] == ""  # 缺少的主键返回空字符串

    def test_get_primary_attribute_values_unconfigured(self):
        """测试未配置元素返回所有属性"""
        identifier = SectionIdentifier(
            element_name="unknown-element",
            attributes={"attr1": "val1", "attr2": "val2"}
        )

        primary_values = identifier.get_primary_attribute_values()
        assert primary_values == {"attr1": "val1", "attr2": "val2"}


# ============================================================================
# 测试组5: 生命周期追踪
# ============================================================================

class TestLifecycleTracking(unittest.TestCase):
    """测试生命周期追踪判断"""

    def test_should_track_lifecycle_true(self):
        """测试需要追踪生命周期的元素"""
        identifier = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={}
        )

        assert identifier.should_track_lifecycle() is True

    def test_should_track_lifecycle_false(self):
        """测试不需要追踪生命周期的元素"""
        identifier = SectionIdentifier(
            element_name="m3-2-p-4-control-of-excipients",
            attributes={}
        )

        assert identifier.should_track_lifecycle() is False

    def test_should_track_lifecycle_unconfigured(self):
        """测试未配置元素默认不追踪"""
        identifier = SectionIdentifier(
            element_name="unknown-element",
            attributes={}
        )

        assert identifier.should_track_lifecycle() is False


# ============================================================================
# 测试组6: 工具函数
# ============================================================================

class TestUtilityFunctions(unittest.TestCase):
    """测试工具函数"""

    def test_create_section_identifier_from_xml_element(self):
        """测试从XML元素创建标识符"""
        identifier = create_section_identifier_from_xml_element(
            element_name="m2-3-s-drug-substance",
            xml_attributes={"substance": "API-A", "manufacturer": "MFR-X"}
        )

        assert identifier.element_name == "m2-3-s-drug-substance"
        assert identifier.attributes["substance"] == "API-A"

    def test_create_section_identifier_from_xml_with_namespace(self):
        """测试从带命名空间的XML元素创建标识符"""
        identifier = create_section_identifier_from_xml_element(
            element_name="{http://www.ich.org/ectd}m2-3-s-drug-substance",
            xml_attributes={"substance": "API-A", "manufacturer": "MFR-X"}
        )

        # 命名空间应该被移除
        assert identifier.element_name == "m2-3-s-drug-substance"
        assert identifier.attributes["substance"] == "API-A"

    def test_get_configured_section_elements(self):
        """测试获取所有已配置的元素"""
        elements = get_configured_section_elements()

        assert isinstance(elements, list)
        assert "m2-3-s-drug-substance" in elements
        assert "m3-2-s-drug-substance" in elements
        assert "m2-3-p-drug-product" in elements
        assert len(elements) == len(SECTION_MATCHING_RULES)

    def test_is_section_configured_true(self):
        """测试元素已配置"""
        assert is_section_configured("m2-3-s-drug-substance") is True

    def test_is_section_configured_false(self):
        """测试元素未配置"""
        assert is_section_configured("unknown-element") is False

    def test_get_section_rule_exists(self):
        """测试获取存在的规则"""
        rule = get_section_rule("m2-3-s-drug-substance")

        assert rule is not None
        # Note: 只有substance是主键
        assert rule["primary_keys"] == ["substance"]
        assert rule["lifecycle_tracking"] is True

    def test_get_section_rule_not_exists(self):
        """测试获取不存在的规则"""
        rule = get_section_rule("unknown-element")

        assert rule is None


# ============================================================================
# 测试组7: 复杂场景
# ============================================================================

class TestComplexScenarios(unittest.TestCase):
    """测试复杂场景"""

    def test_multiple_sections_with_same_element_different_attributes(self):
        """测试相同元素、不同属性的多个section"""
        section1 = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A", "manufacturer": "MFR-X"}
        )
        section2 = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-B", "manufacturer": "MFR-Y"}
        )
        section3 = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A", "manufacturer": "MFR-X"}  # 与section1相同
        )

        assert not section1.matches(section2)
        assert section1.matches(section3)
        assert not section2.matches(section3)

    def test_cross_module_pairing_multiple_candidates(self):
        """测试多个候选中查找正确的配对"""
        m2_section = SectionIdentifier(
            element_name="m2-3-p-drug-product",
            attributes={
                "product-name": "Product A",
                "dosageform": "Tablet",
                "manufacturer": "MFR-X"
            }
        )

        candidates = [
            SectionIdentifier(
                element_name="m3-2-p-drug-product",
                attributes={
                    "product-name": "Product B",
                    "dosageform": "Capsule",
                    "manufacturer": "MFR-Y"
                }
            ),
            SectionIdentifier(
                element_name="m3-2-p-drug-product",
                attributes={
                    "product-name": "Product A",
                    "dosageform": "Tablet",
                    "manufacturer": "MFR-X"
                }
            ),
            SectionIdentifier(
                element_name="m3-2-s-drug-substance",  # 错误的元素类型
                attributes={
                    "product-name": "Product A",
                    "dosageform": "Tablet",
                    "manufacturer": "MFR-X"
                }
            )
        ]

        pair = find_cross_module_pair(m2_section, candidates)
        assert pair is not None
        assert pair.element_name == "m3-2-p-drug-product"
        assert pair.attributes["product-name"] == "Product A"

    def test_section_identifier_hashable(self):
        """测试SectionIdentifier可以用作字典键或集合元素"""
        section1 = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A", "manufacturer": "MFR-X"}
        )
        section2 = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A", "manufacturer": "MFR-X"}
        )
        section3 = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-B", "manufacturer": "MFR-Y"}
        )

        # 测试用作字典键
        section_dict = {section1: "data1", section3: "data3"}
        assert section_dict[section2] == "data1"  # section1和section2相同

        # 测试用作集合元素
        section_set = {section1, section2, section3}
        assert len(section_set) == 2  # section1和section2是重复的

    def test_repr_and_str(self):
        """测试__repr__方法"""
        section = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A", "manufacturer": "MFR-X"}
        )

        repr_str = repr(section)
        assert "SectionIdentifier" in repr_str
        assert "m2-3-s-drug-substance" in repr_str


# ============================================================================
# 测试组8: 边界情况
# ============================================================================

class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_empty_attributes(self):
        """测试空属性"""
        identifier = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={}
        )

        # 应该返回主键（但值为空字符串）
        primary_values = identifier.get_primary_attribute_values()
        # Note: 只有substance是主键
        assert primary_values == {"substance": ""}

        # 匹配键应该包含空值
        key = identifier.get_matching_key()
        assert "substance=" in key

    def test_matching_key_with_empty_values(self):
        """测试包含空值的匹配键"""
        id1 = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "", "manufacturer": ""}
        )
        id2 = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "", "manufacturer": ""}
        )

        assert id1.matches(id2)

    def test_special_characters_in_values(self):
        """测试属性值中的特殊字符"""
        identifier = SectionIdentifier(
            element_name="m2-3-p-drug-product",
            attributes={
                "product-name": "Product A/B (10mg/5ml)",
                "dosageform": "注射液"
            }
        )

        key = identifier.get_matching_key()
        # URL编码应该处理特殊字符
        assert "/" not in key.split("?")[1]  # 查询字符串部分不应包含未编码的/
        assert "(" not in key.split("?")[1]
        assert ")" not in key.split("?")[1]


if __name__ == "__main__":
    unittest.main()
