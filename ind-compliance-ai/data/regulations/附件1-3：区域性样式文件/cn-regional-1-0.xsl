<?xml version="1.0" encoding="UTF-8" ?>
<!--
	BSfile2 cn-regional-1-0.xsl 
	模块一样式文件

	Version 1.0 - 2021年9月1日
-->

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:cn_ectd="cn_ectd" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="cn_ectd ../dtd/cn-regional.xsd" xmlns:xlink="http://www.w3.org/1999/xlink" >
	<xsl:output method="html" encoding="UTF-8" indent="no"/>
	<xsl:template match="/">
		<html>
			<head>
				<title>模块一 - Schema 版本 <xsl:value-of select="//cn_ectd:cn_ectd/@schema-version"/></title>
				<style type="text/css">
					h1, h2, h3, h4 {margin-top:3pt ; margin-bottom:0pt}
					ul {margin-bottom:0pt ; margin-top:0pt}
				</style>
			</head>
			<body>
				<center>
					<h1>模块一</h1>
					<small>Schema 版本 <xsl:value-of select="//cn_ectd:cn_ectd/@schema-version"/></small><br/>
					<small>样式文件版本 1.0</small><br/>
				</center>
				<xsl:apply-templates select="//cn_ectd:cn-envelope"/>
				<br/>
				<xsl:apply-templates select="//cn_ectd:sequence-contact"/>
				<br/>
				<xsl:apply-templates select="//cn_ectd:cn-content"/>
			</body>
		</html>
	</xsl:template>

	<xsl:template match="*|@*" mode="line">
		<xsl:value-of select="."/>
		<xsl:if test="position() != last()"><br/></xsl:if>
	</xsl:template>

	<xsl:template match="*|@*" mode="application-type">
		<xsl:choose>			
			<xsl:when test="@code='cnapt1'">临床试验申请</xsl:when>
			<xsl:when test="@code='cnapt2'">新药申请</xsl:when>
			<xsl:when test="@code='cnapt3'">仿制药申请</xsl:when>
		</xsl:choose>
	</xsl:template>

	<xsl:template match="*|@*" mode="product-type">
		<xsl:choose>
			<xsl:when test="@code='cnprt1'">化学药品</xsl:when>
			<xsl:when test="@code='cnprt2'">生物制品</xsl:when>
		</xsl:choose>
	</xsl:template>

	<xsl:template match="*|@*" mode="regulatory-activity-type">
		<xsl:choose>
			<xsl:when test="@code='cnrat1'">首次申请</xsl:when>
			<xsl:when test="@code='cnrat2'">补充申请</xsl:when>
			<xsl:when test="@code='cnrat3'">备案</xsl:when>
			<xsl:when test="@code='cnrat4'">报告</xsl:when>
			<xsl:when test="@code='cnrat5'">新适应症和联合用药</xsl:when>
			<xsl:when test="@code='cnrat6'">新适应症</xsl:when>
			<xsl:when test="@code='cnrat7'">研发期间安全性报告</xsl:when>
			<xsl:when test="@code='cnrat8'">再注册</xsl:when>
			<xsl:when test="@code='cnrat9'">基线</xsl:when>
		</xsl:choose>
	</xsl:template>

	<xsl:template match="*|@*" mode="sequence-type">
		<xsl:choose>
			<xsl:when test="@code='cnsqt1'">首次提交</xsl:when>
			<xsl:when test="@code='cnsqt2'">回复</xsl:when>
			<xsl:when test="@code='cnsqt3'">撤回</xsl:when>
			<xsl:when test="@code='cnsqt4'">格式转换</xsl:when>
		</xsl:choose>
	</xsl:template>
	
	<xsl:template match="cn_ectd:cn-envelope">
		<center>
			<table width="90%" border="1px" frame="border" rules="groups" cellpadding="2" cellspacing="0">
				<tr>
					<td width="20%" valign="top">申请编号: </td>
					<td><xsl:apply-templates select="cn_ectd:application-id" /></td>
				</tr>
				<tr>
					<td valign="top">申请类型: </td>
					<td><xsl:apply-templates select="cn_ectd:application-type" mode="application-type"/></td>
				</tr>
				<tr>
					<td valign="top">产品类型: </td>
					<td><xsl:apply-templates select="cn_ectd:product-type" mode="product-type"/></td>
				</tr>
				<tr>
					<td valign="top">原始编号: </td>
					<td><xsl:apply-templates select="cn_ectd:product-number" mode="line"/></td>
				</tr>
				<tr>
					<td>相关序列: </td>
					<td><xsl:apply-templates select="cn_ectd:related-sequence"/></td>
				</tr>
				<tr>
					<td valign="top">注册行为类型: </td>
					<td><xsl:apply-templates select="cn_ectd:regulatory-activity-type" mode="regulatory-activity-type"/></td>
				</tr>
				<tr>
					<td>序列号: </td>
					<td><xsl:apply-templates select="cn_ectd:sequence-number"/></td>
				</tr>
				<tr>
					<td valign="top">序列类型: </td>
					<td><xsl:apply-templates select="cn_ectd:sequence-type" mode="sequence-type"/></td>
				</tr>
				<tr>
					<td>序列描述: </td>
					<td><xsl:apply-templates select="cn_ectd:sequence-description"/></td>
				</tr>
			</table>
		</center>
	</xsl:template>
	
	<xsl:template match="cn_ectd:cn-envelope/cn_ectd:sequence-contact">
		<center>
			<table width="90%" border="1px" frame="border" rules="groups" cellpadding="2" cellspacing="0">
				<tr>
					<td width="20%" valign="top">序列联系人</td>
					<td></td>
				</tr>
				<tr>
					<td width="20%" valign="top">姓名: </td>
					<td><xsl:apply-templates select="./cn_ectd:name" /></td>
				</tr>
				<tr>
					<td width="20%" valign="top">电话: </td>
					<td><xsl:apply-templates select="./cn_ectd:phone" mode="line"/></td>
				</tr>
				<tr>
					<td width="20%" valign="top">邮箱: </td>
					<td><xsl:apply-templates select="./cn_ectd:email" /></td>
				</tr>
			</table>
		</center>
	</xsl:template>
	
	<xsl:template match="cn_ectd:leaf">
		<li>
			<a>
				<xsl:attribute name="href"><xsl:value-of select="@xlink:href"/></xsl:attribute>
				<xsl:value-of select="cn_ectd:title"/>
			</a>
			<xsl:text> </xsl:text>
			(<font color="red"><xsl:value-of select="@operation"/></font>)
			<xsl:if test="position() != last()"><br/></xsl:if>
		</li>
	</xsl:template>
	
	<xsl:template match="cn_ectd:cn-content">
		<center>
			<table width="90%" cellpadding="5" cellspacing="2">
				<tr>
					<td colspan="2"><h2>模块一</h2></td>
				</tr>
		
				<!-- 1.0 -->
				<tr>
					<td width="5%" valign="top"><h3>1.0</h3></td>
					<td width="95%">
						<h3>说明函</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-0"/>
						</ul>						
					</td>
				</tr>
				<!-- 1.1 -->
				<tr>
					<td valign="top"><h3>1.1</h3></td>
					<td>
						<h3>目录</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-1"/>
						</ul>
					</td>
				</tr>
				<!-- 1.2 -->
				<tr>
					<td valign="top"><h3>1.2</h3></td>
					<td>
						<h3>申请表</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-2"/>
						</ul>
					</td>
				</tr>
				<!-- 1.3 -->
				<tr>
					<td valign="top"><h3>1.3</h3></td>
					<td>
						<h3>产品信息相关材料</h3>
					</td>
				</tr>
				<!-- 1.3.1 -->
				<tr>
					<td valign="top"><h3>1.3.1</h3></td>
					<td>
						<h3>说明书</h3>
					</td>
				</tr>
				<!-- 1.3.1.1 -->
				<tr>
					<td valign="top"><h3>1.3.1.1</h3></td>
					<td>
						<h3>研究药物说明书及修订说明（适用于临床试验申请）</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-3/cn_ectd:cn-1-3-1/cn_ectd:cn-1-3-1-1"/>
						</ul>
					</td>
				</tr>
				<!-- 1.3.1.2 -->
				<tr>
					<td valign="top"><h3>1.3.1.2</h3></td>
					<td>
						<h3>上市药品说明书及修订说明（适用于上市及上市后变更申请）</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-3/cn_ectd:cn-1-3-1/cn_ectd:cn-1-3-1-2"/>
						</ul>
					</td>
				</tr>
				<!-- 1.3.2 -->
				<tr>
					<td valign="top"><h3>1.3.2</h3></td>
					<td>
						<h3>包装标签</h3>
					</td>
				</tr>
				<!-- 1.3.2.1 -->
				<tr>
					<td valign="top"><h3>1.3.2.1</h3></td>
					<td>
						<h3>研究药物包装标签（适用于临床试验申请）</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-3/cn_ectd:cn-1-3-2/cn_ectd:cn-1-3-2-1"/>
						</ul>
					</td>
				</tr>
				<!-- 1.3.2.2 -->
				<tr>
					<td valign="top"><h3>1.3.2.2</h3></td>
					<td>
						<h3>上市药品包装标签（适用于上市及上市后变更申请）</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-3/cn_ectd:cn-1-3-2/cn_ectd:cn-1-3-2-2"/>
						</ul>
					</td>
				</tr>
				<!-- 1.3.3 -->
				<tr>
					<td valign="top"><h3>1.3.3</h3></td>
					<td>
						<h3>产品质量标准和生产工艺/制造及检定规程</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-3/cn_ectd:cn-1-3-3"/>
						</ul>
					</td>
				</tr>
				<!-- 1.3.4 -->
				<tr>
					<td valign="top"><h3>1.3.4</h3></td>
					<td>
						<h3>临床试验相关资料（适用于临床试验申请）</h3>
					</td>
				</tr>
				<!-- 1.3.4.1 -->
				<tr>
					<td valign="top"><h3>1.3.4.1</h3></td>
					<td>
						<h3>临床试验计划和方案</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-3/cn_ectd:cn-1-3-4/cn_ectd:cn-1-3-4-1"/>
						</ul>
					</td>
				</tr>
				<!-- 1.3.4.2 -->
				<tr>
					<td valign="top"><h3>1.3.4.2</h3></td>
					<td>
						<h3>知情同意书样稿</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-3/cn_ectd:cn-1-3-4/cn_ectd:cn-1-3-4-2"/>
						</ul>
					</td>
				</tr>
				<!-- 1.3.4.3 -->
				<tr>
					<td valign="top"><h3>1.3.4.3</h3></td>
					<td>
						<h3>研究者手册</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-3/cn_ectd:cn-1-3-4/cn_ectd:cn-1-3-4-3"/>
						</ul>
					</td>
				</tr>
				<!-- 1.3.5 -->
				<tr>
					<td valign="top"><h3>1.3.5</h3></td>
					<td>
						<h3>药品通用名称核准申请材料</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-3/cn_ectd:cn-1-3-5"/>
						</ul>
					</td>
				</tr>
				<!-- 1.3.6 -->
				<tr>
					<td valign="top"><h3>1.3.6</h3></td>
					<td>
						<h3>检查相关信息（适用于上市申请和涉及检查检验的补充申请）</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-3/cn_ectd:cn-1-3-6"/>
						</ul>
					</td>
				</tr>
				<!-- 1.3.7 -->
				<tr>
					<td valign="top"><h3>1.3.7</h3></td>
					<td>
						<h3>疫苗生物安全及环境影响评价</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-3/cn_ectd:cn-1-3-7"/>
						</ul>
					</td>
				</tr>
				<!-- 1.3.8 -->
				<tr>
					<td valign="top"><h3>1.3.8</h3></td>
					<td>
						<h3>产品相关证明性文件（如适用）</h3>
					</td>
				</tr>
				<!-- 1.3.8.1 -->
				<tr>
					<td valign="top"><h3>1.3.8.1</h3></td>
					<td>
						<h3>原料药、药用辅料及药包材证明文件</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-3/cn_ectd:cn-1-3-8/cn_ectd:cn-1-3-8-1"/>
						</ul>
					</td>
				</tr>
				<!-- 1.3.8.2 -->
				<tr>
					<td valign="top"><h3>1.3.8.2</h3></td>
					<td>
						<h3>专利信息及证明文件</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-3/cn_ectd:cn-1-3-8/cn_ectd:cn-1-3-8-2"/>
						</ul>
					</td>
				</tr>
				<!-- 1.3.8.3 -->
				<tr>
					<td valign="top"><h3>1.3.8.3</h3></td>
					<td>
						<h3>特殊药品研制立项批准文件</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-3/cn_ectd:cn-1-3-8/cn_ectd:cn-1-3-8-3"/>
						</ul>
					</td>
				</tr>
				<!-- 1.3.8.4 -->
				<tr>
					<td valign="top"><h3>1.3.8.4</h3></td>
					<td>
						<h3>商标信息及证明文件</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-3/cn_ectd:cn-1-3-8/cn_ectd:cn-1-3-8-4"/>
						</ul>
					</td>
				</tr>
				<!-- 1.3.8.5 -->
				<tr>
					<td valign="top"><h3>1.3.8.5</h3></td>
					<td>
						<h3>对照药来源证明文件</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-3/cn_ectd:cn-1-3-8/cn_ectd:cn-1-3-8-5"/>
						</ul>
					</td>
				</tr>
				<!-- 1.3.8.6 -->
				<tr>
					<td valign="top"><h3>1.3.8.6</h3></td>
					<td>
						<h3>药物临床试验相关证明文件（适用于上市申请）</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-3/cn_ectd:cn-1-3-8/cn_ectd:cn-1-3-8-6"/>
						</ul>
					</td>
				</tr>
				<!-- 1.3.8.7 -->
				<tr>
					<td valign="top"><h3>1.3.8.7</h3></td>
					<td>
						<h3>研究机构资质证明文件</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-3/cn_ectd:cn-1-3-8/cn_ectd:cn-1-3-8-7"/>
						</ul>
					</td>
				</tr>
				<!-- 1.3.8.8 -->
				<tr>
					<td valign="top"><h3>1.3.8.8</h3></td>
					<td>
						<h3>药械组合产品相关证明性文件</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-3/cn_ectd:cn-1-3-8/cn_ectd:cn-1-3-8-8"/>
						</ul>
					</td>
				</tr>
				<!-- 1.3.8.9 -->
				<tr>
					<td valign="top"><h3>1.3.8.9</h3></td>
					<td>
						<h3>允许药品上市销售证明文件（适用于境外已上市的药品）</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-3/cn_ectd:cn-1-3-8/cn_ectd:cn-1-3-8-9"/>
						</ul>
					</td>
				</tr>
				<!-- 1.3.8.10 -->
				<tr>
					<td valign="top"><h3>1.3.8.10</h3></td>
					<td>
						<h3>允许药品变更的证明文件</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-3/cn_ectd:cn-1-3-8/cn_ectd:cn-1-3-8-10"/>
						</ul>
					</td>
				</tr>
				<!-- 1.3.9 -->
				<tr>
					<td valign="top"><h3>1.3.9</h3></td>
					<td>
						<h3>其他产品信息相关材料</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-3/cn_ectd:cn-1-3-9"/>
						</ul>
					</td>
				</tr>
				<!-- 1.4 -->
				<tr>
					<td valign="top"><h3>1.4</h3></td>
					<td>
						<h3>申请状态（如适用）</h3>
					</td>
				</tr>
				<!-- 1.4.1 -->
				<tr>
					<td valign="top"><h3>1.4.1</h3></td>
					<td>
						<h3>既往批准情况</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-4/cn_ectd:cn-1-4-1"/>
						</ul>
					</td>
				</tr>
				<!-- 1.4.2 -->
				<tr>
					<td valign="top"><h3>1.4.2</h3></td>
					<td>
						<h3>申请调整临床试验方案、暂停或者终止临床试验</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-4/cn_ectd:cn-1-4-2"/>
						</ul>
					</td>
				</tr>
				<!-- 1.4.3 -->
				<tr>
					<td valign="top"><h3>1.4.3</h3></td>
					<td>
						<h3>暂停后申请恢复临床试验</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-4/cn_ectd:cn-1-4-3"/>
						</ul>
					</td>
				</tr>
				<!-- 1.4.4 -->
				<tr>
					<td valign="top"><h3>1.4.4</h3></td>
					<td>
						<h3>终止后重新申请临床试验</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-4/cn_ectd:cn-1-4-4"/>
						</ul>
					</td>
				</tr>
				<!-- 1.4.5 -->
				<tr>
					<td valign="top"><h3>1.4.5</h3></td>
					<td>
						<h3>申请撤回尚未批准的药物临床试验申请、上市注册许可申请、补充申请或再注册申请</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-4/cn_ectd:cn-1-4-5"/>
						</ul>
					</td>
				</tr>
				<!-- 1.4.6 -->
				<tr>
					<td valign="top"><h3>1.4.6</h3></td>
					<td>
						<h3>申请上市注册审评期间变更仅包括申请人更名、变更注册地址名称等不涉及技术审评内容的变更</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-4/cn_ectd:cn-1-4-6"/>
						</ul>
					</td>
				</tr>
				<!-- 1.4.7 -->
				<tr>
					<td valign="top"><h3>1.4.7</h3></td>
					<td>
						<h3>申请注销药品注册证书</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-4/cn_ectd:cn-1-4-7"/>
						</ul>
					</td>
				</tr>
				<!-- 1.5 -->
				<tr>
					<td valign="top"><h3>1.5</h3></td>
					<td>
						<h3>加快上市注册程序申请（如适用）</h3>
					</td>
				</tr>
				<!-- 1.5.1 -->
				<tr>
					<td valign="top"><h3>1.5.1</h3></td>
					<td>
						<h3>加快上市注册程序申请</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-5/cn_ectd:cn-1-5-1"/>
						</ul>
					</td>
				</tr>
				<!-- 1.5.2 -->
				<tr>
					<td valign="top"><h3>1.5.2</h3></td>
					<td>
						<h3>加快上市注册程序终止申请</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-5/cn_ectd:cn-1-5-2"/>
						</ul>
					</td>
				</tr>
				<!-- 1.5.3 -->
				<tr>
					<td valign="top"><h3>1.5.3</h3></td>
					<td>
						<h3>其他加快注册程序申请</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-5/cn_ectd:cn-1-5-3"/>
						</ul>
					</td>
				</tr>
				<!-- 1.6 -->
				<tr>
					<td valign="top"><h3>1.6</h3></td>
					<td>
						<h3>沟通交流会议（如适用）</h3>
					</td>
				</tr>
				<!-- 1.6.1 -->
				<tr>
					<td valign="top"><h3>1.6.1</h3></td>
					<td>
						<h3>会议申请</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-6/cn_ectd:cn-1-6-1"/>
						</ul>
					</td>
				</tr>
				<!-- 1.6.2 -->
				<tr>
					<td valign="top"><h3>1.6.2</h3></td>
					<td>
						<h3>会议背景资料</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-6/cn_ectd:cn-1-6-2"/>
						</ul>
					</td>
				</tr>
				<!-- 1.6.3 -->
				<tr>
					<td valign="top"><h3>1.6.3</h3></td>
					<td>
						<h3>会议相关信函、会议纪要以及答复</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-6/cn_ectd:cn-1-6-3"/>
						</ul>
					</td>
				</tr>
				<!-- 1.7 -->
				<tr>
					<td valign="top"><h3>1.7</h3></td>
					<td>
						<h3>临床试验过程管理信息（如适用）</h3>
					</td>
				</tr>
				<!-- 1.7.1 -->
				<tr>
					<td valign="top"><h3>1.7.1</h3></td>
					<td>
						<h3>临床试验期间增加适应症</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-7/cn_ectd:cn-1-7-1"/>
						</ul>
					</td>
				</tr>
				<!-- 1.7.2 -->
				<tr>
					<td valign="top"><h3>1.7.2</h3></td>
					<td>
						<h3>临床试验方案变更、非临床或者药学的变化或者新发现等可能增加受试者安全性风险的</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-7/cn_ectd:cn-1-7-2"/>
						</ul>
					</td>
				</tr>
				<!-- 1.7.3 -->
				<tr>
					<td valign="top"><h3>1.7.3</h3></td>
					<td>
						<h3>要求申办者调整临床试验方案、暂停或终止药物临床试验</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-7/cn_ectd:cn-1-7-3"/>
						</ul>
					</td>
				</tr>
				<!-- 1.8 -->
				<tr>
					<td valign="top"><h3>1.8</h3></td>
					<td>
						<h3>药物警戒与风险管理（如适用）</h3>
					</td>
				</tr>
				<!-- 1.8.1 -->
				<tr>
					<td valign="top"><h3>1.8.1</h3></td>
					<td>
						<h3>研发期间安全性更新报告及附件</h3>
					</td>
				</tr>
				<!-- 1.8.1.1 -->
				<tr>
					<td valign="top"><h3>1.8.1.1</h3></td>
					<td>
						<h3>研发期间安全性更新报告</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-8/cn_ectd:cn-1-8-1/cn_ectd:cn-1-8-1-1"/>
						</ul>
					</td>
				</tr>
				<!-- 1.8.1.2 -->
				<tr>
					<td valign="top"><h3>1.8.1.2</h3></td>
					<td>
						<h3>严重不良反应（SAR）累计汇总表</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-8/cn_ectd:cn-1-8-1/cn_ectd:cn-1-8-1-2"/>
						</ul>
					</td>
				</tr>
				<!-- 1.8.1.3 -->
				<tr>
					<td valign="top"><h3>1.8.1.3</h3></td>
					<td>
						<h3>报告周期内境内死亡受试者列表</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-8/cn_ectd:cn-1-8-1/cn_ectd:cn-1-8-1-3"/>
						</ul>
					</td>
				</tr>
				<!-- 1.8.1.4 -->
				<tr>
					<td valign="top"><h3>1.8.1.4</h3></td>
					<td>
						<h3>报告周期内境内因任何不良事件而退出临床试验的受试者列表</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-8/cn_ectd:cn-1-8-1/cn_ectd:cn-1-8-1-4"/>
						</ul>
					</td>
				</tr>
				<!-- 1.8.1.5 -->
				<tr>
					<td valign="top"><h3>1.8.1.5</h3></td>
					<td>
						<h3>报告周期内发生的药物临床试验方案变更或者临床方面的新发现、非临床或者药学的变化或者新发现总结表</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-8/cn_ectd:cn-1-8-1/cn_ectd:cn-1-8-1-5"/>
						</ul>
					</td>
				</tr>
				<!-- 1.8.1.6 -->
				<tr>
					<td valign="top"><h3>1.8.1.6</h3></td>
					<td>
						<h3>下一报告周期内总体研究计划概要</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-8/cn_ectd:cn-1-8-1/cn_ectd:cn-1-8-1-6"/>
						</ul>
					</td>
				</tr>
				<!-- 1.8.2 -->
				<tr>
					<td valign="top"><h3>1.8.2</h3></td>
					<td>
						<h3>其他潜在的严重安全性风险信息</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-8/cn_ectd:cn-1-8-2"/>
						</ul>
					</td>
				</tr>
				<!-- 1.8.3 -->
				<tr>
					<td valign="top"><h3>1.8.3</h3></td>
					<td>
						<h3>风险管理计划（RMP）</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-8/cn_ectd:cn-1-8-3"/>
						</ul>
					</td>
				</tr>				
				<!-- 1.9 -->
				<tr>
					<td valign="top"><h3>1.9</h3></td>
					<td>
						<h3>上市后研究（如适用）</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-9"/>
						</ul>
					</td>
				</tr>				
				<!-- 1.10 -->
				<tr>
					<td valign="top"><h3>1.10</h3></td>
					<td>
						<h3>上市后变更（如适用）</h3>
					</td>
				</tr>
				<!-- 1.10.1 -->
				<tr>
					<td valign="top"><h3>1.10.1</h3></td>
					<td>
						<h3>审批类变更</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-10/cn_ectd:cn-1-10-1"/>
						</ul>
					</td>
				</tr>
				<!-- 1.10.2 -->
				<tr>
					<td valign="top"><h3>1.10.2</h3></td>
					<td>
						<h3>备案类变更</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-10/cn_ectd:cn-1-10-2"/>
						</ul>
					</td>
				</tr>
				<!-- 1.10.3 -->
				<tr>
					<td valign="top"><h3>1.10.3</h3></td>
					<td>
						<h3>报告类变更</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-10/cn_ectd:cn-1-10-3"/>
						</ul>
					</td>
				</tr>
				<!-- 1.11 -->
				<tr>
					<td valign="top"><h3>1.11</h3></td>
					<td>
						<h3>申请人/生产企业证明性文件</h3>
					</td>
				</tr>
				<!-- 1.11.1 -->
				<tr>
					<td valign="top"><h3>1.11.1</h3></td>
					<td>
						<h3>境内生产药品申请人/生产企业资质证明文件</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-11/cn_ectd:cn-1-11-1"/>
						</ul>
					</td>
				</tr>
				<!-- 1.11.2 -->
				<tr>
					<td valign="top"><h3>1.11.2</h3></td>
					<td>
						<h3>境外生产药品申请人/生产企业资质证明文件</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-11/cn_ectd:cn-1-11-2"/>
						</ul>
					</td>
				</tr>
				<!-- 1.11.3 -->
				<tr>
					<td valign="top"><h3>1.11.3</h3></td>
					<td>
						<h3>注册代理机构证明文件</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-11/cn_ectd:cn-1-11-3"/>
						</ul>
					</td>
				</tr>			
				<!-- 1.12 -->
				<tr>
					<td valign="top"><h3>1.12</h3></td>
					<td>
						<h3>小微企业证明文件（如适用）</h3>
						<ul type="square">
							<xsl:apply-templates select="cn_ectd:cn-1-12"/>
						</ul>
					</td>
				</tr>
			</table>
		</center>
	</xsl:template>
</xsl:stylesheet>