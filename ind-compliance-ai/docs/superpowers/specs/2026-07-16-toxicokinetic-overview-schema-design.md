# Toxicokinetic Overview Schema Design

## Goal

Project eight-column toxicokinetic overview tables as typed overview inventory objects with source-backed `位置 -> 卷/页码` multilevel headers.

## Schema

The canonical leaf columns are:

1. `试验类型`
2. `试验系统`
3. `给药方法`
4. `剂量(mg/kg)`
5. `GLP依从性`
6. `试验编号`
7. `卷`
8. `页码`

The profile identity is `toxicokinetic_overview_inventory_table`. It is distinct from the seven-column PK overview and ten-column nonclinical overview profiles.

## Admission Evidence

Admission requires the stable axes `试验类型`, `试验系统`, `给药方法`, `剂量`, `GLP/依从性`, and `试验编号`, plus locator evidence from `位置`, `卷`, or `页码`. This profile must be selected before the broader nonclinical fallback.

## Header Reconstruction

Header anchors come from source word geometry. The main header band supplies the first six leaves. A lower header band supplies `卷` and `页码`; an upper `位置` word remains a parent candidate. The shared dynamic location-group detector then creates a span over leaf indexes 6-7.

Packed source-grid text such as `位置 页码` is not a canonical leaf. It is replaced by the source-word projection.

## Row Projection

Data rows project to the eight canonical anchors. A row with a four-or-more-digit `试验编号` and sufficient populated cells is a complete record. A sparse continuation containing the wrapped first-column text is appended to the preceding record, preserving entries such as `3个月剂量范围探索试验`.

## Regression Scope

- Page 91 gains the typed profile, two-row location header, eight-column data, and wrapped first record.
- Page 77/79 seven-column PK and page 88/89 ten-column nonclinical profiles remain unchanged.
- Source-less `卷/页码` leaves do not synthesize a parent.
- Markdown renders the two header rows before page-91 data.
