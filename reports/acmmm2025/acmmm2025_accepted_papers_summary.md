# ACM MM 2025 全量 accepted papers 快速筛选（标题信号版）

数据来源：ACM MM 2025 官网所有 accepted 页面（WordPress JSON API）。

## 1) 全量覆盖统计

| 类别 | 数量 | 占比 |
|---|---:|---:|
| Regular Papers | 1250 | 84.69% |
| Datasets | 123 | 8.33% |
| Brave New Ideas | 44 | 2.98% |
| Demo/Video | 30 | 2.03% |
| Open Source Software | 14 | 0.95% |
| Interactive Art | 12 | 0.81% |
| Doctoral Symposium | 3 | 0.20% |

总计：**1476** 篇（已遍历全部 accepted 列表条目）。

## 2) 按标题规则的可实现性初筛

* KEEP（倾向小改动/易落地）：**226**
* MAYBE（信息不足，需二次确认）：**1173**
* PASS（标题即显示实现成本较高）：**77**

说明：本次为**全量标题级别**筛选；S3（是否有代码）默认 Unknown/0，后续可对 KEEP 集合再批量补 arXiv/GitHub 校验。

## 3) 输出文件

* `acmmm2025_all_papers_screening.csv`：所有论文逐条记录（含 task/type/score/verdict/ease）。
* `acmmm2025_accepted_papers_by_category.csv`：按类别统计。