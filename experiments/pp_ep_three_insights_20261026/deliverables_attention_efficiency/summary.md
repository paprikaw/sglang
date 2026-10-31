# Attention 本地效率与 Expert 集中程度

共5页。第二页改为比较本地算子效率：同一总batch B8下，EP2按卡分成B4+B4，PP2两槽微批也各为B4；两边Attention TP1、匹配token数及上下文后，本地Attention算子形状与每次调用的平均耗时应接近。

Expert有结构差异：EP每卡每层持有E/2专家，通过Dispatch/Combine汇聚两卡路由token；PP每卡每个本地层持有E专家，处理当前B4微批。本地专家工作集扩大、平均单专家token减少，可能降低Expert执行效率。示意块只表示E/2与E的比例。未把图形尺寸当作耗时数据。

协议已核验Attention TP1、全局B8、PP两槽、实际本地B4。匹配本地Attention形状是该示例的关键；不将相同总batch单独解释为任何微批策略下的耗时恒等。第一张部署图与后三组数据保持一致，无新增GPU实验。详细来源在备注与sources.json。重建：build.py --with-attention --out <output_directory>。
