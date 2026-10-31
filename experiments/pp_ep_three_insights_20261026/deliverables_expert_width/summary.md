# 一张部署图与三组数据

共4页：统一上下结构的EP2/PP2部署示意 → GPU数据 → 模型数据 → 阶段数据。

原第二页替换原第一页，移除重复示意。PP2每个本地层的Expert长条表示该层全部E个专家，宽度为EP2每卡E/2专家块的2倍；Attention块维持同样宽度。Expert长度表示专家数量，流程图不编码耗时。三组数据表和六张数据图保留，三张数据页按四页版页码核验一致。

PPT文字和示意图原生可编辑；未启动新GPU实验。来源、配置和统计口径保留在备注、sources.json和profile_data中。重建：build.py --replace-intro --out <output_directory>。
