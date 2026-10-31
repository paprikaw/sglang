# EP2 / PP2 部署对比与三组数据

共4页：第1页用4层、2GPU示意EP2每层分专家及逐层Dispatch/Combine；PP2按层划分阶段，并用微批A/B展示流水并行。第2–4页保留原GPU、模型、阶段三组insight，以及每组的端到端延迟和Expert/通信图。

图中箭头表示通信发生的位置；示意层数与时间槽不作为测量数据。数据表和六张数据图与三页版逐字节核验一致。部署图为原生可编辑PowerPoint形状和文字，详细来源及统计口径在备注和sources.json中。

没有新增GPU实验；原三页版及其归档保留。重建：使用build.py --with-deployment --out <output_directory>。PDF和PNG是同一布局的预览。
