"""Matched local Attention calls versus the changed local Expert working set."""
from deck_engine import BLUE, TEAL, ORANGE, INK, MUTED

ATTENTION_FILL='#EDF0F5'
ATTENTION_LINE='#D2D9E5'


def attention_cell(deck,x,label):
    deck.text(label,x,2.56,2.26,.24,11,INK,True,align='center')
    deck.rect(x,2.93,2.26,.65,ATTENTION_FILL,line=ATTENTION_LINE)
    deck.text('Attention',x+.04,3.015,2.18,.27,15,INK,True,align='center')
    deck.text('本地 batch = 4',x+.04,3.33,2.18,.20,10.5,MUTED,align='center')


def expert_set(deck,x,sharded,fill):
    deck.rect(x,4.91,2.26,1.13,'#F7F9FD',line='#D9E2F0')
    deck.text('本地 E / 2 专家'if sharded else'本地 E 专家',
        x+.05,5.025,2.16,.25,12.5,fill,True,align='center')
    # Tiles illustrate the local expert-set ratio, not a measured expert count.
    for row in range(1 if sharded else 2):
        y=5.58 if sharded else 5.405+row*.28
        for col in range(4):
            deck.rect(x+.18+col*.49,y,.42,.20,fill)


def add_attention_slide(deck,source):
    deck.slide('Attention 效率应接近，Expert 效率随部署改变',
        '同一 workload、总 batch 相同 · B=8，EP2 每卡 B=4，PP2 两槽微批各 B=4 · Attention TP=1 · E 为每层专家数',
        sources=[source],
        notes=('This slide illustrates local kernel efficiency rather than only model-level Attention FLOPs. '
            'Use the saved B8 Decode protocol: Attention TP1, EP Attention DP2, PP two slots with max '
            'microbatch4. The saved profiles explicitly verify actual_local_batch4 in both layouts. '
            'EP GPU0/1 execute an unsharded Attention operator for the same layer on different request groups. '
            'PP GPU0/1 execute unsharded Attention operators in their respective stages on different '
            'microbatches. At matched token/context sizes and the same Attention implementation, these '
            'local operator shapes are comparable, so mean time per Attention layer-call should be close. '
            'Equal total batch alone does not establish this for arbitrary PP microbatching; this explicit '
            'two-slot example supplies the matched local batch. This is not an equality claim for '
            'complete per-rank Attention totals or call counts. Expert placement changes: each EP GPU '
            'holds E/2 experts for a layer; each PP GPU holds E experts for each of its local layers. '
            'EP exchanges routed tokens from B8 before expert execution; PP experts see the current '
            'local B4 microbatch. Averaged over experts at the same routing fanout, this gives approximately '
            'twice the tokens per EP expert. PP can thus have a larger local expert weight set and smaller '
            'per-expert GEMMs, reducing reuse and execution efficiency, while avoiding per-layer cross-GPU '
            'dispatch/combine. Tiles are conceptual E/2 versus E, not physical expert counts or measured '
            'durations. The drawing illustrates the deployment intuition; measured results follow on '
            'slides3-5. Pipeline communication and scheduling remain part of E2E and slide1. No new GPU timing.'))
    for x in [.78,6.82]:
        deck.rect(x,1.76,5.74,4.88)
    deck.text('EP2',1.03,1.98,1.05,.42,23,BLUE,True)
    deck.text('每层分专家',2.16,2.06,3.95,.30,15,BLUE,True)
    deck.text('PP2',7.07,1.98,1.05,.42,23,TEAL,True)
    deck.text('按层分阶段',8.20,2.06,3.95,.30,15,TEAL,True)
    for x,label in [(1.13,'GPU 0 · 同一层'),(3.94,'GPU 1 · 同一层'),
                    (7.17,'GPU 0 · 阶段 0'),(9.98,'GPU 1 · 阶段 1')]:
        attention_cell(deck,x,label)

    deck.rect(.78,3.79,11.78,.47,'#E7ECF4')
    deck.text('Attention：相同的本地 token 数与算子形状 → 平均耗时应接近',
        1.00,3.905,11.34,.29,15.5,INK,True,align='center')
    deck.text('跨卡 Dispatch / Combine，汇聚两卡 token',1.03,4.49,5.20,.28,13.2,ORANGE,True)
    deck.text('专家本地执行，处理当前微批 token',7.07,4.49,5.20,.28,13.2,TEAL,True)
    for x in [1.13,3.94]:expert_set(deck,x,True,BLUE)
    for x in [7.17,9.98]:expert_set(deck,x,False,TEAL)
    deck.text('本地专家更少，单专家 tokens 更多',1.03,6.17,5.20,.25,13,BLUE,True)
    deck.text('工作集更小，单专家 token 复用更高',1.03,6.435,5.20,.20,10.5,MUTED)
    deck.text('本地专家更多，单专家 tokens 更少',7.07,6.17,5.20,.25,13,TEAL,True)
    deck.text('工作集更大、复用更低 → Expert 效率可能下降',7.07,6.435,5.20,.20,10.5,MUTED)
    deck.rect(.78,6.76,11.78,.33,'#E4EDF9')
    deck.text('重点比较：Expert 执行效率的变化    ↔    Dispatch / Combine 通信代价',
        1.00,6.815,11.30,.25,13.5,INK,True)
    deck.finish_slide()
