"""Attention placement and the common per-token computation in EP2 and PP2."""
from deck_engine import BLUE, TEAL, ORANGE, INK, MUTED

ATTENTION_FILL='#EDF0F5'
ATTENTION_LINE='#D2D9E5'


def attention_cell(deck,x,y,layer):
    deck.rect(x,y,.91,.72,ATTENTION_FILL,line=ATTENTION_LINE)
    deck.text(f'L{layer}',x+.025,y+.105,.86,.25,14,INK,True,align='center')
    deck.text('Attention',x+.025,y+.425,.86,.21,10.5,MUTED,align='center')


def add_attention_slide(deck,source):
    deck.slide('Attention 算子不变，重点看 Expert 与专家交换',
        '同一模型与 workload · Attention TP=1 · A / B 表示两组等量请求',
        sources=[source],
        notes=('Conceptual four-layer placement diagram for the saved Attention TP1 configurations. '
            'EP2 uses Attention DP2: both GPUs hold all Attention layers; GPU0 owns request group A '
            'and GPU1 group B, including each group\'s request-local KV. Routed experts are split '
            'across both GPUs and token dispatch/combine returns results to the owning Attention rank. '
            'PP2 uses two stages with Attention TP1: GPU0 holds layers1/2 and GPU1 layers3/4; '
            'both request groups pass through both stages as separate microbatches. A request is '
            'therefore evaluated by the same unsharded Attention operator at every model layer. '
            'With equal model, tokens and contexts, model-level Attention work is invariant. '
            'The diagram does not assert exact equality of measured Attention GPU time: batching, '
            'kernel shapes, cache access and scheduling can affect execution efficiency. '
            'The slide identifies the principal expert-placement trade-off, not a complete additive '
            'latency model; pipeline boundary communication and fill/drain shown on slide1 remain '
            'part of the end-to-end deployment comparison. No new timing data are introduced.'))
    for x in [.78,6.82]:
        deck.rect(x,1.76,5.74,3.70)
    deck.text('EP2',1.03,1.98,1.05,.42,23,BLUE,True)
    deck.text('Attention 按请求分组（DP2）',2.16,2.06,4.00,.31,14,BLUE,True)
    deck.text('每卡：全部 Attention 层，处理一组请求',1.03,2.49,5.20,.30,13.5,INK)
    deck.text('PP2',7.07,1.98,1.05,.42,23,TEAL,True)
    deck.text('Attention 随模型层分阶段',8.20,2.06,4.00,.31,14,TEAL,True)
    deck.text('每卡：一半 Attention 层，先后处理 A、B',7.07,2.49,5.20,.30,13.5,INK)
    ep_x=[1.92+i*1.065 for i in range(4)]
    pp_x=[7.96+i*1.065 for i in range(4)]
    for gpu,group,y in [(0,'A 组请求',3.08),(1,'B 组请求',4.05)]:
        deck.text(f'GPU {gpu}',1.03,y+.13,.79,.23,11,INK,True)
        deck.text(group,1.03,y+.43,.83,.20,9.5,MUTED)
        for i,x in enumerate(ep_x):
            attention_cell(deck,x,y,i+1)
    for gpu,y,local in [(0,3.08,{1,2}),(1,4.05,{3,4})]:
        deck.text(f'GPU {gpu}',7.07,y+.13,.79,.23,11,INK,True)
        deck.text('A / B',7.07,y+.43,.83,.20,10,MUTED)
        for i,x in enumerate(pp_x):
            if i+1 in local:
                attention_cell(deck,x,y,i+1)
            else:
                deck.rect(x,y,.91,.72,'#FFFFFF',line='#E9EDF4')
                deck.text('—',x+.03,y+.25,.85,.24,13,'#C9D2E1',align='center')
    deck.text('每卡工作量：全部层 × 半数请求',1.03,4.99,5.20,.27,13,BLUE,True)
    deck.text('每卡工作量：半数层 × 全部请求',7.07,4.99,5.20,.27,13,TEAL,True)

    deck.rect(.78,5.69,11.78,.49,'#E7ECF4')
    deck.text('相同 token / 上下文：每层 Attention 的算子与计算量相同',
        1.01,5.805,11.31,.30,16,INK,True,align='center')

    deck.text('主要权衡',1.03,6.55,1.30,.31,15,INK,True)
    deck.rect(2.65,6.39,3.77,.62,'#E8F0FF')
    deck.text('Expert 本地执行效率',2.82,6.565,3.43,.30,16,BLUE,True,align='center')
    deck.text('↔',6.65,6.50,.56,.40,23,INK,True,align='center')
    deck.rect(7.43,6.39,5.13,.62,'#FFF0DF')
    deck.text('Dispatch / Combine 专家交换',7.60,6.565,4.79,.30,16,ORANGE,True,align='center')
    deck.finish_slide()
