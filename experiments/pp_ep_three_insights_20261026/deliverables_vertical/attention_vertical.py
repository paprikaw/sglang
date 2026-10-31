"""Both EP layer execution and PP stage progression read from top to bottom."""
from deck_engine import BLUE, TEAL, ORANGE, INK, MUTED
from deployment_diagram import arrow


def operator(deck,x,y,label,expert=False,ep=False):
    fill=('#EAF1FF'if ep else'#E5F5F0')if expert else'#EDF0F5'
    border=('#AFC7F4'if ep else'#ABD8CB')if expert else'#D2D9E5'
    ink=(BLUE if ep else TEAL)if expert else INK
    deck.rect(x,y,2.05,.40,fill,line=border)
    deck.text(label,x+.04,y+.082,1.97,.25,12.5,ink,True,align='center')


def down(deck,center,y,height,fill=MUTED):
    arrow(deck,center-.045,y,.09,height,direction='down',fill=fill)


def add_attention_slide(deck,source):
    deck.slide('Attention 效率应接近，Expert 效率随部署改变',
        '统一从上到下读图 · 2 层示意 · 总 B=8 / 本地 B=4 · Attention TP=1 · E 为每层专家数',
        sources=[source],notes=(
            'Vertical layout revision. Both panels progress from top to bottom through a conceptual two-layer '
            'model. EP places the Attention and E/2 expert shard of each layer on each GPU. GPU columns '
            'within the left panel denote parallel rank placement, never a stage-to-stage progression. '
            'Each EP layer reads Attention -> Dispatch -> local experts -> Combine, then the next layer. '
            'PP stage1/GPU0 is above stage2/GPU1; each contains Attention -> E local experts. The downward '
            'orange arrow between stages denotes activation transfer. All operator and stage progression '
            'arrows point down. Boxes and arrows show logical order, not measured time or pipeline slot lengths. '
            'The saved protocol confirms Attention TP1, EP Attention DP2, global batch8, two PP B4 slots '
            'and actual Attention local batch4 in both layouts. PP passes each B4 microbatch through both '
            'stages, while EP splits requests across Attention ranks. Attention blocks use equal dimensions '
            'and matched local B4 labels to express comparable local operator shapes and expected average '
            'time per layer-call at matched token/context sizes. Equal total batch alone would not ensure '
            'matched local shapes for arbitrary PP microbatching. Each local PP layer has E experts versus '
            'E/2 in EP; PP sees current-microbatch routed tokens, while EP dispatch pools tokens from both '
            'ranks. PP can have larger expert working sets and fewer tokens per expert, reducing reuse. '
            'Measured data remain on slides3-5; no new GPU timing or exact Attention latency equality claimed.'))
    for x in [.78,6.82]:deck.rect(x,1.76,5.74,4.88)
    deck.text('EP2',1.03,1.98,1.05,.42,23,BLUE,True)
    deck.text('每层两卡协同，向下逐层执行',2.16,2.06,3.95,.30,14,BLUE,True)
    deck.text('PP2',7.07,1.98,1.05,.42,23,TEAL,True)
    deck.text('阶段 1 在上，阶段 2 在下',8.20,2.06,3.95,.30,14,TEAL,True)

    # EP: fixed GPU columns, and the same layer's operators arranged vertically.
    gpu_x=[1.55,4.08]
    centers=[x+1.025 for x in gpu_x]
    for gpu,x in enumerate(gpu_x):
        deck.text(f'GPU {gpu}',x,2.57,2.05,.23,11,INK,True,align='center')
    for layer,attn_y in [(1,2.99),(2,5.10)]:
        deck.text(f'L{layer}',1.01,attn_y+.11,.43,.24,12,BLUE,True)
        dispatch_y=attn_y+.48
        expert_y=attn_y+.77
        combine_y=attn_y+1.25
        for x in gpu_x:
            operator(deck,x,attn_y,'Attention · B4')
            operator(deck,x,expert_y,'Expert · E/2',True,True)
        deck.rect(1.55,dispatch_y,4.58,.21,'#FFF0DF')
        deck.text('Dispatch：跨卡分发 token',1.62,dispatch_y+.023,4.44,.19,10.2,ORANGE,True,align='center')
        deck.rect(1.55,combine_y,4.58,.21,'#FFF0DF')
        deck.text('Combine：跨卡归并结果',1.62,combine_y+.023,4.44,.19,10.2,ORANGE,True,align='center')
        for center in centers:
            down(deck,center,attn_y+.415,.055)
            down(deck,center,dispatch_y+.223,.055)
            down(deck,center,expert_y+.415,.055)
    for center in centers:down(deck,center,4.50,.50)
    deck.text('下一层',3.37,4.69,.94,.23,11,MUTED,align='center')

    # PP: both the stage transition and local Attention -> Expert direction are vertical.
    for gpu,stage,layer,top,attn_y,expert_y in [
        (0,1,1,2.55,2.99,3.76),
        (1,2,2,4.63,5.10,5.87),
    ]:
        deck.rect(7.07,top,5.24,1.74,'#F6FBF9',line='#CAE6DC')
        deck.text(f'GPU {gpu} · 阶段 {stage}（L{layer}）',7.29,top+.12,4.80,.25,12.5,TEAL,True)
        operator(deck,8.665,attn_y,'Attention · B4')
        down(deck,9.69,attn_y+.45,.25)
        operator(deck,8.665,expert_y,'Expert · E',True,False)
    down(deck,9.69,4.34,.24,ORANGE)
    deck.text('阶段边界传激活',9.99,4.37,2.00,.22,10.5,ORANGE,True)
    deck.text('本地专家更多，单专家 token 复用更低',7.07,6.43,5.24,.20,10.5,MUTED)

    deck.rect(.78,6.76,11.78,.33,'#E4EDF9')
    deck.text('重点比较：Expert 执行效率的变化    ↔    Dispatch / Combine 通信代价',
        1.00,6.815,11.30,.25,13.5,INK,True)
    deck.finish_slide()
