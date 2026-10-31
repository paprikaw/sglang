"""Native editable EP2/PP2 deployment diagram, with a matching PDF preview."""
from PIL import Image
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches

from deck_engine import BLUE, TEAL, ORANGE, INK, MUTED, PPI, color


def arrow(deck, x, y, width, height, direction='both', fill=ORANGE):
    """Vertical exchange arrow; the PPT and preview use the same polygon."""
    deck._bounds(x, y, width, height, 'communication_arrow')
    kind = MSO_SHAPE.UP_DOWN_ARROW if direction == 'both' else MSO_SHAPE.DOWN_ARROW
    shape = deck.current.shapes.add_shape(kind, Inches(x), Inches(y), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color(fill)
    shape.line.fill.background()
    head = min(height * .22, width * .65)
    stem = width * .30
    c = x + width / 2
    if direction == 'both':
        points = [(c,y),(x+width,y+head),(c+stem/2,y+head),
            (c+stem/2,y+height-head),(x+width,y+height-head),(c,y+height),
            (x,y+height-head),(c-stem/2,y+height-head),
            (c-stem/2,y+head),(x,y+head)]
    else:
        points = [(c-stem/2,y),(c+stem/2,y),(c+stem/2,y+height-head),
            (x+width,y+height-head),(c,y+height),(x,y+height-head),
            (c-stem/2,y+height-head)]
    deck.draw.polygon([(round(px*PPI),round(py*PPI)) for px,py in points], fill=fill)


def layer(deck, x, y, width, label, experts, fill, line):
    deck.rect(x,y,width,.64,fill,line=line)
    deck.text(label,x+.03,y+.065,width-.06,.26,14,INK,True,align='center')
    deck.text(experts,x+.03,y+.37,width-.06,.20,10.5,MUTED,align='center')


def add_deployment_slide(deck, source):
    deck.slide('EP2 分专家、逐层交换；PP2 分层、阶段传输',
        '固定 2 GPU · 用 4 层示意部署与执行 · A / B 表示两份微批',
        sources=[source],
        notes=('Conceptual four-layer example, not a measured execution trace. EP2 partitions the routed experts '
            'of every layer over both GPUs; both GPUs participate in each successive layer of a given forward. '
            'Orange double arrows denote dispatch and combine exchange at every routed MoE layer. '
            'PP2 assigns layers L1/L2 to GPU0 and L3/L4 to GPU1, with the full expert set for each local layer. '
            'Forward activations cross the stage boundary. The A/B microbatch diagram illustrates the middle '
            'pipeline slot: GPU0 handles B at stage0 while GPU1 handles A at stage1. The EP layer sequence '
            'describes one microbatch, not a general prohibition on request concurrency or overlapped kernels. '
            'Arrows are schematic communication locations, not measured byte counts or time durations. '
            'Attention TP1 / DP2 EP versus Attention TP1 PP follows the saved deployment description.'))
    for x in [.78,6.82]:
        deck.rect(x,1.76,5.74,4.82)
    deck.text('EP2',1.03,1.98,1.05,.42,23,BLUE,True)
    deck.text('每层分专家',2.16,2.055,3.85,.30,15,BLUE,True)
    deck.text('每卡持有全部层的半数专家',1.03,2.47,5.20,.30,14,INK)
    deck.text('PP2',7.07,1.98,1.05,.42,23,TEAL,True)
    deck.text('按层分阶段',8.20,2.055,3.85,.30,15,TEAL,True)
    deck.text('每卡持有一半层、所分层的全部专家',7.07,2.47,5.20,.30,14,INK)

    # EP: each column is one layer, distributed across the two GPUs.
    ep_x = [1.92+i*1.065 for i in range(4)]
    for gpu,y in [(0,3.00),(1,4.15)]:
        deck.text(f'GPU {gpu}',1.03,y+.21,.78,.25,11,INK,True)
        for i,x in enumerate(ep_x):
            layer(deck,x,y,.91,f'L{i+1}','½ 专家','#EDF3FF','#ADC4F4')
    for x in ep_x:
        arrow(deck,x+.38,3.71,.15,.36)
    deck.text('每层都要 Dispatch + Combine',1.03,4.91,5.20,.30,13.5,ORANGE,True)

    # PP: the two stage-local layers contain the complete expert sets.
    for gpu,y,start in [(0,3.00,1),(1,4.15,3)]:
        deck.text(f'GPU {gpu}',7.07,y+.21,.78,.25,11,INK,True)
        for i,x in enumerate([8.01,10.02]):
            layer(deck,x,y,1.80,f'L{start+i}','全部专家','#E9F7F3','#A2D7CC')
    arrow(deck,9.79,3.71,.16,.36,direction='down')
    deck.text('传激活',10.13,3.77,1.44,.24,11.5,ORANGE,True)
    deck.text('跨卡通信集中在阶段边界',7.07,4.91,5.20,.30,13.5,ORANGE,True)

    deck.text('同一微批：两卡共同逐层执行',1.03,5.37,5.20,.30,13.5,INK,True)
    for i,x in enumerate(ep_x):
        deck.rect(x,5.86,.91,.39,BLUE)
        deck.text(f'L{i+1}',x+.04,5.915,.83,.25,13,'#FFFFFF',True,align='center')
        if i<3:
            deck.text('→',x+.935,5.91,.12,.24,12,MUTED,align='center')
    deck.text('当前层完成 → 进入下一层',1.92,6.33,4.20,.21,10.5,MUTED,align='center')

    deck.text('不同微批：两个阶段可以并行',7.07,5.37,5.20,.30,13.5,INK,True)
    time_x=[8.04,9.36,10.68]
    # Highlight the slot in which the two GPUs execute different layers concurrently.
    deck.rect(9.30,5.72,1.30,.79,'#FFF2E3',line='#EDC597')
    for i,x in enumerate(time_x):
        deck.text(f't{i+1}',x,5.64,1.18,.18,9,MUTED,align='center')
    for gpu,y in [(0,5.88),(1,6.22)]:
        deck.text(f'GPU {gpu}',7.07,y+.015,.80,.20,10,INK,True)
    for x,y,label,fill in [
        (time_x[0],5.88,'A · L1–2',BLUE),
        (time_x[1],5.88,'B · L1–2',TEAL),
        (time_x[1],6.22,'A · L3–4',BLUE),
        (time_x[2],6.22,'B · L3–4',TEAL),
    ]:
        deck.rect(x,y,1.18,.26,fill)
        deck.text(label,x+.015,y+.032,1.15,.20,10,'#FFFFFF',True,align='center')

    deck.rect(.78,6.72,11.78,.34,'#E4EDF9')
    deck.text('EP2：每层付出专家交换代价    ｜    PP2：将跨卡通信集中到阶段边界',
        1.00,6.775,11.30,.25,13.5,INK,True)
    deck.finish_slide()


def contact_sheet(deck, path):
    sheet=Image.new('RGB',(1600,450*((len(deck.pages)+1)//2)),'#DDE5F0')
    for i,page in enumerate(deck.pages):
        sheet.paste(page.resize((800,450),Image.Resampling.LANCZOS),((i%2)*800,(i//2)*450))
    sheet.save(path)
