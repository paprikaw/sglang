"""Editable proposed architecture and matching standalone diagram preview."""
import json
from pathlib import Path

from pptx.enum.shapes import MSO_CONNECTOR
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

from deck_engine import BLUE, TEAL, ORANGE, INK, MUTED, PPI, color


def flow(deck, points, ink=BLUE, dashed=False):
    """Native PowerPoint connectors with matching preview arrows."""
    for index, ((x0, y0), (x1, y1)) in enumerate(zip(points, points[1:])):
        deck._bounds(min(x0, x1), min(y0, y1), max(abs(x1-x0), .01),
                     max(abs(y1-y0), .01), 'connector')
        shape = deck.current.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT, Inches(x0), Inches(y0), Inches(x1), Inches(y1))
        shape.line.color.rgb = color(ink)
        shape.line.width = Pt(1.65 if dashed else 2.0)
        properties = shape.line._get_or_add_ln()
        if dashed:
            dash = OxmlElement('a:prstDash')
            dash.set('val', 'dash')
            properties.append(dash)
        if index == len(points)-2:
            end = OxmlElement('a:tailEnd')
            end.set('type', 'triangle')
            end.set('w', 'med')
            end.set('len', 'med')
            properties.append(end)
        start, stop = (x0*PPI, y0*PPI), (x1*PPI, y1*PPI)
        if dashed:
            distance = ((stop[0]-start[0])**2+(stop[1]-start[1])**2)**.5
            for k in range(0, round(distance), 11):
                lo, hi = k/distance, min(k+6, distance)/distance
                deck.draw.line((start[0]+(stop[0]-start[0])*lo,
                                start[1]+(stop[1]-start[1])*lo,
                                start[0]+(stop[0]-start[0])*hi,
                                start[1]+(stop[1]-start[1])*hi), fill=ink, width=3)
        else:
            deck.draw.line((*start, *stop), fill=ink, width=4)
    (x0, y0), (x1, y1) = points[-2:]
    length = ((x1-x0)**2+(y1-y0)**2)**.5
    dx, dy = (x1-x0)/length, (y1-y0)/length
    tip = (x1*PPI, y1*PPI)
    base = ((x1-dx*.075)*PPI, (y1-dy*.075)*PPI)
    deck.draw.polygon([tip, (base[0]-dy*5, base[1]+dx*5),
                       (base[0]+dy*5, base[1]-dx*5)], fill=ink)


def card(deck, x, y, width, height, heading, accent, heading_size=18):
    deck.rect(x, y, width, height, '#FFFFFF', line='#D2DCEB')
    deck.rect(x+.02, y+.16, .045, height-.32, accent, radius=False)
    deck.text(heading, x+.18, y+.16, width-.36, .34, heading_size, accent, True)


def add_proposed_architecture(deck, out):
    title = 'Proposed architecture: profile, predict, and scale'
    subtitle = 'Choose an initial PP / EP layout, then reuse the predictor to choose how to scale as load grows'
    takeaway = 'One predictor, two decisions: initial placement and PP vs. EP scale-out.'
    notes = (
        'Proposed system architecture, not an implemented or experimentally validated autoscaling system. '
        'The preceding five slides motivate the design. The Profiler collects a small set of representative '
        'Expert GEMM measurements over local expert working set and routed tokens per expert, '
        'Dispatch+Combine timings over payload and communication domain, and PP boundary transfer costs. '
        'Profiles are keyed by model/precision, backend, GPU, topology and phase, rather than treating '
        'model parameter count or GPU count as the only deployment inputs. '
        'The Performance Model uses these profiles to predict latency and throughput for memory-feasible '
        'PP x EP candidates under the available GPU budget and representative Prefill/Decode workload. '
        'A short end-to-end calibration captures the remaining Attention/dense compute and scheduling '
        'effects, including overlap and pipeline bubbles; independent profiled component spans should '
        'not simply be added and equated with measured request latency. Matching Attention execution '
        'shapes can support reuse of its baseline; a changed shape should be recalibrated. '
        'Initial placement selects a candidate with high predicted serving capacity while respecting '
        'memory and the latency target. Hybrid PP x EP layouts can be included in the same candidate set. '
        'At runtime the Autoscaler observes request rate, queue depth, latency and Prefill/Decode mix. '
        'When the current layout approaches its useful capacity, it queries the same model for two '
        'expansion directions: increase the number of pipeline stages, or widen the expert-parallel group. '
        'PP scale-out repartitions layers and retains smaller expert-exchange domains; EP scale-out '
        'repartitions experts in each layer and can cross a new communication boundary. The selected '
        'action must meet the new load and latency target with the available resources; memory, placement '
        'and reconfiguration cost are part of feasibility. Stability controls such as hysteresis can be '
        'added during implementation. Reconfiguration is applied to the serving deployment. Measured '
        'latency and throughput feed back into the predictor, with targeted re-profiling when a new '
        'GPU/domain or local execution shape lies outside the existing profile coverage. '
        'No new measurements or quantitative speedup are asserted by this architecture page.')
    deck.slide(title, subtitle, notes=notes)
    deck.rect(.78, 1.73, 11.78, .40, '#E4EDF9')
    deck.text('Model & memory  ·  GPU topology / available GPUs  ·  Workload & latency target',
              1.00, 1.80, 11.34, .24, 12.0, INK, True, align='center')
    deck.text('01  INITIAL PLACEMENT', .83, 2.30, 3.45, .24, 11.7, MUTED, True)
    deck.text('02  RUNTIME SCALING', .83, 4.56, 3.45, .24, 11.7, MUTED, True)
    top, top_h, bottom, bottom_h = 2.62, 1.65, 4.97, 1.30
    left, mid, right = .78, 4.78, 9.10
    left_w, mid_w, right_w = 3.20, 3.50, 3.46
    card(deck, left, top, left_w, top_h, 'Lightweight Profiler', BLUE, 17.4)
    deck.text('Expert GEMM: tokens per expert\nLocal expert working set\nDispatch + Combine by domain\nPP stage transfer',
              left+.18, 3.19, left_w-.36, .87, 11.5, INK)
    card(deck, mid, top, mid_w, top_h, 'Performance Model', BLUE)
    deck.text('Predict latency and throughput\nfor feasible PP × EP layouts.',
              mid+.18, 3.20, mid_w-.36, .49, 12.0, INK)
    deck.rect(mid+.17, 3.84, mid_w-.34, .27, '#EDF3FF')
    deck.text('Fit memory + meet latency target', mid+.24, 3.88, mid_w-.48, .20,
              10.3, BLUE, True, align='center')
    card(deck, right, top, right_w, top_h, 'PP / EP Deployment', TEAL)
    deck.text('Initial layout chosen by the model', right+.18, 3.20, right_w-.36, .25, 11.5, INK)
    for x, label, fill, ink in [(right+.18, 'PP layout', '#EAF6F3', TEAL),
                                (right+1.89, 'EP layout', '#EDF3FF', BLUE)]:
        deck.rect(x, 3.65, 1.37, .39, fill)
        deck.text(label, x+.05, 3.73, 1.27, .23, 11.8, ink, True, align='center')
    deck.text('or', right+1.59, 3.74, .24, .20, 10.2, MUTED, align='center')
    card(deck, left, bottom, left_w, bottom_h, 'Live Workload', MUTED, 17.4)
    deck.text('Request rate · queue depth\nPrefill / Decode mix · latency',
              left+.18, bottom+.59, left_w-.36, .49, 11.5, INK)
    card(deck, mid, bottom, mid_w, bottom_h, 'Autoscaler', ORANGE)
    deck.text('Compare PP and EP scale-out\nfor the new load and topology.',
              mid+.18, bottom+.59, mid_w-.36, .49, 11.5, INK)
    deck.rect(right, bottom, right_w, bottom_h, '#FFFFFF', line='#D2DCEB')
    for x, heading, body, accent, fill in [
            (right+.12, 'PP scale-out', 'Add pipeline\nstages', TEAL, '#EAF6F3'),
            (right+1.80, 'EP scale-out', 'Widen the\nexpert group', BLUE, '#EDF3FF')]:
        deck.rect(x, bottom+.12, 1.54, 1.06, fill)
        deck.text(heading, x+.06, bottom+.24, 1.42, .25, 12.0, accent, True, align='center')
        deck.text(body, x+.06, bottom+.60, 1.42, .41, 10.6, INK, align='center')
    for y, label1, label2, accent in [(3.47, 'Profile', 'Initialize', BLUE),
                                      (5.64, 'Observe', 'Select', ORANGE)]:
        flow(deck, [(4.04, y), (4.71, y)], accent)
        flow(deck, [(8.34, y), (9.03, y)], accent)
        deck.text(label1, 4.01, y-.30, .73, .20, 9.7, MUTED, align='center')
        deck.text(label2, 8.31, y-.30, .75, .20, 9.7, MUTED, align='center')
    flow(deck, [(6.53, 4.30), (6.53, 4.91)], BLUE)
    deck.text('Reuse model', 6.70, 4.48, 1.40, .21, 10.2, BLUE)
    flow(deck, [(10.83, 4.91), (10.83, 4.30)], ORANGE)
    deck.text('Apply', 11.00, 4.48, 1.02, .21, 10.2, ORANGE)
    flow(deck, [(12.58, 3.47), (12.85, 3.47), (12.85, 2.39),
                (7.65, 2.39), (7.65, 2.59)], MUTED, dashed=True)
    deck.text('Measured latency & throughput', 8.32, 2.18, 3.89, .19, 9.8,
              MUTED, align='center')
    deck.rect(.78, 6.64, 11.78, .40, '#E4EDF9')
    deck.text(takeaway, 1.00, 6.72, 11.30, .25, 12.7, INK, True)
    deck.finish_slide()
    out = Path(out)
    deck.canvas.save(out/'figures/proposed_architecture.png')
    deck.canvas.save(out/'figures/proposed_architecture.pdf', resolution=PPI)
    specification = dict(
        status='proposed design', implemented=False, empirical_speedup_claimed=False,
        profiler=['Expert GEMM vs tokens per expert and local expert working set',
                  'Dispatch+Combine vs payload, phase and communication domain', 'PP stage transfer'],
        performance_model=dict(inputs=['model and memory', 'GPU topology and budget',
                                      'phase-specific workload', 'component profiles', 'latency target'],
                               outputs=['predicted latency', 'predicted throughput', 'initial PP x EP layout'],
                               calibration='Short E2E calibration covers remaining compute and scheduling'),
        autoscaler=dict(observations=['request rate', 'queue depth', 'Prefill/Decode mix', 'latency'],
                        candidates=['PP scale-out: increase pipeline stages', 'EP scale-out: widen expert group'],
                        topology_check='Re-evaluate costs when additional GPUs cross a communication domain',
                        constraints=['memory', 'available GPUs', 'latency target', 'reconfiguration cost']),
        feedback='Measured serving latency and throughput update the predictor', notes=notes)
    (out/'architecture_design.json').write_text(json.dumps(specification, indent=2)+'\n')
    return dict(title=title, subtitle=subtitle, takeaway=takeaway)
