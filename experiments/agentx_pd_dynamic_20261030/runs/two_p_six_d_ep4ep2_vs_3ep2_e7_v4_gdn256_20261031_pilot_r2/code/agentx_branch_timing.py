"""Branch-timing sensitivity protocol for the pinned AgentX-derived graphs.

Matches the pinned harness's SPAWN/FORK issue-versus-return dispatch rules.
This does not implement official trajectory sampling, chat rendering or scoring.
Pre-session branches are rejected: the finite-root admission protocol does not
reproduce official once-per-phase background dispatch.
"""
import math

POLICY = 'official_branch_issue_and_return_v1'


def timestamp(value):
    return float(value) if isinstance(value, (int,float)) and math.isfinite(value) else None


def branch_start(branch, conversations):
    start = timestamp(branch.get('start_timestamp_ms'))
    if start is not None:
        return start
    starts = [timestamp(conversations[c]['turns'][0].get('timestamp'))
              for c in branch['child_conversation_ids'] if conversations[c]['turns']]
    starts = [x for x in starts if x is not None]
    return min(starts) if starts else None


def branch_dispatch_rules(graph):
    """Return one dispatch event and delay for every child conversation."""
    conversations = {c['session_id']:c for c in graph['conversations']}
    branches, anchors, owners = {}, {}, {}
    for sid, conv in conversations.items():
        for branch in conv['branches']:
            bid=branch['branch_id']
            assert bid not in branches
            branches[bid]=branch
            if branch.get('dispatch_timing','post')!='post':
                raise ValueError('Pre-session branch timing requires a separate phase protocol')
            for child in branch['child_conversation_ids']:
                assert child in conversations and child not in owners
                owners[child]=sid
        for i,turn in enumerate(conv['turns']):
            for bid in turn['branch_ids']:
                assert bid not in anchors
                anchors[bid]=(sid,i)
    assert branches.keys()==anchors.keys()

    def depth(sid, visited=()):
        assert sid not in visited, 'Cyclic conversation ancestry'
        return 0 if sid not in owners else 1+depth(owners[sid],visited+(sid,))

    rules={}
    for bid, branch in branches.items():
        sid,index=anchors[bid]
        parent=conversations[sid]['turns'][index]
        start=branch_start(branch,conversations)
        pstart=timestamp(parent.get('timestamp'))
        api=timestamp(parent.get('api_time_ms'))
        mode=branch['mode']
        assert mode in ('spawn','fork')
        overlap=(mode=='spawn' and depth(sid)==0 and pstart is not None
                 and api is not None and api>0 and start is not None and start<pstart+api)
        event='issued' if overlap else 'complete'
        origin=pstart if overlap else start if mode=='spawn' else None
        for child in branch['child_conversation_ids']:
            assert conversations[child]['turns'], 'Empty child not supported in finite replay'
            child_start=timestamp(conversations[child]['turns'][0].get('timestamp'))
            delay=0.0 if origin is None or child_start is None else max(0.0,child_start-origin)/1000
            rules[child]=dict(anchor=(sid,index),event=event,delay_s=delay,
                branch_id=bid,branch_start_ms=start,dispatch_origin_ms=origin)
    return rules


def build_vertices(graph,prefix):
    conversations={c['session_id']:c for c in graph['conversations']}
    branches={b['branch_id']:b for c in conversations.values() for b in c['branches']}
    rules=branch_dispatch_rules(graph)
    vertices={}
    for sid,conv in conversations.items():
        for i,turn in enumerate(conv['turns']):
            complete,issued=[],[]
            if i:
                complete.append(((prefix+sid,i-1),max(0,(turn['delay'] or 0)/1000)))
            elif sid in rules:
                rule=rules[sid]
                dependency=((prefix+rule['anchor'][0],rule['anchor'][1]),rule['delay_s'])
                (issued if rule['event']=='issued' else complete).append(dependency)
            for pred in turn.get('replay_predecessors',[]):
                complete.append(((prefix+pred['conversation_id'],pred['turn_index']),0))
            for prerequisite in turn['prerequisites']:
                if prerequisite['kind']!='spawn_join':
                    raise ValueError('Unsupported prerequisite: '+str(prerequisite))
                for child in branches[prerequisite['branch_id']]['child_conversation_ids']:
                    complete.append(((prefix+child,len(conversations[child]['turns'])-1),0))
            vertices[prefix+sid,i]=dict(turn=turn,requirements=complete,
                issue_requirements=issued,graph=graph,namespace=prefix,
                root_id=prefix+graph['id'],sid=prefix+sid)
    for vertex in vertices.values():
        assert all(k in vertices for k,_ in vertex['requirements']+vertex['issue_requirements'])
    resolved=set()
    while len(resolved)<len(vertices):
        ready={key for key,v in vertices.items() if key not in resolved and all(
            k in resolved for k,_ in v['requirements']+v['issue_requirements'])}
        if not ready:
            raise ValueError('Cyclic replay graph')
        resolved.update(ready)
    return vertices


def due_time(vertex,completed,issued):
    if not all(k in completed for k,_ in vertex['requirements']):
        return None
    if not all(k in issued for k,_ in vertex['issue_requirements']):
        return None
    return max([vertex['admitted_at']]
               +[completed[k]+delay for k,delay in vertex['requirements']]
               +[issued[k]+delay for k,delay in vertex['issue_requirements']])
