"""Use cgroup headroom when deployed in a NAS container."""
from pathlib import Path
import shutil
import psutil

def cgroup_memory(root=Path('/sys/fs/cgroup')):
    layouts=[('memory.max','memory.current'),('memory/memory.limit_in_bytes','memory/memory.usage_in_bytes'),('memory.limit_in_bytes','memory.usage_in_bytes')]
    for maximum,current in layouts:
        try:
            raw=(root/maximum).read_text().strip()
            if raw=='max':continue
            limit=int(raw);used=int((root/current).read_text().strip())
            if 0<limit<2**60:return limit,max(0,used)
        except (OSError,ValueError):continue
    return None,None


def observe(root,cgroup_root=Path('/sys/fs/cgroup')):
    memory=psutil.virtual_memory();available=memory.available;scope='host';limit=None
    limit,used=cgroup_memory(cgroup_root)
    if limit is not None:available=min(available,max(0,limit-used));scope='container'
    return {'availableRamBytes':available,'processRssBytes':psutil.Process().memory_info().rss,
        'freeDiskBytes':shutil.disk_usage(root).free,'memoryLimitBytes':limit,'totalRamBytes':memory.total,'observationScope':scope}


def budget(store):
    """NAS deployment envelope, bounded by the machine/container actually running."""
    observed=observe(store.root);policy=store.get('setting','resourcePolicy',{})
    target=policy.get('nasRamBytes',32*1024**3)
    reserve=policy.get('systemReserveBytes',8*1024**3)
    total=min(target,observed['totalRamBytes'],observed.get('memoryLimitBytes') or target)
    # A smaller test/container environment still needs its own non-model headroom.
    effective_reserve=min(reserve,max(512*1024**2,total//4))
    ceiling=max(0,total-effective_reserve)
    if policy.get('mode')=='fixed' and policy.get('maxRssBytes') is not None:ceiling=min(ceiling,policy['maxRssBytes'])
    return {**observed,'nasRamBytes':target,'systemReserveBytes':effective_reserve,'maxRssBytes':ceiling,
        'mode':policy.get('mode','adaptive'),'targetDevice':'DS923+','nasMeasured':False}


def configure(store,body,expected=None):
    from LLM.core import require
    old=store.get('setting','resourcePolicy',{})
    value={**old,**body}
    require(value.get('mode','adaptive') in ('adaptive','fixed'),'INVALID_RESOURCE_POLICY')
    for field,default in [('nasRamBytes',32*1024**3),('systemReserveBytes',8*1024**3)]:
        value.setdefault(field,default)
        require(isinstance(value[field],int) and not isinstance(value[field],bool) and value[field]>0,'INVALID_RESOURCE_POLICY',field=field)
    require(value['nasRamBytes']<=32*1024**3 and value['systemReserveBytes']<value['nasRamBytes'],'INVALID_NAS_ENVELOPE')
    if value.get('mode')=='fixed':require(isinstance(value.get('maxRssBytes'),int) and 0<value['maxRssBytes']<=value['nasRamBytes']-value['systemReserveBytes'],'INVALID_RESOURCE_POLICY')
    return store.put('setting',value,'resourcePolicy',expected=expected)
