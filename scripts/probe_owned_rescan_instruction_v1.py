"""Linux-only: briefly inspect RIP of an explicitly verified owned Rescan process."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import time

pid=int(sys.argv[1]);out=Path(sys.argv[2]);cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
assert b'/vlm&lifelongslam/third_party/Rescan_memory_port_v1/bin/pose_proposal' in cmd[0]
assert any(b'rescan_protected_archive_v1/scene_a/prediction.rsdb' in v for v in cmd)
lib=ctypes.CDLL(None,use_errno=True);lib.ptrace.restype=ctypes.c_long
rows=[]
for _ in range(3):
    if lib.ptrace(16,pid,None,None)!=0:raise OSError(ctypes.get_errno(),'attach failed')
    try:
        os.waitpid(pid,0);regs=(ctypes.c_ulonglong*27)()
        if lib.ptrace(12,pid,None,ctypes.byref(regs))!=0:raise OSError(ctypes.get_errno(),'getregs failed')
        rip=regs[16];mapping=next(x for x in Path(f'/proc/{pid}/maps').read_text().splitlines() if int(x.split()[0].split('-')[0],16)<=rip<int(x.split()[0].split('-')[1],16))
        t=mapping.split();base=int(t[0].split('-')[0],16)-int(t[2],16);path=t[-1]
        # addr2line only resolves symbols; no memory modification.
        symbol=subprocess.run(['addr2line','-f','-C','-e',path,hex(rip-base)],capture_output=True,text=True)
        stack_symbols=[]
        start,end=[int(x,16) for x in t[0].split('-')]
        with open(f'/proc/{pid}/mem','rb',buffering=0) as memory:
            stack_map=next(x for x in Path(f'/proc/{pid}/maps').read_text().splitlines() if int(x.split()[0].split('-')[0],16)<=regs[19]<int(x.split()[0].split('-')[1],16))
            stack_end=int(stack_map.split()[0].split('-')[1],16)
            memory.seek(regs[19]);stack=memory.read(min(16384,stack_end-regs[19]))
        addresses=[]
        for k,value in enumerate(struct.unpack('<'+'Q'*(len(stack)//8),stack)):
            if start<=value<end and value not in addresses:
                addresses.append(value)
                result=subprocess.run(['addr2line','-f','-C','-e',path,hex(value-base)],capture_output=True,text=True)
                stack_symbols.append({'stack_word':k,'address':hex(value-base),'symbol':result.stdout})
        rows.append({'rip':hex(rip),'mapping':mapping,'file_offset':hex(rip-base),'symbol':symbol.stdout,
                     'possible_return_addresses_not_unwound_stack':stack_symbols})
    finally:
        lib.ptrace(17,pid,None,None)
    time.sleep(.1)
out.write_text(json.dumps({'pid':pid,'command':[x.decode() for x in cmd if x],'samples':rows},indent=2))
print(json.dumps(rows,indent=2))
