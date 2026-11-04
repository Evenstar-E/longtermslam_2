"""Versioned fixed-pool runner; sealed V9/V10 modules remain untouched."""
from pathlib import Path

def main():
    s=Path('scripts/run_material_oriented_surface_control_v10.py').read_text(encoding='utf-8')
    s=s.replace("from inspection_decisions.material_surface_patch_registration_v9 import SurfaceScore", "from inspection_decisions.material_surface_patch_registration_v9 import SurfaceScore\nfrom inspection_decisions.material_inverse_plane_control_v11 import resample_banks,project as inverse_project\nfrom inspection_decisions import material_oriented_surface_control_v10 as oriented\noriginal_setup=setup\ndef setup(cfg,v):\n    data=original_setup(cfg,v);resample_banks(data[7],data[1],cfg);oriented.project=inverse_project;return data")
    s=s.replace("cfg.update(c);return cfg", "cfg.update(c);v11=json.loads(Path('configs/material_inverse_plane_control_v11.json').read_text(encoding='utf-8'));cfg.update(v11);return cfg")
    s=s.replace(".score(T)[:,2:]", ".score(T)")
    s=s.replace("agreement=float(np.max(abs(legacy-scorer(0,0,0).score(A)[:,:2])));assert agreement<1e-12", "agreement=0.0")
    s=s.replace(".score(relative[seeds])[:,2:]", ".score(relative[seeds])").replace(".score(relative[hh])[:,2:]", ".score(relative[hh])")
    s=s.replace("for m in range(2):", "for m in range(4):").replace("origin=np.zeros(2,int)", "origin=np.zeros(4,int)")
    s=s.replace(".score(U[:,0])[:,2:]", ".score(U[:,0])").replace(".score(U[:,1])[:,2:]", ".score(U[:,1])").replace(".score(U[:,1]@inverse(U[:,0]))[:,2:]", ".score(U[:,1]@inverse(U[:,0]))")
    s=s.replace(".reshape(-1,2);", ".reshape(-1,4);").replace("cc=value[best,np.arange(2)].copy()", "cc=value[best,np.arange(4)].copy()")
    s=s.replace("raw_control_max_difference=agreement", "raw_control_max_difference=None,representation_changed=True")
    s=s.replace("orient_banks(banks,frames,cfg)\n    with np.load(parent", "orient_banks(banks,frames,cfg)\n    np.savez_compressed(root/f'{v}_patches.npz',**{f'{e}_{i}_{key}':a[key] for e,row in enumerate(banks) for i,a in enumerate(row) for key in ['patch_xyz','patch_rgb','patch_uv','patch_frame','patch_normal','patch_normal_valid','patch_center_index']})\n    with np.load(parent")
    s=s.replace("cache.append((np.r_[previous['costs'][q,3:5],cc],np.r_[previous['transforms'][q,3:5],TT],origin))", "cache.append((cc,TT,origin))")
    s=s.replace(";assert np.max(abs(values[:,:2]-pg['values'][:,3:5]))<1e-9;best=", ";best=")
    begin=s.index('        if col<2:\n');end=s.index("    save(root/f'{v}_diagnostics.json'",begin)
    s=s[:begin]+"        h=int(best[col]);state=pg['states'][h];TT=np.array([cache[lookup[i,int(j),int(k)]][1][col] for i,(j,k) in enumerate(state)]);cc=np.array([cache[lookup[i,int(j),int(k)]][0][col] for i,(j,k) in enumerate(state)]);np.savez_compressed(root/f'{v}_{m}.npz',target_slots=state,transforms=TT,objectives=cc,search_state=h,triplet_counts=np.zeros(24,int))\n"+s[end:]
    s=s.replace("'configs/material_oriented_surface_control_v10.json','inspection_decisions/material_oriented_surface_control_v10.py'", "'configs/material_oriented_surface_control_v10.json','configs/material_inverse_plane_control_v11.json','inspection_decisions/material_inverse_plane_control_v11.py','inspection_decisions/material_oriented_surface_control_v10.py'")
    s=s.replace("'scripts/check_material_oriented_surface_control_v10.py'", "'scripts/check_material_inverse_plane_control_v11.py','scripts/prepare_material_inverse_plane_control_v11.py'")
    s=s.replace("Standard normal hemisphere compatibility, not novel method.", "Inverse-depth plane support repair, all four scores recomputed on common new patches, not novel theory.")
    s=s.replace("raw_max_difference=float(np.max(abs(new[:,:2]-old))))", "finite=bool(np.isfinite(new).all())))")
    s='\n'.join("        st=time.perf_counter();new=s.score(T);print(json.dumps(dict(public_only=True,poses=len(T),seconds=time.perf_counter()-st,finite=bool(np.isfinite(new).all()))));return" if 'st=time.perf_counter();new=s.score(T)' in line else line for line in s.splitlines())+'\n'
    Path('scripts/run_material_inverse_plane_control_v11.py').write_text(s,encoding='utf-8')

if __name__=='__main__':main()
