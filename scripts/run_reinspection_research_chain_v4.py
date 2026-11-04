import argparse
from scripts.run_evidence_value_pilot_v1 import read


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['status','pilot','audit','summary','confirm']);args=ap.parse_args()
    cfg=read('configs/reinspection_research_chain_v4.json')
    if args.stage=='status':
        print(cfg['status']);print('Completed:',cfg['completed']);print('Missing:',cfg['missing'])
    elif args.stage=='pilot':
        from scripts.run_partial_surface_observation_v1 import main as run
        run()
    elif args.stage=='audit':
        from scripts.audit_partial_surface_observation_v1 import main as audit
        audit()
    elif args.stage=='summary':
        from scripts.summarize_partial_surface_observation_v1 import main as summarize
        summarize()
    else:raise SystemExit('NOT READY: normal-only candidate failed task utility; no confirmation or sample expansion launched.')


if __name__=='__main__':main()
