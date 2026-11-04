import argparse
from scripts.run_evidence_value_pilot_v1 import read


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['status','pilot','audit','confirm']);args=ap.parse_args()
    cfg=read('configs/reinspection_research_chain_v3.json')
    if args.stage=='status':
        print(cfg['status']);print('Completed:',cfg['completed']);print('Missing:',cfg['missing'])
    elif args.stage=='pilot':
        from scripts.run_correspondence_motion_objective_v1 import main as run
        run()
    elif args.stage=='audit':
        from scripts.audit_correspondence_motion_objective_v1 import main as audit
        audit()
    else:raise SystemExit('NOT READY: pilot does not establish independent mechanism gain or literature novelty; confirmation not launched.')


if __name__=='__main__':main()
