"""Record installed packages, build configuration, and fetched transitive revisions."""
import hashlib,json,pathlib,platform,subprocess,sys
project=pathlib.Path(sys.argv[1]);ws=pathlib.Path('/home/tong/uav_khronos_ros1_ws_20261021')
def command(args):
    p=subprocess.run(args,capture_output=True,text=True)
    return {'argv':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
records=[command(['dpkg-query','-W','-f=${binary:Package}\t${Version}\n']),
         command(['cmake','--version']),command(['g++','--version'])]
repos=[]
for git in ws.glob('build/**/.git'):
    root=git.parent
    repos.append({'path':str(root),'head':command(['git','-C',str(root),'rev-parse','HEAD']),
                  'origin':command(['git','-C',str(root),'remote','get-url','origin'])})
record={'platform':platform.platform(),'python':sys.version,'commands':records,'transitive_git_repositories':repos,
 'catkin_config':(ws/'.catkin_tools/profiles/default/config.yaml').read_text(),
 'root_source_revision':'4e484c4879a76a37e4bf6304b5c58132dc11b279',
 'native_package_xml_sha256':hashlib.sha256((ws/'src/khronos/khronos_ros/package.xml').read_bytes()).hexdigest()}
out=project/'results/khronos_reproduction_v1/environment.json'
with out.open('x') as f:json.dump(record,f,indent=2)
print('Recorded packages and',len(repos),'transitive Git repositories')
