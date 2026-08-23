import os
import sys, json
from pathlib import Path
sys.path.insert(0,'/home/ays57/bai')
sys.path.insert(0,'/home/ays57/bai/offline_learning')
sys.path.insert(0,'/home/ays57/bai/offline_learning/scripts')
import offline_learning.curated_plan as cp
SP = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'progs'))
cp.PROGRAMS = SP
import game_profile as gp
gp.PROGRAMS = SP
sys.argv = ['game_profile','--game','rink','--json',os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rink_profile.json')]
gp.main()
