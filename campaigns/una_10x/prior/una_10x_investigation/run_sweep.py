from pathlib import Path
import json
import probe
root=Path(__file__).resolve().parent
probe.tests()
cells=[('tiny',10,26,14,4.),('search_dominated',92,2560,512,20.),('local_medium',256,65536,512,4.),('local_large',512,262144,512,4.)]
for name,side,D,O,r in cells:
    for H in (1,4):
        probe.benchmark(side,D,O,r,H,5,root/'raw'/f'{name}_H{H}.json')
