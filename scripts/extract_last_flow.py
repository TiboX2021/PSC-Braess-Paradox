import numpy as np
import json

flows = np.genfromtxt("out_v2.csv", delimiter=',')

last_flow = flows[-1]

data = json.dumps(list(last_flow), indent=4, ensure_ascii=False)

f = open("last_flow.json", 'w')
f.write(data)
f.close()