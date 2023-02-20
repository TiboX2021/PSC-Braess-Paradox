import numpy as np
import json

flows = np.genfromtxt("out1.csv", delimiter=',')

last_flow = flows[-1]

data = json.dumps(list(last_flow), indent=4, ensure_ascii=False)

f = open("deprecated_last_flow.json", 'w')
f.write(data)
f.close()