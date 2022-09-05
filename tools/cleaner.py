import sys

filename = sys.argv[1]
number = sys.argv[2]

ranking = {}
with open(filename) as f:
        lines = f.readlines()

for line in lines:
        s = line.strip().split(": ")
        ranking[s[0]] = float(s[1])

ranking = dict(sorted(ranking.items(), key=lambda item: item[1]))

for i,key in enumerate(ranking):
        if i >= int(number):
                break
        print(key)
