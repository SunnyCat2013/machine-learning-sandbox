
sfile = "trainLabels.csv"

dList = {}

with open(sfile, 'r') as inf:
    lines = inf.readlines()
    del lines[0]
    for l in lines:
        l = l.strip().split(',')
        key = l[1]
        if dList.has_key(l[1]):
            dList[key] += 1
        else:
            dList[key] = 1

print "length of keys:", len(dList)

ofile = "street-details.txt"
with open(ofile, 'w') as of:
    count = 0
    keys = dList.keys()
    keys.sort()
    for key in keys:
        print key, ":", dList[key]
        print >>of,"%s: %d"%(key, dList[key])
        count += dList[key]
    print >>of, "length of keys: %d"%len(dList)
    print >>of, "total samples: %d"%count
