def parseNL(path):
    f = open(path, 'r')
    names = []
    labels = []
    lines = f.readlines()
    for line in lines:
        if len(line.split()) == 1:
            name = line.split()
            names.append(name[0])
        if len(line.split()) == 2:
            [name, label] = line.split()
            names.append(name)
            labels.append(label)
    if len(labels) == 0:
        return names
    else:
        return names, labels