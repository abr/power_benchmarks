import random


def sample(data):
    for x in data:
        yield random.choice(data)


def bootstrapci(data, func, n=3000, p=0.95):
    index = int(n*(1-p)/2)
    r = [func(list(sample(data))) for i in range(n)]
    r.sort()
    return r[index], r[-index]


def mean(x):
    return sum(x)/len(x)
