import time

last = 0
def inittime():
    global last
    last = time.time()

def timecheck(label, echo=True):
    global last
    now = time.time()
    diff = now - last
    if echo:
        print(label, '%.2f ms' % (diff * 1000))
    last = now
    return diff * 1000

