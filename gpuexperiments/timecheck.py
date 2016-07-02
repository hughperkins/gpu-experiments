import time

last = 0
def inittime():
    global last
    last = time.time()

def timecheck(label):
    global last
    now = time.time()
    diff = now - last
    print(label, '%.2f ms' % (diff * 1000))
    last = now
    return diff * 1000

