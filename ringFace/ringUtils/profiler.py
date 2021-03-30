import time

runtimes = {}
fullStart = time.time()

def addRuntime(codeBlock, runtime):
    if codeBlock in runtimes:
        runtimes[codeBlock] = runtimes[codeBlock] + runtime
    else:
        runtimes[codeBlock] = runtime

def printRuntimes():
    addRuntime("fullTime", time.time() - fullStart)

    for codeBlock in runtimes:
        percent = runtimes[codeBlock] * 100 / runtimes["fullTime"]
        print(f"{codeBlock}:    {percent} %, {runtimes[codeBlock]} sec")