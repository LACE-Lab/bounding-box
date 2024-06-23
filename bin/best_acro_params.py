import argparse
import pathlib
import glob
import collections
import sys

def getFinalPerformance(fileGlob):
    files = glob.glob(fileGlob)
    if len(files) == 0:
        print("Could not find files matching: " + fileGlob, file=sys.stderr)
        return None
    else:
        total = 0
        count = 0
        for f in files:
            fin = open(f, "r")
            q = collections.deque(fin, 101)
            if len(q) < 101:
                print(f + " only has " + str(len(q)) + " lines!", file=sys.stderr)
            else:
                q.popleft()
                for line in q:
                    total += float(line.split()[5])
                    count += 1
            fin.close()
        return total/count

def getBestParams(filePrefix, benchmark, qGlob, useTemp):
    stepSizes = [".a5e-1", ".a2e-1", ".a1e-1", ".a5e-2"]
    if useTemp:
        temperatures = [".m1e2", ".m1e1", ".m1", ".m1e-1", ".m1e-2"]
    else:
        temperatures = [""]

    qFiles = glob.glob(qGlob)
    qFiles.sort()
    
    bestParams = None
    bestScore = float("-inf")
    bestBadParams = None
    bestBadScore = float("-inf")
    for a in stepSizes:
        for m in temperatures:
            fileGlob = filePrefix + a + m + ".t*.result"
            finalScore = getFinalPerformance(fileGlob)
            if finalScore != None:
                print(a.replace(".", ""), m.replace(".", ""), finalScore, benchmark, file=sys.stderr)
                if finalScore > bestBadScore:
                    bestBadParams = (a, m)
                    bestBadScore = finalScore
                if finalScore >= benchmark:
                    scoreSum = 0
                    files = glob.glob(fileGlob)
                    files.sort()
                    for i in range(len(files)):
                        fin = open(files[i], "r")
                        lines = fin.readlines()
                        qin = open(qFiles[i], "r")
                        qlines = qin.readlines()

                        q = collections.deque(fin, 100)
                        qq = collections.deque(fin, 100)
                        for j in range(1, 101):                            
                            q.append(float(lines[j].split()[5]))
                            qq.append(float(qlines[j].split()[5]))
                        idx = 101
                        x = int(lines[idx].split()[0])
                        qidx = 101
                        qx = int(qlines[idx].split()[0])
                        while idx < len(lines) and qidx < len(qlines):
                            scoreSum += sum(q) - sum(qq)
                            if x < qx:
                                q.append(float(lines[idx].split()[5]))
                                x = int(lines[idx].split()[0])
                                idx += 1
                            elif x > qx:
                                qq.append(float(qlines[qidx].split()[5]))
                                qx = int(qlines[qidx].split()[0])
                                qidx += 1
                            else:
                                q.append(float(lines[idx].split()[5]))
                                x = int(lines[idx].split()[0])
                                idx += 1
                                qq.append(float(qlines[qidx].split()[5]))
                                qx = int(qlines[qidx].split()[0])
                                qidx += 1
                    if scoreSum > bestScore:
                        bestScore = scoreSum
                        bestParams = (a, m)
                        
                    print(a.replace(".", ""), m.replace(".", ""), scoreSum/len(files), file=sys.stderr)                    
    if bestParams == None:
        return bestBadParams
    else:
        return bestParams
    
def main():
    parser = argparse.ArgumentParser(description="Prints the best configurations from an Acrobot parameter sweep.")

    parser.add_argument("--path", metavar="PATH", type=str, default="./", help="path where result files are found")

    args = parser.parse_args()

    path = args.path

    print("Looking for result files in " + str(pathlib.PurePath(path)), file=sys.stderr)
    
    leafMaxes = ["100"]
    hiddenSizes = ["8"]
    sampleSizes = ["40"]
    nnStepSizes = ["1e-3"]
    batchSizes = ["4"]    

    numFiles = 0

    for g in ["A", "AD"]:
        benchmark = float("-inf")
        bestQParam = None
        for a in [".a5e-1", ".a2e-1", ".a1e-1", ".a5e-2"]:
            glob = str(pathlib.PurePath(path).joinpath(g + ".pQ" + a + ".t*result"))
            score = getFinalPerformance(glob)
            if score > benchmark:
                benchmark = score
                bestQParam = a
        qGlob = str(pathlib.PurePath(path).joinpath(g + ".pQ" + bestQParam + ".t*result"))
        
        # Oracle/Benchmark
        for p in ["Q", "P"]:
            prefix = str(pathlib.PurePath(path).joinpath(g + ".p" + p))
            print(prefix, file=sys.stderr)
            bestParams = getBestParams(prefix, benchmark, qGlob, False)
            print(prefix + bestParams[0] + bestParams[1] + ".t*.result")

        # Decision Trees
        for l in leafMaxes:
            for p in ["E", "S"]:
                prefix = str(pathlib.PurePath(path).joinpath(g + ".FIRT.l" + l + ".p" + p))
                print(prefix, file=sys.stderr)
                bestParams = getBestParams(prefix, benchmark, qGlob, False)
                print(prefix + bestParams[0] + bestParams[1] + ".t*.result")        

            for k in sampleSizes:
                for p in ["MCTR", "MCTV"]:
                    prefix = str(pathlib.PurePath(path).joinpath(g + ".FIRT.l" + l + ".p" + p + ".k" + k))
                    print(prefix, file=sys.stderr)
                    bestParams = getBestParams(prefix, benchmark, qGlob, True)
                    print(prefix + bestParams[0] + bestParams[1] + ".t*.result")

            for p in ["TR", "SRR", "SRV"]:
                prefix = str(pathlib.PurePath(path).joinpath(g + ".FIRT.l" + l + ".p" + p))
                print(prefix, file=sys.stderr)
                bestParams = getBestParams(prefix, benchmark, qGlob, True)
                print(prefix + bestParams[0] + bestParams[1] + ".t*.result")

        # NN
        for b in batchSizes:
            for s in nnStepSizes:
                for h in hiddenSizes:
                    for p in ["E", "S"]:
                        prefix = str(pathlib.PurePath(path).joinpath(g + ".NN.h" + h + ".b" + b + ".s" + s + ".p" + p))
                        print(prefix, file=sys.stderr)
                        bestParams = getBestParams(prefix, benchmark, qGlob, False)
                        print(prefix + bestParams[0] + bestParams[1] + ".t*.result")        

                    for k in sampleSizes:
                        for p in ["MCTR", "MCTV"]:
                            prefix = str(pathlib.PurePath(path).joinpath(g + ".NN.h" + h + ".b" + b + ".s" + s + ".p" + p + ".k" + k))
                            print(prefix, file=sys.stderr)
                            bestParams = getBestParams(prefix, benchmark, qGlob, True)
                            print(prefix + bestParams[0] + bestParams[1] + ".t*.result")

                    for p in ["TR", "SRV", "SRR"]:
                        prefix = str(pathlib.PurePath(path).joinpath(g + ".NN.h" + h + ".b" + b + ".s" + s + ".p" + p))
                        print(prefix, file=sys.stderr)
                        bestParams = getBestParams(prefix, benchmark, qGlob, True)
                        print(prefix + bestParams[0] + bestParams[1] + ".t*.result")
                    
if __name__ == "__main__":
    main()

