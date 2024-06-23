import argparse
import pathlib
import glob
import collections
import sys

colToEval = 9

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
                    total += float(line.split()[colToEval])
                    count += 1
            fin.close()
        return total/count

def getBestParams(filePrefix, benchmark, qGlob, useTemp):
    stepSizes = [".a2e-1", ".a1e-1", ".a5e-2", ".a1e-2"]
    if useTemp:
        temperatures = [".m1e1", ".m1", ".m1e-1", ".m1e-2", ".m1e-3"]
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
                        for j in range(1, 100):                            
                            q.append(float(lines[j].split()[colToEval]))
                            qq.append(float(qlines[j].split()[colToEval]))
                        for j in range(100, len(lines)):
                            q.append(float(lines[j].split()[colToEval]))
                            qq.append(float(qlines[j].split()[colToEval]))
                            scoreSum += sum(q) - sum(qq)
                            
                    if scoreSum > bestScore:
                        bestScore = scoreSum
                        bestParams = (a, m)
                    print(a.replace(".", ""), m.replace(".", ""), scoreSum, file=sys.stderr)
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
    
    flagNums = ["2", "10"]
    discounts = ["0.85", "0.9"]
    sampleSizes = ["10", "40"]
    nnStepSizes = ["1e-3"]
    batchSizes = ["4"]    

    numFiles = 0

    for g in discounts:
        for f in flagNums:
            benchmark = float("-inf")
            bestQParams = None
            for a in [".a2e-1", ".a1e-1", ".a5e-2", ".a1e-2"]:
                glob = str(pathlib.PurePath(path).joinpath("gor.g" + g + ".f" + f + ".pQ" + a + ".t*result"))
                score = getFinalPerformance(glob)
                if score > benchmark:
                    benchmark = score
                    bestQParams = a
            qGlob = str(pathlib.PurePath(path).joinpath("gor.g" + g + ".f" + f + ".pQ" + bestQParams + ".t*result"))

            # Oracle/Benchmark
            for p in ["Q", "P"]:
                prefix = str(pathlib.PurePath(path).joinpath("gor.g" + g + ".f" + f + ".p" + p))
                print(prefix, file=sys.stderr)
                bestParams = getBestParams(prefix, benchmark, qGlob, False)
                print(prefix + bestParams[0] + bestParams[1] + ".t*.result")

            for h in ["2", "5"]:
                for p in ["IE", "IS"]:
                    prefix = str(pathlib.PurePath(path).joinpath("gor.g" + g + ".f" + f + ".p" + p + ".h" + h))
                    print(prefix, file=sys.stderr)
                    bestParams = getBestParams(prefix, benchmark, qGlob, False)
                    print(prefix + bestParams[0] + bestParams[1] + ".t*.result")        

            for k in sampleSizes:
                for p in ["IMCTR", "IMCTV"]:
                    prefix = str(pathlib.PurePath(path).joinpath("gor.g" + g + ".f" + f + ".p" + p + ".k" + k))
                    print(prefix, file=sys.stderr)
                    bestParams = getBestParams(prefix, benchmark, qGlob, True)
                    print(prefix + bestParams[0] + bestParams[1] + ".t*.result")

            for p in ["ITR", "ISRR", "ISRV"]:
                prefix = str(pathlib.PurePath(path).joinpath("gor.g" + g + ".f" + f + ".p" + p))
                print(prefix, file=sys.stderr)
                bestParams = getBestParams(prefix, benchmark, qGlob, True)
                print(prefix + bestParams[0] + bestParams[1] + ".t*.result")

            # Decision Trees
            for p in ["A", "E", "S"]:
                prefix = str(pathlib.PurePath(path).joinpath("gor.g" + g + ".f" + f + ".FIRT.p" + p))
                print(prefix, file=sys.stderr)
                bestParams = getBestParams(prefix, benchmark, qGlob, False)
                print(prefix + bestParams[0] + bestParams[1] + ".t*.result")        

            for k in sampleSizes:
                for p in ["MCTR", "MCTV"]:
                    prefix = str(pathlib.PurePath(path).joinpath("gor.g" + g + ".f" + f + ".FIRT.p" + p + ".k" + k))
                    print(prefix, file=sys.stderr)
                    bestParams = getBestParams(prefix, benchmark, qGlob, True)
                    print(prefix + bestParams[0] + bestParams[1] + ".t*.result")

            for p in ["TR"]:
                prefix = str(pathlib.PurePath(path).joinpath("gor.g" + g + ".f" + f + ".FIRT.p" + p))
                print(prefix, file=sys.stderr)
                bestParams = getBestParams(prefix, benchmark, qGlob, True)
                print(prefix + bestParams[0] + bestParams[1] + ".t*.result")

            # NN
            for b in batchSizes:
                for s in nnStepSizes:
                    for p in ["A", "E", "S"]:
                        prefix = str(pathlib.PurePath(path).joinpath("gor.g" + g + ".f" + f + ".NN.b" + b + ".s" + s + ".p" + p))
                        print(prefix, file=sys.stderr)
                        bestParams = getBestParams(prefix, benchmark, qGlob, False)
                        print(prefix + bestParams[0] + bestParams[1] + ".t*.result")

                    for k in sampleSizes:
                        for p in ["MCTR", "MCTV"]:
                            prefix = str(pathlib.PurePath(path).joinpath("gor.g" + g + ".f" + f + ".NN.b" + b + ".s" + s + ".p" + p + ".k" + k))
                            print(prefix, file=sys.stderr)
                            bestParams = getBestParams(prefix, benchmark, qGlob, True)
                            print(prefix + bestParams[0] + bestParams[1] + ".t*.result")

                    for p in ["TR"]:
                        prefix = str(pathlib.PurePath(path).joinpath("gor.g" + g + ".f" + f + ".NN.b" + b + ".s" + s + ".p" + p))
                        print(prefix, file=sys.stderr)
                        bestParams = getBestParams(prefix, benchmark, qGlob, True)
                        print(prefix + bestParams[0] + bestParams[1] + ".t*.result")
                    
if __name__ == "__main__":
    main()

