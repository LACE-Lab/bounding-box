import numpy as np
import matplotlib
import matplotlib.pyplot as plot
import matplotlib.ticker as mtick
import sys
import argparse
import math
import glob
import pathlib

# Takes a list of data and smooths it
# Each entry in the resulting list is the average smooth entries in the given list
def smoothData(rawData, smooth):
    if smooth > 0:
        smoothed = [sum(rawData[0:smooth+1])/(smooth+1)]
        for i in range(smooth+1, len(rawData)):
            newSmoothed = smoothed[-1] - (rawData[i - smooth - 1] - rawData[i])/(smooth+1)
            smoothed.append(newSmoothed)

        return smoothed
    else:
        return rawData

def plotData(axis, globs, groupnames, col, qimprovement=True, colors=None, markerColors=None, markers=None, markEverys=None, smooth=1, skipRows=0):
    stepCol = 7

    data = []
    steps = []
    labels = []
    fileIdx = 1

    if markers == None:
        markers = [""]*len(globs)
    if markEverys == None:
        markEverys = [1]*len(globs)
    
    for a in range(len(globs)):
        group = glob.glob(globs[a])
        if len(group) == 0:
            sys.stderr.write("Warning: file group " + globs[a] + " is empty\n")
        else:
            data.append([])
            steps.append([])
            labels.append(groupnames[a])

            for filename in group:
                data[-1].append([])
                steps[-1].append([])
                curStep = 0

                try:
                    fin = open(filename, 'r')
                    fin.readline() # Consume the headings

                    #Read from the file
                    line = fin.readline()
                    while line != '':
                        splitLine = line.split()
                        score = float(splitLine[col])
                        data[-1][-1].append(score)
                        step = int(splitLine[stepCol])
                        curStep += step
                        steps[-1][-1].append(curStep)
                        line = fin.readline()
                    fileInfo = str(fileIdx) + ": " + filename
                    fileInfo += " (" + str(len(steps[-1][-1])) + " eps"
                    fileInfo += ", " + str(steps[-1][-1][-1]) + " frames"
                    fileInfo += ")"
                    print(fileInfo)
                    fin.close()
                except Exception as inst:
                    sys.stderr.write("Error reading " + filename + "\n")
                    sys.stderr.write(str(inst) + "\n")

                fileIdx += 1
    print('---')

    # How much should we smooth?
    smooth = smooth - 1

    if colors == None:
        # Automate color selection for curves        
        axis.set_prop_cycle('color', [plot.cm.gnuplot(i) for i in np.linspace(0.1, 0.9, len(globs))])

    fmt = mtick.ScalarFormatter()
    fmt.set_powerlimits((-3, 3))
    axis.xaxis.set_major_formatter(fmt)         

    qsmoothed = []
    qxCoords = []
    for i in range(len(data[0])):
        qsmoothed.append(smoothData(data[0][i], smooth))
        qxCoords.append(steps[0][i][smooth:])

    fileIdx = 0
    for g in range(0, len(data)):
        print(groupnames[g])
        #First smooth the data
        smoothed = []
        xCoords = []
        for i in range(len(data[g])):
            smoothed.append(smoothData(data[g][i], smooth))
            xCoords.append(steps[g][i][smooth:])

        #Compute the averages
        combinedXCoords = []        
        avgData = []
        upperErr = []
        lowerErr = []
        indices = [0]*len(smoothed)
        qindices = [0]*len(qsmoothed)
        remainingIndices = [i for i in range(len(indices)) if indices[i] < len(smoothed[i])]
        remainingQIndices = [i for i in range(len(qindices)) if qindices[i] < len(qsmoothed[i])]
        skipCounter = 0
        while len(remainingIndices) == len(smoothed) and len(remainingQIndices) == len(qsmoothed):            
            curX = [xCoords[i][indices[i]] for i in remainingIndices]
            if qimprovement:
                curX += [qxCoords[i][qindices[i]] for i in remainingQIndices]
            minX = min(curX)
            if skipCounter == 0:
                combinedXCoords.append(minX)

            # Find the y-coordinate for this data point (average)
            if qimprovement:
                curY = [smoothed[i][indices[i]] - qsmoothed[i][qindices[i]] for i in remainingIndices]
            else:
                curY = [smoothed[i][indices[i]] for i in remainingIndices]
                
            sampleAvg = sum(curY)/len(curY)
            if skipCounter == 0:
                avgData.append(sampleAvg)

                # Find the standard error for the average
                if len(curY) > 1:
                    sqErrs = [(y - sampleAvg)*(y - sampleAvg) for y in curY]
                    stdDev = math.sqrt(sum(sqErrs)/(len(curY)-1))
                else:
                    stdDev = 0
                stdErr = stdDev/math.sqrt(len(curY))
                upperErr.append(sampleAvg+stdErr)
                lowerErr.append(sampleAvg-stdErr)

            minXIndices = [i for i in range(len(remainingIndices)) if curX[i] == minX]
            for idx in minXIndices:
                indices[remainingIndices[idx]] += 1
            remainingIndices = [i for i in range(len(indices)) if indices[i] < len(smoothed[i])]

            if qimprovement:
                minQIndices = [i for i in range(len(remainingQIndices)) if curX[i+len(remainingIndices)] == minX]
                for idx in minQIndices:
                    qindices[remainingQIndices[idx]] += 1
                remainingQIndices = [i for i in range(len(qindices)) if qindices[i] < len(qsmoothed[i])]
            skipCounter += 1
            if skipCounter > skipRows:
                skipCounter = 0

        if colors == None:
            color = next(axis._get_lines.prop_cycler)['color']
        else:
            color = matplotlib.colors.to_rgba(colors[g])
        if markerColors == None:
            markerColor = color
        else:
            markerColor = markerColors[g]

        lineLabel = labels[g]
        # Plot the averages
        p = axis.plot(combinedXCoords, avgData, label=lineLabel, color=color, marker=markers[g], markevery=markEverys[g], markerfacecolor=markerColor, zorder=2)
        if len(data[g]) > 1:
            # Plot the standard error
            lighter = (color[0], color[1], color[2], 0.25)
            axis.fill_between(combinedXCoords, lowerErr, upperErr, color=lighter, zorder=1)

        fileIdx += len(data[g])

def main():
    plot.rc('text', usetex=True)
    plot.rc('text.latex', preamble=plot.rcParams["text.latex.preamble"].join([r"\usepackage{siunitx}"]))
    
    parser = argparse.ArgumentParser(description='Plot learning curves from Acrobot data.')

    parser.add_argument('results_path', metavar='RESULT_PATH', type=str, help='path to the results files')
    parser.add_argument('-p', '--output_path', metavar='OUTPUT_PATH', type=str, default="./", help='the path where output files should be generated (default: ./)')
    parser.add_argument('-k', '--skiprows', metavar='NUMROWS', type=int, default=0, help='Skip NUMROWS rows for every row read in from a file (default: %(default)s, which is no skipping)')
    parser.add_argument('-s', '--smooth', type=int, default=1, help='the size of the smoothing window (default: %(default)s, which is no smoothing).')

    args = parser.parse_args()

    globPrefix = str(pathlib.PurePath(args.results_path).joinpath("A"))

    if args.skiprows > 0:
        markEverys = [(s, int(5000/args.skiprows)) for s in range(0, int(5000/args.skiprows), int(500/args.skiprows))]
    else:
        markEverys = [(s, 5000) for s in range(0, 5000, 500)]
        
    for qimprovement in [True, False]:
        # DT
        fig, axes = plot.subplots(nrows=1, ncols=2, squeeze=False, layout='constrained')
        titles = [r"Regression Trees: \texttt{Acrobot}", r"Regression Trees: \texttt{Distractrobot}"]
        gameNames = ["", "D"]
        if qimprovement:
            axes[0][0].set_ylabel("Avg. Total Reward Centered on Q-learning")
        else:
            axes[0][0].set_ylabel("Avg. Total Reward")
        for i in range(2):
            axes[0][i].set_title(titles[i])
            axes[0][i].set_xlabel("Timestep")
            if qimprovement:
                axes[0][i].set_ylim(ymin=-10, ymax=60)

            algs = ["E", "S", "MCTV.k40", "MCTR.k40", "SRV", "SRR", "TR"]
            algs = [gameNames[i] + ".FIRT.l100.p"+a for a in algs]
            algs.insert(0, gameNames[i]+".pQ")
            #algs.insert(1, gameNames[i]+".pP")
            groupnames = ["Q-learning", "Expect", "Sample", "MCTV($k=40$)", "MCTR($k=40$)", "1SPV", "1SPR", "BBI"]
            globs = [globPrefix + a + ".*result" for a in algs]
            markers = ["+", "o", "D", 10, "^", 9, ">", "s"]
            colors = ["black", "rosybrown", "darkkhaki", "darkviolet", "lightseagreen", "slateblue", "cadetblue", "darkorange"]        
            plotData(axes[0][i], globs, groupnames, 5, qimprovement=qimprovement, colors=colors, markers=markers, markEverys=markEverys, smooth=args.smooth, skipRows=args.skiprows)
        handles, labels = axes[0][0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='outside right upper')
        fig.set_size_inches((10, 3))
        if qimprovement:
            filename = "acro.dt.pdf"
        else:
            filename = "acro.dt.appendix.pdf"
        plot.savefig(str(pathlib.PurePath(args.output_path).joinpath(filename)))

        # NN
        fig, axes = plot.subplots(nrows=1, ncols=2, squeeze=False, layout='constrained')
        titles = [r"Neural Networks: \texttt{Acrobot}", r"Neural Networks: \texttt{Distractrobot}"]
        gameNames = ["", "D"]
        if qimprovement:
            axes[0][0].set_ylabel("Avg. Total Reward Centered on Q-learning")
        else:
            axes[0][0].set_ylabel("Avg. Total Reward")
        for i in range(2):
            axes[0][i].set_title(titles[i])
            axes[0][i].set_xlabel("Timestep")
            if qimprovement:
                axes[0][i].set_ylim(ymin=-10, ymax=60)

            algs = ["E", "S", "MCTV.k40", "MCTR.k40", "SRV", "SRR", "TR"]
            algs = [gameNames[i] + ".NN.h8.b4.s1e-3.p"+a for a in algs]
            algs.insert(0, gameNames[i]+".pQ")
            #algs.insert(1, gameNames[i]+".pP")
            groupnames = ["Q-learning", "Expect", "Sample", "MCTV($k=40$)", "MCTR($k=40$)", "1SPV", "1SPR", "BBI"]
            globs = [globPrefix + a + ".*result" for a in algs]
            markers = ["+", "o", "D", 10, "^", 9, ">", "s"]
            colors = ["black", "rosybrown", "darkkhaki", "darkviolet", "lightseagreen", "slateblue", "cadetblue", "darkorange"]        
            plotData(axes[0][i], globs, groupnames, 5, qimprovement=qimprovement, colors=colors, markers=markers, markEverys=markEverys, smooth=args.smooth, skipRows=args.skiprows)
        handles, labels = axes[0][0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='outside right upper')
        fig.set_size_inches((10, 3))
        if qimprovement:
            filename = "acro.nn.pdf"
        else:
            filename = "acro.nn.appendix.pdf"
        plot.savefig(str(pathlib.PurePath(args.output_path).joinpath(filename)))

if __name__ == '__main__':
    main()
