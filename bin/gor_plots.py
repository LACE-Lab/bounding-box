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

def plotData(axis, globs, groupnames, col, colors=None, markerColors=None, markers=None, markEverys=None, smooth=1, skipRows=0, negate=False):
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
                        if negate:
                            score = -score
                        data[-1][-1].append(score)
                        step = int(splitLine[stepCol])
                        curStep += step
                        steps[-1][-1].append(curStep)
                        line = fin.readline()
                        # Handle line skipping
                        skip = 0
                        while skip < skipRows and line != '':
                            nextLine = fin.readline()
                            if nextLine != '':
                                if stepCol > 0:
                                    curStep += int(line.split()[stepCol])
                                line = nextLine
                            skip += 1
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

    fileIdx = 0
    for g in range(len(data)):
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
        remainingIndices = [i for i in range(len(indices)) if indices[i] < len(smoothed[i])]
        totals = [len(remainingIndices)]
        while len(remainingIndices) == len(smoothed):            
            curX = [xCoords[i][indices[i]] for i in remainingIndices]
            minX = min(curX)
            combinedXCoords.append(minX)

            # Find the y-coordinate for this data point (average)
            curY = [smoothed[i][indices[i]] for i in remainingIndices]
            sampleAvg = sum(curY)/len(curY)
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

            minXIndices = [i for i in range(len(curX)) if curX[i] == minX]
            for idx in minXIndices:
                indices[remainingIndices[idx]] += 1
            remainingIndices = [i for i in range(len(indices)) if indices[i] < len(smoothed[i])]

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

def addParams(globs, groupnames):
    for i in range(len(groupnames)):
        alpha = globs[i].split(".a")[1].split(".")[0]
        groupnames[i] += r" ($\alpha$=\num[tight-spacing=true]{" + alpha + "}"
        if ".m" in globs[i]:
            tau = globs[i].split(".m")[1].split(".")[0]
            groupnames[i] += r", $\tau$=\num[tight-spacing=true]{" + tau + "}"
        groupnames[i] += ")"
    
def main():
    plot.rc('text', usetex=True)
    plot.rc('text.latex', preamble=plot.rcParams["text.latex.preamble"].join([r"\usepackage{siunitx}"]))
    
    parser = argparse.ArgumentParser(description='Plots learning curves from Go Right data.')

    parser.add_argument('results_path', metavar='RESULT_PATH', type=str, help='path to the results files')
    parser.add_argument('-p', '--output_path', metavar='OUTPUT_PATH', type=str, default="./", help='the path where output files should be generated (default: ./)')
    parser.add_argument('-k', '--skiprows', metavar='NUMROWS', type=int, default=0, help='Skip NUMROWS rows for every row read in from a file (default: %(default)s, which is no skipping)')
    parser.add_argument('-s', '--smooth', type=int, default=1, help='the size of the smoothing window (default: %(default)s, which is no smoothing).')

    args = parser.parse_args()

    globPrefix = str(pathlib.PurePath(args.results_path).joinpath("gor."))

    markEverys = [(s, 100) for s in range(0, 110, 10)]

    colToPlot = 9
    
    # Oracle Unselective
    fig, axes = plot.subplots(nrows=1, ncols=1, squeeze=False, layout='constrained')    
    axes[0][0].set_title(r"\texttt{Go-Right} Baselines")
    axes[0][0].set_ylabel("Avg. Discounted Return")
    axes[0][0].set_xlabel("Timestamp")
    axes[0][0].set_ylim(-0.1, 1.85)
    
    algs = ["Q", "P", "IE.h2", "IE.h5", "IS.h2", "IS.h5"]
    algs = ["g0.9.f2.p"+a for a in algs]
    groupnames = ["Q-learning", "Perfect", "Expect ($h=2$)", "Expect ($h=5$)", "Sample ($h=2$)", "Sample ($h=5$)"]
    globs = [globPrefix + a + "*result" for a in algs]
    markers = ["+", "x", ".", "o", "d", "D"]
    colors = ["black", "black", "darksalmon", "rosybrown", "goldenrod", "darkkhaki"]
    plotData(axes[0][0], globs, groupnames, colToPlot, colors=colors, markers=markers, markEverys=markEverys, smooth=args.smooth, skipRows=args.skiprows)
    handles, labels = axes[0][0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='outside right upper')
    filename = "gor.g0.9.f2.oracle.unselective.pdf"
    fig.set_size_inches((5, 3))
    plot.savefig(str(pathlib.PurePath(args.output_path).joinpath(filename)))    
    
    # Oracle Selective
    fig, axes = plot.subplots(nrows=1, ncols=2, squeeze=False, layout='constrained')
    gameNames = [r"Hand-Coded: \texttt{Go-Right}", r"Hand-Coded: \texttt{Go-Right-10}"]
    flagNums = ["2", "10"]
    axes[0][0].set_ylabel("Avg. Discounted Return")
    for i in range(2):
        f = flagNums[i]
        axes[0][i].set_title(gameNames[i])
        axes[0][i].set_xlabel("Training Steps")
        axes[0][i].set_ylim(-0.1, 1.85)

        algs = ["Q", "P", "IMCTV.k10", "IMCTV.k40", "IMCTR.k10", "IMCTR.k40", "ISRV", "ISRR", "ITR"]
        algs = ["g0.9.f"+f+".p"+a for a in algs]
        groupnames = ["Q-learning", "Perfect", "MCTV($k=10$)", "MCTV($k=40$)", "MCTR($k=10$)", "MCTR($k=40$)", "1SPV", "1SPR", "BBI"]
        globs = [globPrefix + a + "*result" for a in algs]
        markers = ["+", "x", 11, 10, "v", "^", 9, ">", "s"]
        colors = ["black", "black", "mediumpurple", "darkviolet", "teal", "lightseagreen", "slateblue", "cadetblue", "darkorange"]        
        plotData(axes[0][i], globs, groupnames, colToPlot, colors=colors, markers=markers, markEverys=markEverys, smooth=args.smooth, skipRows=args.skiprows)
    handles, labels = axes[0][0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='outside right upper')
    fig.set_size_inches((10, 3))
    filename = "gor.g0.9.oracle.selective.pdf"
    plot.savefig(str(pathlib.PurePath(args.output_path).joinpath(filename)))

    # Oracle Discount 0.85
    fig, axes = plot.subplots(nrows=1, ncols=2, squeeze=False, layout='constrained')
    gameNames = [r"Hand-Coded: \texttt{Go-Right} ($\gamma=0.85$)", r"Hand-Coded: \texttt{Go-Right-10} ($\gamma=0.85$)"]
    flagNums = ["2", "10"]
    axes[0][0].set_ylabel("Avg. Discounted Return")
    for i in range(2):
        f = flagNums[i]
        axes[0][i].set_title(gameNames[i])
        axes[0][i].set_xlabel("Training Steps")
        axes[0][i].set_ylim(-0.0055, 0.0005)        

        algs = ["Q", "P", "IMCTV.k10", "IMCTV.k40", "IMCTR.k10", "IMCTR.k40", "ISRV", "ISRR", "ITR", "IE.h5", "IS.h5"]
        algs = ["g0.85.f"+f+".p"+a for a in algs]
        groupnames = ["Q-learning", "Perfect", "MCTV($k=10$)", "MCTV($k=40$)", "MCTR($k=10$)", "MCTR($k=40$)", "1SPV", "1SPR", "BBI", "Expect ($h=5$)", "Sample ($h=5$)"]
        globs = [globPrefix + a + "*result" for a in algs]
        markers = ["+", "x", 11, 10, "v", "^", 9, ">", "s", "o", "D"]
        colors = ["black", "black", "mediumpurple", "darkviolet", "teal", "lightseagreen", "slateblue", "cadetblue", "darkorange", "rosybrown", "darkkhaki"]        
        plotData(axes[0][i], globs, groupnames, colToPlot, colors=colors, markers=markers, markEverys=markEverys, smooth=args.smooth, skipRows=args.skiprows)
    handles, labels = axes[0][0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='outside right upper')
    fig.set_size_inches((10, 3))
    filename = "gor.g0.85.oracle.pdf"
    plot.savefig(str(pathlib.PurePath(args.output_path).joinpath(filename)))
    
    # DT
    for g in ["0.85", "0.9"]:
        fig, axes = plot.subplots(nrows=1, ncols=2, squeeze=False, layout='constrained')
        gameNames = [r"Regression Trees: \texttt{Go-Right}", r"Regression Trees: \texttt{Go-Right-10}"]
        if g == "0.85":
            for i in range(len(gameNames)):
                gameNames[i] += r" ($\gamma=0.85$)"        
        flagNums = ["2", "10"]
        axes[0][0].set_ylabel("Avg. Discounted Return")
        for i in range(2):
            f = flagNums[i]
            axes[0][i].set_title(gameNames[i])
            axes[0][i].set_xlabel("Training Steps")
            if g == "0.85":
                axes[0][i].set_ylim(-0.0055, 0.0005)
            else:
                axes[0][i].set_ylim(-0.1, 1.85)

            algs = ["A", "MCTV.k40", "MCTR.k40", "TR"]
            algs = ["g"+g+".f"+f+".FIRT.p"+a for a in algs]
            algs.insert(0, "g"+g+".f"+f+".pQ")
            groupnames = ["Q-learning", "Sufficient", "MCTV($k=40$)", "MCTR($k=40$)", "BBI"]
            globs = [globPrefix + a + "*result" for a in algs]
            markers = ["+", "*", 10, "^", "s"]
            colors = ["black", "grey", "darkviolet", "lightseagreen", "darkorange"]        
            plotData(axes[0][i], globs, groupnames, colToPlot, colors=colors, markers=markers, markEverys=markEverys, smooth=args.smooth, skipRows=args.skiprows)
        handles, labels = axes[0][0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='outside right upper')
        fig.set_size_inches((10, 3))
        filename = "gor.g"+g+".dt.pdf"
        plot.savefig(str(pathlib.PurePath(args.output_path).joinpath(filename)))
    
        # NN
        fig, axes = plot.subplots(nrows=1, ncols=2, squeeze=False, layout='constrained')
        gameNames = [r"Neural Networks: \texttt{Go-Right}", r"Neural Networks: \texttt{Go-Right-10}"]
        if g == "0.85":
            for i in range(len(gameNames)):
                gameNames[i] += r" ($\gamma=0.85$)"
        flagNums = ["2", "10"]
        axes[0][0].set_ylabel("Avg. Discounted Return")
        axes[0][i].set_xlabel("Training Steps")
        for i in range(2):
            f = flagNums[i]
            axes[0][i].set_title(gameNames[i])
            if g == "0.85":
                axes[0][i].set_ylim(-0.0055, 0.0005)
            else:
                axes[0][i].set_ylim(-0.1, 1.85)

            algs = ["A", "MCTV.k40", "MCTR.k40", "TR"]
            algs = ["g"+g+".f"+f+".NN.b4.s1e-3.p"+a for a in algs]
            algs.insert(0, "g"+g+".f"+f+".pQ")
            groupnames = ["Q-learning", "Sufficient", "MCTV($k=40$)", "MCTR($k=40$)", "BBI"]
            globs = [globPrefix + a + "*result" for a in algs]
            markers = ["+", "*", 10, "^", "s"]
            colors = ["black", "grey", "darkviolet", "lightseagreen", "darkorange"]        
            plotData(axes[0][i], globs, groupnames, colToPlot, colors=colors, markers=markers, markEverys=markEverys, smooth=args.smooth, skipRows=args.skiprows)
        handles, labels = axes[0][0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='outside right upper')
        fig.set_size_inches((10, 3))
        filename = "gor.g"+g+".nn.pdf"
        plot.savefig(str(pathlib.PurePath(args.output_path).joinpath(filename)))

    # DT (unc err)
    fig, axes = plot.subplots(nrows=1, ncols=2, squeeze=False, layout='constrained')
    gameNames = [r"Regression Trees: \texttt{Go-Right-10}", r"Neural Networks: \texttt{Go-Right-10}"]
    axes[0][0].set_ylabel("Avg. Median Uncertainty Error")
    f = "10"

    algs = ["MCTV.k40", "MCTR.k40", "TR"]
    groupnames = ["MCTV($k=40$)", "MCTR($k=40$)", "BBI"]
    markers = [10, "^", "s"]
    colors = ["darkviolet", "lightseagreen", "darkorange"]        

    axes[0][0].set_title(gameNames[0])
    axes[0][0].set_xlabel("Training Steps")
    axes[0][0].set_ylim(-0.5, 2.5)
    dtalgs = ["g"+g+".f"+f+".FIRT.p"+a for a in algs]
    globs = [globPrefix + a + "*result" for a in dtalgs]
    plotData(axes[0][0], globs, groupnames, 66, colors=colors, markers=markers, markEverys=markEverys, smooth=args.smooth, skipRows=args.skiprows, negate=True)

    axes[0][1].set_title(gameNames[1])
    axes[0][1].set_xlabel("Training Steps")
    axes[0][1].set_ylim(-0.5, 2.5)
    nnalgs = ["g"+g+".f"+f+".NN.b4.s1e-3.p"+a for a in algs]
    globs = [globPrefix + a + "*result" for a in nnalgs]
    plotData(axes[0][1], globs, groupnames, 66, colors=colors, markers=markers, markEverys=markEverys, smooth=args.smooth, skipRows=args.skiprows, negate=True)

    handles, labels = axes[0][0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='outside right upper')
    fig.set_size_inches((10, 3))
    filename = "gor.uncerr.pdf"
    plot.savefig(str(pathlib.PurePath(args.output_path).joinpath(filename)))
        
if __name__ == '__main__':
    main()
