import argparse
import pathlib

def main():
    parser = argparse.ArgumentParser(description="Generates config files to evaluate selected configurations")

    parser.add_argument('list_file', metavar='FILENAME', type=str, help='The file containing the list of selected file globs')
    parser.add_argument('-p', '--output_path', metavar='PATH', type=str, default='./', help='the path where output files should be generated (default: ./)')
    parser.add_argument("--num_trials", metavar="NUM_TRIALS", type=int, default=50, help="the number of trials for each configuration (default: 50)")
    parser.add_argument("--first_seed", metavar="FIRST_SEED", type=int, default=1, help="the first seed in the sequence (default: 1)")
    parser.add_argument("--gen_manifest", action="store_true", help="generate a manifest file listing all files generated")
    parser.add_argument("--gen_globs", action="store_true", help="generate a file listing a glob over trials for the results of each selected configuration")
    
    args = parser.parse_args()

    print("Generating files in " + str(pathlib.PurePath(args.output_path)))

    try:
        listFileIn = open(args.list_file, 'r')
    except:
        print("Could not open list file: " + args.list_file)
        exit(1)

    if (args.gen_manifest):
        manFilename = str(pathlib.PurePath(args.output_path).joinpath("manifest.txt"))
        try:
            manOut = open(manFilename, "w")
        except:
            print("Could not open manifest file: " + manFilename)
            manOut = None
    else:
        manOut = None

    if (args.gen_globs):
        globFilename = str(pathlib.PurePath(args.output_path).joinpath("globs.txt"))
        try:
            globOut = open(globFilename, "w")
        except:
            print("Could not open glob file: " + globFilename)
            globOut = None
    else:
        globOut = None
        
    globList = listFileIn.read().split()
    fileList = []
    for g in globList:
        fileList.append(g.split(".result")[0])

    paramList = []
    for f in fileList:
        try:
            fin = open(f.replace("*", "1")+".config", 'r')
        except:
            print("Could not open config file: " + f.replace("*", "1")+".config")
            exit(1)

        params = {}
        for line in fin:
            sline = line.split()
            params[sline[0]] = sline[2]

        params["seed"] = "*"
        filePrefix = pathlib.PurePath(f).parts[-1]
        params["output"] = str(pathlib.PurePath(args.output_path).joinpath(filePrefix))
        if globOut:
            globOut.write(params["output"] + ".result\n")
        paramList.append(params)
        
    numFiles = 0        
    for t in range(args.first_seed, args.first_seed + args.num_trials):
        for params in paramList:
            filename = params["output"].replace("*", str(t)) + ".config"
            try:
                fout = open(filename, "w")
                if manOut:
                    manOut.write(filename + "\n")
                for p in params:
                    fout.write(p + " = " + params[p].replace("*", str(t)) + "\n")
                fout.close()
                numFiles += 1
            except:
                print("Could not open file for writing: " + filename)
                exit(1)

        print("Trial " + str(t) + ": " + str(numFiles) + " so far")
    print(str(numFiles) + " files generated")
    
if __name__ == "__main__":
    main()
