import argparse
import pathlib

def genCFG(params, manifestFile):
    try:
        fout = open(params["output"] + ".config", "w")
        if (manifestFile):
            manifestFile.write(params["output"] + ".config\n")
        for k in params:
            fout.write(k + " = " + params[k] + "\n")
        fout.close()
    except:
        print("Could not open file for writing: " + params["output"] + ".config")
    
def main():
    parser = argparse.ArgumentParser(description="Generates config files for Go-Up parameter sweeps.")

    parser.add_argument("--path", metavar="PATH", type=str, default="./", help="path where config files (and experimental output files) should be generated")
    parser.add_argument("--num_trials", metavar="NUM_TRIALS", type=int, default=100, help="the number of trials for each configuration")
    parser.add_argument("--first_seed", metavar="FIRST_SEED", type=int, default=1, help="the first seed in the sequence")
    parser.add_argument("--gen_manifest", action="store_true", help="generate a manifest file listing all files generated")

    args = parser.parse_args()

    path = args.path

    print("Generating files in " + str(pathlib.PurePath(path)))
    
    if (args.gen_manifest):
        manFilename = str(pathlib.PurePath(path).joinpath("manifest.txt"))
        try:
            manOut = open(manFilename, "w")
        except:
            print("Could not open manifest file: " + manFilename)
            manOut = None
    else:
        manOut = None

    discounts = ["0.9", "0.85"]
    indNums = ["2", "10"]
    stepSizes = ["1e-2", "5e-2", "1e-1", "2e-1"]
    temperatures = ["1e-3", "1e-2", "1e-1", "1", "1e1"]
    sampleSizes = ["10", "40"]
    nnStepSizes = ["1e-3"]
    batchSizes = ["4"]    

    trials = range(args.first_seed, args.first_seed + args.num_trials)

    params = {}
    params["game"] = "GR"
    params["num_frames"] = "300000"
    params["use_nn"] = "0"
    params["predict_change"] = "1"
    params["hidden_size"] = "64"

    numFiles = 0
    
    for t in trials:
        params["seed"] = str(t)
        for g in discounts:
            params["discount"] = g
            for f in indNums:        
                params["gor_num_ind"] = f
                for a in stepSizes:
                    params["step_size"] = a
                    for p in ["Q", "P"]:
                        params["gor_num_ind"] = f
                        filename = "gor.g" + g + ".f" + f + ".p" + p + ".a" + a + ".t" + str(t)
                        params["output"] = str(pathlib.PurePath(path).joinpath(filename))
                        params["planner"] = p
                        genCFG(params, manOut)
                        numFiles += 1

                    for p in ["IE", "IS"]:
                        params["planner"] = p
                        for h in ["2", "5"]:
                            filename = "gor.g" + g + ".f" + f + ".p" + p + ".h" + h + ".a" + a + ".t" + str(t)
                            params["output"] = str(pathlib.PurePath(path).joinpath(filename))
                            params["horizon"] = h
                            genCFG(params, manOut)
                            numFiles += 1

                    # Decision Tree
                    params["update_every"] = "100"
                    for p in ["A", "E", "S"]:
                        filename = "gor.g" + g + ".f" + f + ".FIRT.p" + p + ".a" + a + ".t" + str(t)
                        params["output"] = str(pathlib.PurePath(path).joinpath(filename))
                        params["planner"] = p
                        genCFG(params, manOut)
                        numFiles += 1                    

                    # Neural Net
                    params["use_nn"] = "1"
                    for b in batchSizes:
                        params["batch_size"] = b
                        params["update_every"] = b
                        for s in nnStepSizes:
                            params["nn_step_size"] = s
                            for p in ["A", "S", "E"]:                                
                                filename = "gor.g" + g + ".f" + f + ".NN.b" + b + ".s" + s + ".p" + p + ".a" + a + ".t" + str(t)
                                params["output"] = str(pathlib.PurePath(path).joinpath(filename))
                                params["planner"] = p
                                genCFG(params, manOut)
                                numFiles += 1
                    params["use_nn"] = "0"                                

                    for m in temperatures:
                        params["temperature"] = m
                        for p in ["ITR", "ISRR", "ISRV"]:
                            filename = "gor.g" + g + ".f" + f + ".p" + p + ".a" + a + ".m" + m + ".t" + str(t)
                            params["output"] = str(pathlib.PurePath(path).joinpath(filename))
                            params["planner"] = p
                            numFiles += 1
                            genCFG(params, manOut)

                        # Decision Tree
                        params["update_every"] = "100"
                        for p in ["TR"]:
                            params["planner"] = p
                            filename = "gor.g" + g + ".f" + f + ".FIRT.p" + p + ".a" + a + ".m" + m + ".t" + str(t)
                            params["output"] = str(pathlib.PurePath(path).joinpath(filename))
                            numFiles += 1
                            genCFG(params, manOut)

                        # Neural Net
                        params["use_nn"] = "1"
                        for p in ["TR"]:
                            params["planner"] = p
                            for b in batchSizes:
                                params["batch_size"] = b
                                params["update_every"] = b
                                for s in nnStepSizes:
                                    params["nn_step_size"] = s
                                    filename = "gor.g" + g + ".f" + f + ".NN.b" + b + ".s" + s + ".p" + p + ".a" + a + ".m" + m + ".t" + str(t)
                                    params["output"] = str(pathlib.PurePath(path).joinpath(filename))
                                    genCFG(params, manOut)
                                    numFiles += 1
                        params["use_nn"] = "0"

                        for k in sampleSizes:
                            params["num_samples"] = k
                            for p in ["IMCTV", "IMCTR"]:
                                filename = "gor.g" + g + ".f" + f + ".p" + p + ".k" + k + ".a" + a + ".m" + m + ".t" + str(t)
                                params["output"] = str(pathlib.PurePath(path).joinpath(filename))
                                params["planner"] = p
                                genCFG(params, manOut)
                                numFiles += 1

                            for p in ["MCTV", "MCTR"]:
                                params["planner"] = p

                                # Decision Tree
                                params["update_every"] = "100"
                                filename = "gor.g" + g + ".f" + f + ".FIRT.p" + p + ".k" + k + ".a" + a + ".m" + m + ".t" + str(t)
                                params["output"] = str(pathlib.PurePath(path).joinpath(filename))
                                genCFG(params, manOut)
                                numFiles += 1

                                # Neural Net
                                params["use_nn"] = "1"
                                for b in batchSizes:
                                    params["batch_size"] = b
                                    params["update_every"] = b
                                    for s in nnStepSizes:
                                        params["nn_step_size"] = s
                                        filename = "gor.g" + g + ".f" + f + ".NN.b" + b + ".s" + s + ".p" + p + ".k" + k + ".a" + a + ".m" + m + ".t" + str(t)
                                        params["output"] = str(pathlib.PurePath(path).joinpath(filename))
                                        genCFG(params, manOut)
                                        numFiles += 1
                                params["use_nn"] = "0"
        print("Trial " + str(t) + ": " + str(numFiles) + " so far")
    print(str(numFiles) + " files generated")
if __name__ == "__main__":
    main()

