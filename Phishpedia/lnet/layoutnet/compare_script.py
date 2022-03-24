with open(r"/Volumes/GoogleDrive/Meine Ablage/phish_results_compare.txt", "r") as cFile:
    lines = cFile.readlines()
    c = 0
    for line in lines:
        line = line.split("\t")
        if line[1] != line[4] and line[4] == "None":
            print(line)
            c += 1
    print(c)
    print(len(lines))


with open(r"/Volumes/GoogleDrive/Meine Ablage/LNet/Phishpedia/lnet/test_compare.txt", "r") as cFile:
    with open(r"/Volumes/GoogleDrive/Meine Ablage/LNet/Phishpedia/lnet/test.txt", "r") as cFile2:
        lines1 = cFile.readlines()
        lines2 = cFile2.readlines()
        c = 0
        pedia = 0
        for i, line in enumerate(lines1):
            line = line.split("\t")
            line2 = lines2[i].split("\t")
            nline = [line2[0], line2[1], line2[2], line[3], line[4]]
            if nline[4] != "None":
                pedia += 1
            if nline[1] != nline[4] and nline[4] != "None":
                print(nline)
                c += 1
        print(c)
        print(c / len(lines1))

with open(r"/Volumes/GoogleDrive/Meine Ablage/LNet/Phishpedia/lnet/test_compare.txt", "r") as cFile:
    with open(r"/Volumes/GoogleDrive/Meine Ablage/LNet/Phishpedia/lnet/test.txt", "r") as cFile2:
        with open(r"/Volumes/GoogleDrive/Meine Ablage/LNet/Phishpedia/lnet/combine.txt", "a") as wFile:
            lines1 = cFile.readlines()
            lines2 = cFile2.readlines()
            all_lines = []
            c = 0
            pedia = 0
            for i, line in enumerate(lines1):
                line = line.split("\t")
                line2 = lines2[i].split("\t")
                nline = [line2[0], line2[1], line2[2], line[3], line[4]]
                if nline[4] != "None":
                    pedia += 1
                if nline[1] != nline[4] and nline[4] != "None":
                    print(nline)
                    c += 1
                wFile.write("\t".join(nline) + "\n")
