import os
import argparse
from io import StringIO
import pickle
import tqdm
from lnet_main import main

if __name__ == "__main__":
    buffer = StringIO()
    counter = 0

    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder", help='Input folder path to parse',  default=r'/Volumes/GoogleDrive/Meine Ablage/holle2/screenshots/benign/')
    parser.add_argument('-r', "--results", help='Input results file name', default=r'/Volumes/GoogleDrive/Meine Ablage/benign_results_mac.txt')
    args = parser.parse_args()
    date = args.folder.split('/')[-1]
    directory = args.folder
    results_path = args.results.split('.txt')[0] + ".txt"
    counter = 0

    # Uncomment to scan single file for debugging purposes
    # phish_category, phish_target, plotvis, siamese_conf, pred1, conf2 = main(url="https://test.com", screenshot_path="/Volumes/Extern/holle-screenshots/screenshots/benign/00234877-screenshot.png")
    # print(phish_category, "|", phish_target, "|", siamese_conf, "|", pred1, "|", conf2)
    # cv2.imwrite(os.path.join("/Volumes/GoogleDrive/Meine Ablage/Phishpedia/lnet/layoutnet", "00234852-screenshot.png"), plotvis)
    # exit(0)

    buffer.write("File" + "\t")
    #f.write("url" + "\t")
    buffer.write("LNet" + "\t")
    buffer.write("LNet conf." + "\t")  # write top1 prediction only
    buffer.write("Orig." + "\t")
    buffer.write("Orig. Conf" + "\n")

    # Save directory structure for Google Colab to avoid IO error
    dir_struct_name = r"holle-" + os.path.basename(os.path.normpath(directory)) + "-struct.pickle"
    if not os.path.exists(dir_struct_name):
        # Create buffer for directory items
        dirbuff = []
        # Load dir.
        dir_struct = tqdm(os.listdir(directory))
        # Add all items to list as str.
        for i in dir_struct:
            print(str(os.path.basename(os.path.normpath(i))))
            dirbuff.append(str(os.path.basename(os.path.normpath(i))))
        # Store
        with open(dir_struct_name, 'wb') as dir_file:
            pickle.dump(dirbuff, dir_file)
        # Delete buffer
        dirbuff = []
    else:
        with open(dir_struct_name, 'rb') as dir_file:
            dir_struct = pickle.load(dir_file)

    # If file already exists, find last file
    startFrom = False
    if os.path.exists(results_path):
        with open(results_path, "r") as rfile:
            alllines = rfile.readlines()
            lines = alllines[::-1]
            buffer.writelines(alllines)
            for item in lines:
                item = item.split("\t")
                if len(item) > 3:
                    startFrom = item[0]
                    break

    print("Starting from:" + str(startFrom))

    for item in dir_struct:
        start_time = time.time()
        full_path = os.path.join(directory, item)

        # Skip until file found
        if startFrom:
            if str(item) == str(startFrom):
                startFrom = False
                continue
            else:
                continue

        # if item in open(results_path, encoding='ISO-8859-1').read(): # have been predicted
        #     continue
        try:
            screenshot_path = os.path.join(directory, item)
            #url = open(os.path.join(full_path, 'info.txt'), encoding='ISO-8859-1').read()
            url = "https://www.test-on-holle.com"

            if 0 == 1:
                continue
            else:
                phish_category, phish_target, plotvis, siamese_conf, pred_target_2, siamese_conf_2, lnet_conf_ret = main(url=url, screenshot_path=screenshot_path)
                counter += 1

                ''''# FIXME: call VTScan only when phishpedia report it as phishing
                vt_result = "None"
                if 0 == 1:
                    try:
                        if vt_scan(url) is not None:
                            positive, total = vt_scan(url)
                            print("Positive VT scan!")
                            vt_result = str(positive) + "/" + str(total)
                        else:
                            print("Negative VT scan!")
                            vt_result = "None"

                    except Exception as e:
                        print('VTScan is not working...')
                        vt_result = "error"'''

                vt_result = "None"

                # write results as well as predicted images
                buffer.write(item + "\t")  # write top1 prediction only
                buffer.write(str(phish_target) + "\t")  # write top1 prediction only
                buffer.write(str(siamese_conf) + "\t")
                buffer.write(str(lnet_conf_ret) + "\t")
                buffer.write(str(pred_target_2) + "\t")  # write top1 prediction only
                buffer.write(str(siamese_conf_2) + "\n")  # write top1 prediction only

                print("RESULT", item, phish_target, siamese_conf, pred_target_2, siamese_conf_2)

                if counter > 20:
                    if os.path.exists(results_path):
                        os.remove(results_path)
                    with open(results_path, "w+", encoding='ISO-8859-1') as f:
                        f.write(buffer.getvalue())
                        f.close()
                    counter = 0

                #cv2.imwrite(os.path.join(full_path, "./local_predict.png"), plotvis)
                #cv2.imwrite(os.path.join(full_path, "predict.png"), plotvis)

        except Exception as e:
            raise

    if os.path.exists(results_path):
        os.remove(results_path)
    with open(results_path, "w+", encoding='ISO-8859-1') as f:
        f.write(buffer.getvalue())
        f.close()




