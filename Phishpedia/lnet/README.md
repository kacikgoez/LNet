# Spatial Relation Network for Identity Logo Identifiaction

Expanded version of Phishpedia that takes in the spatial relationship of different logos, and evaluates the chance of it being the identity logo based on size, neighbours and its absolute and relative spatial relationship.

- ```phishpedia_main.py```:
  
Modified version of main.py for simple directory structure. Runs the model on every image found in the given directory, and evaluates every screenshot indepedent of any meta information.
```shell
cd path/to/lnet
python phishpedia_main.py --folder /folder/to/screenshots/ --results /path/to/store_result.txt
```

- ```phishpedia_main_single.py```:
  
Lets you evaluate single screenshot for debugging purposes, and prints the binning and confidence. The console will ask you for a default path (optional), so you can iterate through multiple
files within the same directory without specifying the path. Then it will ask for the file name / file path in a while loop. Last screenshot is stored in the ```lnet``` folder under ```single-screenshot.png```
```shell
python phishpedia_main_single.py 
```

- ```phishpedia_main_url.py```:
  
Lets you evaluate a single or multiple URLs. Once started, a prompt asks for a URL (or multiple separated by space).

```pyshot.py``` creates a screenshot of the given URL, then proceeds to store it under ```/datasets/testsites/{url}``` with the proposed logos.
```shell 
python phishpedia_main_url.py 
```

- ```lnet_main.py```:
```lnet_main.py``` contains the altered procedure of Phishpedia and all the new functionality added.


- ```transparency-scan.py```:
```transparency-scan.py``` looks for image files with a transparent background. Transparent backgrounds can cause issues with the Siamese network.