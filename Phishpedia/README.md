<h2>Expanded version of Phishpedia</h2>

First download the repository, and run:

```bash
pip -r install requirements.txt
```

Then proceed to install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

Install the Chrome Driver and make sure to add it to your environment:
```bash
apt-get update
apt install chromium-chromedriver
cp /usr/lib/chromium-browser/chromedriver /usr/bin
```
Finally, download the models from Google Drive (since LSF is most likely out of bandwith):
```bash
./download.sh
```

<h3>LNet</h3>
To try out the layout network, go into the `lnet` folder and read the `README` file there.