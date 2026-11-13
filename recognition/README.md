# Human recognition study

Authors look at filography images one at a time and enter what they recognise.

## 1. Build the images

```bash
python recognition/build_dataset.py --out D:/IMAGE_DATASET --exclude used_sources.txt
python recognition/build_dataset.py --out D:/IMAGE_DATASET --verify
```

This renders 10 algorithms x 10 thread counts (2k to 20k, step 2k) x 5 images = 500 PNGs
(3 train / 1 val / 1 test per cell), using each COCO val2017 scene at most once.
`--exclude` lists scenes that must not be used, such as those shown in an earlier
round; the list is kept as `excluded_sources.txt`. The first round (5000 images)
and its answers are backed up in `D:/Backup/IMAGE_DATASET_5000_round1_2026-11-09`. `manifest.csv` records each image's algorithm,
thread count, source, benchmark split and COCO classes.

## 2. Configure

Edit `config/recognition.json`:

| Key | Meaning |
| --- | --- |
| `usernames` | Who can sign in. Case does not matter. |
| `demo_usernames` | Test accounts with every image. They stay out of exports and scores, and Reset also replays their first sign-in. |
| `raters_per_image` | How many accounts get each image. Equal to the number of usernames means everyone labels every image. |
| `data_dir` | Folder holding `manifest.csv` and `images/`. |
| `contact` | Email shown on the consent form. |

Assignments are fixed the first time the server starts. While everyone labels
every image (`raters_per_image` equals the number of usernames), you can add new
usernames at the end of the list and raise `raters_per_image` to match, then
restart; existing answers are kept. Any other change needs a new data folder.

## 3. Run

```bash
pip install -r recognition/requirements.txt
python recognition/app.py
```

The server prints a local link, a LAN link and a private admin link. The admin
page shows each account's sign-ins, status and progress, and can export CSVs,
download a database backup, pause an account, or reset one. Reset moves answers
to the `responses_archive` table.

For authors outside your network, publish the server with a tunnel:

```bash
cloudflared tunnel --url http://localhost:8765
```

Share the `https://....trycloudflare.com` link it prints. Anyone with that link
and a valid username can sign in, so use usernames that are not easy to guess.
Keep the PC awake while authors are working.

## 4. Results

Answers are in `<data_dir>/responses.sqlite`. Download a CSV from the admin
page, or score them against COCO:

```bash
python recognition/score.py
python recognition/score.py --split test
```

`--split test` limits scoring to the images held out from YOLO training, for a
direct comparison with the benchmark.

## What authors see

First sign-in: a consent form (tick and full name), then a 12-chapter animated guide.
Authors move through it with Next, and Start appears after the last chapter.
Help reopens the guide at any time.

Each image shows a text box and a Recognized objects list. Answers are typed
freely with no suggestions: a word, a list or a sentence, stored exactly as typed.
Authors can also drag on the image to draw a box and choose which object it
shows. Boxes are optional and stored in image pixels (`boxes` in the CSV). Images are ordered so two in a row never share an
algorithm or thread count. A skipped image, or one open when the page closes,
returns near the end of the queue.

A Size slider resizes the image and is remembered per user on each device.

The bar under the header shows done, unidentified, skipped and remaining tasks.
Click it to see every task and open any of them.

## Shortcuts

| Key | Action |
| --- | --- |
| Enter | Add the typed object; on an empty box, save and move on |
| Enter with no objects | Mark the image Unidentified |
| Left | Previous image, to change an answer |
| Right | Next image; an untouched new image is skipped |
| Backspace | Remove the last object |
| 1-9 | Choose the object for a new box |

Arrow keys move the cursor while the box has text.

The tutorial uses [anime.js](https://github.com/juliangarnier/anime) (MIT),
kept in `static/vendor/`.
