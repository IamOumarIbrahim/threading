"""COCO labels used to score typed answers. Authors never see this list."""

from __future__ import annotations

import re

GROUPS = {
    "People": ["person"],
    "Vehicles": ["bicycle", "car", "motorcycle", "airplane", "bus", "train",
                 "truck", "boat"],
    "Outdoor": ["traffic light", "fire hydrant", "stop sign", "parking meter",
                "bench"],
    "Animals": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                "bear", "zebra", "giraffe"],
    "Accessories": ["backpack", "umbrella", "handbag", "tie", "suitcase"],
    "Sports": ["frisbee", "skis", "snowboard", "sports ball", "kite",
               "baseball bat", "baseball glove", "skateboard", "surfboard",
               "tennis racket"],
    "Kitchen": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
    "Food": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot",
             "hot dog", "pizza", "donut", "cake"],
    "Furniture": ["chair", "couch", "potted plant", "bed", "dining table",
                  "toilet"],
    "Electronics": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone"],
    "Appliances": ["microwave", "oven", "toaster", "sink", "refrigerator"],
    "Indoor": ["book", "clock", "vase", "scissors", "teddy bear", "hair drier",
               "toothbrush"],
}

# Everyday words that mean a COCO class. Keys are lowercase.
SYNONYMS = {
    "man": "person", "woman": "person", "boy": "person", "girl": "person",
    "child": "person", "kid": "person", "baby": "person", "people": "person",
    "human": "person", "face": "person", "player": "person",
    "bike": "bicycle", "cycle": "bicycle",
    "automobile": "car", "taxi": "car", "van": "car", "suv": "car",
    "motorbike": "motorcycle", "scooter": "motorcycle",
    "plane": "airplane", "aeroplane": "airplane", "jet": "airplane",
    "aircraft": "airplane",
    "lorry": "truck", "pickup": "truck", "ship": "boat", "sailboat": "boat",
    "tram": "train", "locomotive": "train",
    "stoplight": "traffic light", "hydrant": "fire hydrant",
    "kitten": "cat", "puppy": "dog", "cattle": "cow", "bull": "cow",
    "pony": "horse", "lamb": "sheep", "teddy": "teddy bear",
    "bag": "handbag", "purse": "handbag", "rucksack": "backpack",
    "luggage": "suitcase", "necktie": "tie",
    "ball": "sports ball", "football": "sports ball", "soccer ball": "sports ball",
    "basketball": "sports ball", "tennis ball": "sports ball",
    "racket": "tennis racket", "racquet": "tennis racket", "bat": "baseball bat",
    "glove": "baseball glove", "ski": "skis", "skate board": "skateboard",
    "surf board": "surfboard", "snow board": "snowboard",
    "glass": "wine glass", "mug": "cup", "plate": "bowl", "dish": "bowl",
    "hotdog": "hot dog", "doughnut": "donut", "burger": "sandwich",
    "sofa": "couch", "settee": "couch", "table": "dining table", "desk": "dining table",
    "plant": "potted plant", "flower pot": "potted plant", "toilet seat": "toilet",
    "television": "tv", "monitor": "tv", "screen": "tv", "notebook": "laptop",
    "computer": "laptop", "phone": "cell phone", "mobile": "cell phone",
    "smartphone": "cell phone", "mobile phone": "cell phone",
    "remote control": "remote", "fridge": "refrigerator", "freezer": "refrigerator",
    "stove": "oven", "hair dryer": "hair drier", "hairdryer": "hair drier",
}

CLASSES = [c for group in GROUPS.values() for c in group]


LOOKUP = {**{c: c for c in CLASSES}, **SYNONYMS}
IRREGULAR = {"men": "man", "women": "woman", "children": "child"}


def singular(word: str) -> str:
    if word in IRREGULAR:
        return IRREGULAR[word]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def classes_in(text: str) -> set[str]:
    """Return every COCO class named in free text: a word, list or sentence.

    Longer phrases win, so "hot dog" is not read as a dog, and other words
    are ignored, so "a red car" counts as car.
    """
    words = re.findall(r"[a-z]+", text.lower())
    used = [False] * len(words)
    found: set[str] = set()
    for size in (3, 2, 1):
        for i in range(len(words) - size + 1):
            if any(used[i:i + size]):
                continue
            part = words[i:i + size]
            for phrase in (" ".join(part), " ".join(map(singular, part))):
                if phrase in LOOKUP:
                    found.add(LOOKUP[phrase])
                    used[i:i + size] = [True] * size
                    break
    return found


def canonical(label: str) -> str | None:
    """Return one COCO class named in the label, or None."""
    return min(classes_in(label), default=None)
