"""
TO-DO:
    
    1. Quality control and cleaning of the bibles:  
        - Match verse number                        
        - Solve the n in acholi                     
        - Find and remove empty verses          
"""
import os
import sys
from pathlib import Path
import re

if __name__ == "__main__":

    p = Path(sys.argv[1])
    bible_dict = {}
    filenames = []
    for file in p.iterdir():
        if file.name.startswith("."):
            continue
        else:
            filenames.append(file.name)

        with file.open() as f:
            lines = f.readlines()

            # Remove comments
            text = [line for line in lines if line.startswith("#") == False]

            # Standardise spelling
            if file.name == "ach-x-bible.txt":
                text = [re.sub('ŋ', 'ng', line.lower()) for line in text]

            # Remove old testament from Luo
            if file.name == 'luo-x-bible-dc.txt':
                four = [line for line in text if line.startswith("4")]
                five = [line for line in text if line.startswith("5")]
                six = [line for line in text if line.startswith("6") and line.startswith("67") == False and
                line.startswith("68") == False and line.startswith("69") == False]
                text = four + five + six

            # Create dict with dicts
            id_verse = {}
            for line in text:
                id = line.split()[0]
                verse = " ".join(line.split()[1:])
                id_verse[id] = verse
        bible_dict[file.name] = id_verse
    print(bible_dict.keys())

    # Align verses
    id_to_del = []
    for lang, dict in bible_dict.items():
        for id, verse in dict.copy().items():
            if id not in bible_dict[filenames[2]]:
                id_to_del.append(id)
            if verse:
               continue
            else:
                id_to_del.append(id)
    for id in id_to_del:
        for v in bible_dict.values():
            try:
                del v[id]
            except:
                continue

    # Join into string and write to files
    i = iter(range(len(filenames)))
    for d in bible_dict.values():
        text = '\n'.join(d.values())
        filename = sys.argv[1] + "clean_" + filenames[next(i)]
        with open(filename, 'w') as f:
            f.write(text)

        







