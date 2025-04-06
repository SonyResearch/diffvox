from pathlib import Path
import yaml


data_dir = "/data/medley1/"

yamls = list(Path(data_dir).glob("v[1-2]*/**/*.yaml"))
print(len(yamls))

dict_set = set()
tracks = []
for filename in yamls:
    with open(filename) as f:
        hierachy = yaml.safe_load(f)
    if hierachy["instrumental"] == "yes":
        continue
    raw_dir = hierachy["raw_dir"]
    stem_dir = hierachy["stem_dir"]
    for stem, stem_v in hierachy["stems"].items():
        instr_name = stem_v["instrument"]
        if len(stem_v["raw"]) == 1 and any(
            map(
                lambda keyword: keyword in instr_name,
                ("speaker", "singer", "vocal", "voice", "speech"),
            )
        ):
            raw_track, *_ = tuple(stem_v["raw"].values())
            tracks.append(
                str(filename.parent)
                # (
                #     str(filename.parent / raw_dir / raw_track["filename"]),
                #     str(filename.parent / stem_dir / stem_v["filename"]),
                # )
            )
            break
print(len(tracks))

with open("medley_vocals_list.txt", "w") as f:
    for l in tracks:
        f.write(l + "\n")
