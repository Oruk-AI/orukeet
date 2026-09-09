"""Stage-2 input_cfg: stage-1 weights rescaled to counter Latvian leakage.
Down-weight lv; up-weight languages whose CV test regressed / show Latin-script leakage."""
import yaml

FIN = "/work/users/nathanroll/parakeet-ft/manifests/final"
MULT = {"lv": 0.4, "uk": 1.8, "ru": 1.5, "sk": 1.8, "sl": 1.6, "cs": 1.5, "el": 1.5, "bg": 1.4,
        "fi": 1.6, "mt": 1.6, "pl": 1.4, "et": 1.3, "hu": 1.2, "it": 1.1}
groups = yaml.safe_load(open(f"{FIN}/train_input_cfg.yaml"))
for g in groups:
    g["weight"] *= MULT.get(g["tags"]["lang"], 1.0)
tot = sum(g["weight"] for g in groups)
for g in groups:
    g["weight"] = round(g["weight"] / tot, 5)
yaml.safe_dump(groups, open(f"{FIN}/stage2_input_cfg.yaml", "w"), sort_keys=False)
for g in sorted(groups, key=lambda g: -g["weight"]):
    print(f"{g['tags']['lang']:3} {100*g['weight']:6.2f}%")
