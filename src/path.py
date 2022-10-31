# -*- coding: utf-8 -*-
from pathlib import Path
from os.path import abspath


class path:
    def __init__(self, set, trial=None):
        # Set paths
        # root
        self.root = Path(abspath(__file__)).absolute().parents[1]

        # dataset directory
        self.dataset = self.root / "datasets" / str(set)
        self.train = self.dataset / "train"
        self.valid = self.dataset / "valid"
        self.test = self.dataset / "test"

        # directory for saving models
        self.model_dir = self.root / "models" / str(set)
        if trial:
            self.trial = trial
        else:
            self.trial = self.get_trial()
        self.model = self.model_dir / self.trial

        # directory for reporting
        self.report_dir = self.root / "results" / str(set)
        self.report = self.report_dir / self.trial

    def get_trial(self):
        tlist = [0]
        # Get trial* list in root/models
        tmp = self.model_dir.glob("trial*")
        tlist.extend([int(s.name.replace("trial", "")) for s in tmp])

        # Get new trial number
        n = max(tlist) + 1
        trial = "trial" + str(n).zfill(2)

        return trial


if __name__ == "__main__":
    p = path("subset_marmoset_23ue")

    p_model = p.model
    p_model_parent = p_model.parent

    print(p_model_parent)
