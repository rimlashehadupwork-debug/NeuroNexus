import pandas as pd
import pytest
from modules.parkinson_model import load_uci_voice_csv


def test_loader_extracts_subject_ids_and_features():
    rows=[]
    for subject in range(1, 7):
        label=1 if subject in (1, 2, 3) else 0
        for rep in (1, 2):
            rows.append({"name":f"phon_R01_S{subject:02d}_{rep}","feature_a":subject*0.1,"feature_b":rep*0.2,"feature_c":1.0,"feature_d":2.0,"feature_e":3.0,"status":label})
    X,y,groups=load_uci_voice_csv(pd.DataFrame(rows))
    assert X.shape == (12,5)
    assert set(y)=={0,1}
    assert len(set(groups))==6
    assert groups[0]==groups[1]


def test_loader_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="name.*status"):
        load_uci_voice_csv(pd.DataFrame({"a":[1,2],"b":[2,3]}))
