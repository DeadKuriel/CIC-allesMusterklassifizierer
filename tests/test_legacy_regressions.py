import numpy as np
import pytest
from allesmusterklassifizierer.metrics.confusion import confusion_matrix_df
from allesmusterklassifizierer.validation.holdout import HoldoutSplitter
from allesmusterklassifizierer.errors import ValidationError

def test_legacy_confusion_does_not_silently_omit_labels():
    with pytest.raises(ValueError): confusion_matrix_df(np.array([0,2]),np.array([0,1]),[0,1])

def test_singletons_fail_instead_of_looping():
    with pytest.raises(ValidationError): list(HoldoutSplitter(test_size=.8,stratify=True).split(np.array([0,1,2])))
