from typing import Dict
import torch
from loguru import logger

try:
    from redaccel.tuner.train.sft.metric import (
        SeqClsMetric,
        SeqClsProblemType,
        seq_cls_metric_registry,
    )

    @seq_cls_metric_registry.register()
    class DummyMetricForMultiLabel(SeqClsMetric):
        # 任务类型
        PROBLEM_TYPE: SeqClsProblemType = SeqClsProblemType.MULTI_LABEL

        def __call__(self, preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
            """calculate metric for SEQ_CLS task.

            Parameters
            ----------
            preds : torch.Tensor
                model predictions, shape is determined by problem type:
                  - SINGLE_LABEL classification: (batch_size, num_classes)
                  - MULTI_LABEL classification: (batch_size, num_classes)
            labels : torch.Tensor
                ground truth label, shape is determined by problem type:
                  - SINGLE_LABEL classification: (batch_size, ), value in [0, num_classes)
                  - MULTI_LABEL classification: (batch_size, num_classes), value is 0 or1

            Return
            --------
            metrics of Dict[str, float]
            """

            return {"dummy_metric_0": 1.0, "dummy_metric_1": 0.5}

except:
    logger.warning("please upgrade redaccel to the latest version")
