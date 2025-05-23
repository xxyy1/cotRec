import json
from typing import Dict, Optional, List
from loguru import logger
import numpy as np
from transformers import EvalPrediction, PreTrainedTokenizer, ProcessorMixin

try:

    from redaccel.tuner.train.sft.metric import (
        CausalLMMetric,
        causal_lm_metric_registry,
        numpify,
        IGNORE_INDEX,
        SmoothingFunction,
        Rouge,
        sentence_bleu,
    )

    @causal_lm_metric_registry.register()
    class DummyMetricForCausalLM(CausalLMMetric):
        tokenizer: PreTrainedTokenizer
        processor: Optional[ProcessorMixin]

        def __call__(
            self, eval_preds: EvalPrediction, compute_result: bool = True
        ) -> Optional[Dict[str, float]]:
            key = "dummy_acc"
            if key not in self.score_dict:
                self.score_dict[key] = []

            preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
            preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
            labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            for pred, label in zip(decoded_preds, decoded_labels):
                self.score_dict["dummy_acc"].append(len(pred) == len(label))

            if compute_result:
                result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}
                self.score_dict[key] = []
                return result

    @causal_lm_metric_registry.register()
    class ComputeRedbiSimilarity(CausalLMMetric):
        r"""
        Computes text similarity scores and supports `batch_eval_metrics`.
        Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
        """

        def _dump(self) -> Optional[Dict[str, float]]:
            result = None
            if hasattr(self, "score_dict"):
                result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}
            self.score_dict = {
                "rouge-1": [],
                "rouge-2": [],
                "rouge-l": [],
                "bleu-4": [],
                "accuracy": [],
            }
            return result

        def _get_redbi_json(self, predict, name_key) -> List[dict]:
            from json_repair import loads, repair_json

            if len(predict) == 0:
                return []
            res = []
            predict = predict.replace("```json`", "").replace("```", "")
            json_obj = ""
            try:
                json_obj = json.loads(repair_json(predict))
            except OSError:
                return res
            if isinstance(json_obj, list):
                if len(json_obj) == 0:
                    return res
                else:
                    json_obj = json_obj[0]
            if isinstance(json_obj, dict) and name_key not in json_obj.keys():
                return res
            for i in json_obj[name_key]:
                if "排序" in i.keys() and i["排序"] in ["", "不排序"]:
                    i.pop("排序")
                if "搜索词(小写去空格)" == i["字段名称"]:
                    i["字段名称"] = "搜索词(区分大小写和空格)"
                res.append(i)
            return res

        def get_redbi_json(self, predict, name_key) -> List[dict]:
            try:
                return self._get_redbi_json(predict, name_key)
            except BaseException as e:
                logger.error(f"get redbi json failed, error: {e}")
                return []

        def json_acc(self, predict, label):
            map_key = {
                "dimension_correct": "维度列表",
                "measure_include": "指标列表",
                "filter_correct": "筛选条件",
            }
            name_key = "维度列表"
            predict_res = self.get_redbi_json(predict, name_key)
            gt_res = self.get_redbi_json(label, name_key)
            predict_res_list = []
            for item_dict in predict_res:
                if item_dict["字段名称"] in ["时间", "日期", "dtm"]:
                    continue
                else:
                    predict_res_list.append(item_dict)
            gt_res_list = []
            for item_dict in gt_res:
                if item_dict["字段名称"] in ["时间", "日期", "dtm"]:
                    continue
                else:
                    gt_res_list.append(item_dict)
            pred_set = {
                frozenset((k, tuple(v) if isinstance(v, list) else v) for k, v in d.items())
                for d in predict_res_list
            }
            gt_set = {
                frozenset((k, tuple(v) if isinstance(v, list) else v) for k, v in d.items())
                for d in gt_res_list
            }
            is_equal = pred_set == gt_set
            logger.info(
                f"revy predict_res-{predict_res}, gt_res-{gt_res}, pred_set-{pred_set}, gt_set-{gt_set} is_equal-{is_equal}"
            )
            return is_equal

        def __call__(
            self, eval_preds: EvalPrediction, compute_result: bool = True
        ) -> Optional[Dict[str, float]]:
            import jieba

            preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
            preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
            labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            for pred, label in zip(decoded_preds, decoded_labels):
                pred_ori = pred
                label_ori = label
                pred = pred.split("维度列表")[-1]
                label = label.split("维度列表")[-1]
                hypothesis = list(jieba.cut(pred))
                reference = list(jieba.cut(label))
                if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                    result = {
                        "rouge-1": {"f": 0.0},
                        "rouge-2": {"f": 0.0},
                        "rouge-l": {"f": 0.0},
                        "accuracy": {"f": 0.0},
                    }
                else:
                    rouge = Rouge()
                    scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                    result = scores[0]
                for k, v in result.items():
                    self.score_dict[k].append(round(v["f"] * 100, 4))
                bleu_score = sentence_bleu(
                    [list(label)], list(pred), smoothing_function=SmoothingFunction().method3
                )
                self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))
                self.score_dict["accuracy"].append(self.json_acc(pred_ori, label_ori))
            if compute_result:
                return self._dump()

except BaseException as e:
    logger.error(f"load plugin failed, error: {e}")
    logger.warning("please upgrade redaccel to the latest version")
