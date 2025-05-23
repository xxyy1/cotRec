import random
import re
from typing import List
from redaccel.tuner.train.grpo.rewards import rewards_registry
from redaccel.tuner.train.grpo.rewards.base import GRPORewards


def extract_nums_and_gt(text: str):
    # 正则表达式提取列表
    list_pattern = r"\[([\d,\s]+)\]"
    list_match = re.search(list_pattern, text)
    numbers = None
    if list_match:
        # 提取列表中的数字并转换为整数
        numbers = list(map(int, list_match.group(1).split(",")))

    # 正则表达式提取整数
    int_pattern = r"equals\s*(\d+)"
    int_match = re.search(int_pattern, text)
    target_number = None
    if int_match:
        target_number = int(int_match.group(1))
    return numbers, target_number


def ensure_think_prefix(s):
    s = s.strip()
    think = "<think>"
    if s[: len(think)] != think:
        return think + s
    return s


@rewards_registry.register()
class MiniR1Format(GRPORewards):
    def __call__(
        self, prompts: List[str], completions: List[str], solutions: List[str], **kwargs
    ) -> List[float]:
        """
        Format: <think>...</think><answer>...</answer>
        Args:
            completions (list[str]): Generated outputs

          Returns:
              list[float]: Reward scores
        """
        rewards = []

        for completion in completions:

            try:
                completion = ensure_think_prefix(completion)
                # Check if the format is correct
                regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

                m = re.search(regex, completion, re.DOTALL)
                # if the format is not correct, reward is 0
                if m is None or len(m.groups()) != 2:
                    rewards.append(0.0)
                else:
                    rewards.append(1.0)
            except Exception:
                rewards.append(0.0)
        return rewards


@rewards_registry.register()
class MiniR1Equation(GRPORewards):
    def __call__(
        self, prompts: List[str], completions: List[str], solutions: List[str], **kwargs
    ) -> List[float]:
        """
        Evaluates completions based on: Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            prompts (list[str]): User Inputs

        Returns:
            list[float]: Reward scores
        """
        rewards = []

        for prompt, completion in zip(prompts, completions):
            try:
                completion = ensure_think_prefix(completion)
                numbers, gt = extract_nums_and_gt(prompt)
                # Check if the format is correct
                match = re.search(r"<answer>(.*?)<\/answer>", completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r"^[\d+\-*/().\s]+$"
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builtins__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards
