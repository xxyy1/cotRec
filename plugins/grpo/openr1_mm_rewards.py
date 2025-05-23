import random
import re
from typing import List
from redaccel.tuner.train.grpo.rewards import rewards_registry
from redaccel.tuner.train.grpo.rewards.base import GRPORewards
from math_verify import parse, verify


@rewards_registry.register(alias=["openr1_mm_accuracy"])
class OpenR1MMAccuracy(GRPORewards):
    def __call__(
        self, prompts: List[str], completions: List[str], solutions: List[str], **kwargs
    ) -> List[float]:
        """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
        rewards = []

        for content, sol in zip(completions, solutions):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r"<answer>(.*?)</answer>", sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r"<answer>(.*?)</answer>", content)
                    student_answer = (
                        content_match.group(1).strip() if content_match else content.strip()
                    )

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail

            rewards.append(reward)

            # if os.getenv("DEBUG_MODE") == "true":
            #     log_path = os.getenv("LOG_PATH")
            #     with open(log_path, "a") as f:
            #         f.write(
            #             f"------------- {current_time} Accuracy reward: {reward} -------------\n"
            #         )
            #         f.write(f"Content: {content}\n")
            #         f.write(f"Solution: {sol}\n")
        return rewards
