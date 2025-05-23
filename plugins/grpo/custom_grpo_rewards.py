import random
import re
from typing import List
from redaccel.tuner.train.grpo.rewards import rewards_registry
from redaccel.tuner.train.grpo.rewards.base import GRPORewards

@rewards_registry.register(alias=["rec_think_reward"])
class RecThinkGRPOReward(GRPORewards):

    def __call__(
        self, prompts: List[str], completions: List[str], solutions: List[str], **kwargs
    ) -> List[float]:
        rewards = []
        for content, sol in zip(completions, solutions):
            reward = 0.0
            match = re.search(r"<think>(.*?)<\/think>", content)
            if match is None:
                rewards.append(reward)
                continue
            answer = match.group(1).strip()
            reward += 0.5
            answer_ids_match = re.search(r"推荐理由是(.*?)", answer)
            if not answer_ids_match:
                rewards.append(reward)
                continue
            reward += 1.0
            rewards.append(reward)
        return rewards

@rewards_registry.register(alias=["rec_fomat_reward"])
class RecFormatGRPOReward(GRPORewards):

    @staticmethod
    def detect_repeated_patterns1(text, target_substrings):
        """
        检测字符串中是否存在连续重复的目标子串（如“用户”、“用户中间”、“用户id”等）。
    
        参数:
        text (str): 需要检测的字符串
        target_substrings (list): 需要检测的目标子串列表（如 ["用户", "用户中间", "用户id"]）
    
        返回:
        bool: 存在重复模式返回True，否则返回False
        """
        for substr in target_substrings:
            # 构建正则表达式模式，匹配目标子串连续重复至少两次
            pattern = rf'({re.escape(substr)}){{2,}}'
            if re.search(pattern, text):
                return True
        return False

    def __call__(
        self, prompts: List[str], completions: List[str], solutions: List[str], **kwargs
    ) -> List[float]:
        rewards = []
        targets = ['用户', '用户中间', '用户id', '用户11', '用户最感兴趣的笔记id'] 
        for content, sol in zip(completions, solutions):
            reward = 0.0
            # 提取<answer>标签中的内容
            match = re.search(r"<answer>(.*?)<\/answer>", content)
            if match is None:
                rewards.append(reward)
                continue
            answer = match.group(1).strip()
            is_repeated_str = self.detect_repeated_patterns1(answer, targets) 
            if is_repeated_str is False:
                reward += 0.0

            answer_ids_match = re.search(r"最感兴趣的笔记id是\[(.*?)\]", answer)
            if not answer_ids_match:
                rewards.append(reward)
                continue
            reward += 1.0
            rewards.append(reward)
        return rewards 

@rewards_registry.register(alias=["rec_reward"])
class MyRecGRPOReward(GRPORewards):
    @staticmethod
    def weighted_lcs(seq1, seq2):
        """
        计算两个序列的加权最长公共子序列的总权重，权重函数 w(i, j) = 1 / (max(i, j) + 1)
        """
        m, n = len(seq1), len(seq2)
        # 初始化DP数组，大小为 (m+1) x (n+1)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        
        # 动态规划计算加权LCS
        for i in range(m):
            for j in range(n):
                if seq1[i] == seq2[j]:
                    # 当匹配时，加上对应位置的权重
                    w = 1.0 / (max(i, j) + 1)
                    dp[i+1][j+1] = dp[i][j] + w
                else:
                    # 不匹配时，取前面较优结果
                    dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
        return dp[m][n]
    @staticmethod
    def normalized_weighted_lcs(seq1, seq2):
        """
        计算归一化的加权LCS得分，归一化因子为较短序列前 k 项（k=min(len(seq1), len(seq2))）的最大可能加权和
        """
        if not seq1 or not seq2:
            return 0.0
        
        weighted_score = MyRecGRPOReward.weighted_lcs(seq1, seq2)
        # 计算较短序列前 k 项的最大可能得分，即每个位置完美匹配时的权重和
        k = max(len(seq1), len(seq2))
        max_possible = sum(1.0 / (i + 1) for i in range(k))
        
        return weighted_score / max_possible, weighted_score

    @staticmethod
    def detect_repeated_patterns(text, target_substrings):
        """
        bool: 存在重复模式返回True，否则返回False
        """
        for substr in target_substrings:
            pattern = rf'({re.escape(substr)}){{2,}}'
            if re.search(pattern, text):
                return True
        return False

    def __call__(
        self, prompts: List[str], completions: List[str], solutions: List[str], **kwargs
    ) -> List[float]:
        """reward class invoke.

        Parameters
        ----------
        prompts : List[str]
            user input prompts, len = batch size
            用户输入的 prompt
        completions : List[str:]
            grpo completions
            policy 模型 rollout 输出的答案，长度和 prompts 一致
        solutions : List[str]
            ground truth answers
            用户输入的数据集中的标准答案，长度和 prompts 一致
        **kwargs :
            - metadata: List[dict] 如果用户输入数据集中带了 metadata，可以直接取用
            - ...

        Returns
        -------;
        float reward score list, length must be equal to batch size
        """
        rewards = []
        targets = ['用户用户', '用户中间', '用户id'] 
        for content, sol in zip(completions, solutions):
            reward = 0.0
            # 提取<answer>标签中的内容
            match = re.search(r"<answer>(.*?)<\/answer>", content)
            if match is None:
                rewards.append(reward)
                continue
            answer = match.group(1).strip()
            # 重复
            is_repeated_str = self.detect_repeated_patterns(answer, targets) 
            if is_repeated_str is True:
                rewards.append(reward)
                continue

            reward += 0.05

            t2_match = re.search(r"<think>(.*?)<\/think>", content)
            #t2_answer = t2_match.group(1).strip()
            if t2_match is not None:
                reward += 0.05

            
            # 解析answer中的笔记ID列表
            answer_ids_match = re.search(r"最感兴趣的笔记id是\[(.*?)\]", answer)
            if not answer_ids_match:
                rewards.append(reward)
                continue
            reward += 0.05
            answer_ids_str = answer_ids_match.group(1)
            answer_ids = [id.strip() for id in answer_ids_str.split(',')]
            
            # 解析solution中的笔记ID列表
            sol_ids_match = re.search(r"最感兴趣的笔记id是\[(.*?)\]", sol)
            if not sol_ids_match:
                rewards.append(reward)
                continue
            sol_ids_str = sol_ids_match.group(1)
            sol_ids = [id.strip() for id in sol_ids_str.split(',')]
            
            # 计算交集
            intersection = set(answer_ids) & set(sol_ids)
            overlap_count = len(intersection)
            sol_count = len(sol_ids)
            
            # 计算占比并设置奖励
            if len(answer_ids) >= len(sol_ids):
                reward += 0.05
            if overlap_count > 0:
                ratio = overlap_count / sol_count
                reward = reward + ratio * 0.05
                if answer_ids[0] == sol_ids[0]:
                    reward += 0.3
            if answer_ids_str == sol_ids_str or (len(answer_ids) >= len(sol_ids) and sol_ids == answer_ids[:len(sol_ids)]):
                reward += 1.0
     
            # 位置与序
            if len(answer_ids) > 2 and len(sol_ids) > 2:
                score, weighted_score = self.normalized_weighted_lcs(answer_ids, sol_ids)
                reward += score
            
            rewards.append(reward)
        return rewards
