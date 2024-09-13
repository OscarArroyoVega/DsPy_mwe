# minimum working example for DSPy


import os 
import sys
import json
import requests
from typing import AsyncGenerator, Literal, Dict, Generator, List, Optional, Union
from typing_extensions import Literal
import openai
from unify.clients import Unify as UnifyClient
import dsp 
import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot
from dsp import Unify
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

# Set up the LM.
class Model_Unify(dsp.Unify):
    def __call__(self, *args, **kwargs):
        # Implement the method here
        print("Called with", args, kwargs)
        return "Result"
model = Model_Unify(endpoint='claude-3-haiku@antrophic', max_tokens=250, api_key="YOUR_API_KEY")

dspy.settings.configure(lm=model)

# Load math questions from the GSM8K dataset.
gsm8k = GSM8K()
gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:10]

print(gsm8k_trainset)
print("--_"*20)


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)
    


# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 4-shot examples of our CoT program.
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

# Optimize! Use the `gsm8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset)
print("--_"*20)

# Set up the evaluator, which can be used multiple times.
evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)

# Evaluate our `optimized_cot` program.
evaluate(optimized_cot)
print("--_"*20)

model.inspect_history(n=1)
