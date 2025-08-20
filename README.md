# BordaRAG: Resolving Knowledge Conflict in Retrieval-Augmented Generation via Borda Voting Process

This is the official implementation of BordaRAG.

Due to copyright restrictions, the experimental data cannot be made publicly available.

### Set Environment

run `pip install -r requirements.txt`.

### Run Command

```
python main.py \
-model_path [your model path] \
-candidate_datapath [your qa data path] \
-candidate_prompt_path [save path for candidate prompts] \
-candidate_result_path [save path for candidate results] \
-preference_datapath [same as candidtae_result_path] \
-preference_result_path [save path for preference scores] \
-qa_datapath [same as preference_result_path] \
-qa_prompt_path [save path for qa prommpt] \
-qa_result_path [save path for qa results] \
-topk [top-k for document filtering]
```

### Evaluate

Use `evaluate.py` for evaluation.
