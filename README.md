# BordaRAG: Resolving Knowledge Conflict in Retrieval-Augmented Generation via Borda Voting Process

This is the official implementation of BordaRAG.

The dataset is accessible on Hugging Face via the link below:
- NQ: https://huggingface.co/datasets/google-research-datasets/nq_open
- PopQA: https://huggingface.co/datasets/akariasai/PopQA
- TriviaQA: https://huggingface.co/datasets/mandarjoshi/trivia_qa

The Google Search API can be accessed by: 
```
https://www.googleapis.com/customsearch/v1?key={YOUR_KEY}&q={SEARCH_WORDS}&cx={YOUR_CX}&start={10}&num={10}
```
The test dataset created by author can be accessed by: https://drive.google.com/drive/folders/1BmDIH5E8boxuIIBANB3-tRUiOd05pAHd?usp=sharing

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
