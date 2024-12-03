import torch
from copy import deepcopy

class RGRanker:

    def __init__(self, tokenizer, model, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.PROMPT = """
        Passage: {passage}
        Query: {query}
        Does the passage
        answer the query?
        Answer Yes/No.
        """
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def rerank(self, hits, query):
        reranked_hits = [(docid, doc, self.score_query_passage(query, doc)) for docid, doc, _ in hits]
        return sorted(reranked_hits, reverse=True, key=lambda x: x[2])

    def score_query_passage(self, query, passage):
        # Prepare the input
        input_text = self.PROMPT.format(passage=passage, query=query)
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

        # Generate the output
        outputs = self.model.generate(input_ids, max_length=50, return_dict_in_generate=True, output_scores=True)
        next_token_logits = outputs.scores[0]
        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        if 'yes' in generated_text.lower() or 'Yes' in generated_text.lower():
            Yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
            yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
            yes_prob = max(torch.softmax(next_token_logits, dim=-1)[0, yes_token_id].item(), torch.softmax(next_token_logits, dim=-1)[0, Yes_token_id].item())
            return 1 + yes_prob
        elif 'no' in generated_text.lower() or 'No' in generated_text.lower():
            No_token_id = self.tokenizer.convert_tokens_to_ids("No")
            no_token_id = self.tokenizer.convert_tokens_to_ids("no")
            no_prob = max(torch.softmax(next_token_logits, dim=-1)[0, no_token_id].item(), torch.softmax(next_token_logits, dim=-1)[0, No_token_id].item())
            return 1 - no_prob
        raise ValueError("The model did not output a yes/no answer. Instead answered: " + generated_text)
