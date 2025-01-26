"""Export Unsloth model to PEFT format.
"""

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="SYX/mistral_based_claim_extractor",
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)

model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)