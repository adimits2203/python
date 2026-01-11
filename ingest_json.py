from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments
import torch
import tempfile
import pandas as pd

# Step 1: Preprocess the JSON data
def preprocess_json(json_file):
        data = pd.read_json('widget.json')
        print(data)
        # Filter and format data as needed for training
        return data.to_string()

# Step 2: Tokenize the preprocessed text
def tokenize_text(text, tokenizer):
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenized_text = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    return tokenized_text # Return only input_ids tensor


# Step 3: Fine-tune the GPT model
def fine_tune_gpt(tokenizer, tokenized_text):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )
    # Convert BatchEncoding to string
    tokenized_text_str = '\n'.join(
        [tokenizer.decode(token.tolist(), skip_special_tokens=True) for token in tokenized_text['input_ids']])
    print("Tokenized text:", tokenized_text_str)
    # Save preprocessed text to a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        temp_file.write(tokenized_text_str)
        temp_file_path = temp_file.name
    print("Temporary file path:", temp_file_path)
    dataset = TextDataset(tokenizer=tokenizer, file_path=temp_file_path, block_size=128)
    print("Number of samples in dataset:", len(dataset))
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model()

def answer_question(question, model, tokenizer):
    input_ids = tokenize_text(question, tokenizer)
    # Access the actual tensor of input IDs
    input_ids = input_ids['input_ids']
    input_ids = torch.unsqueeze(input_ids, 0)  # Add batch dimension
    # Use prepare_inputs_for_generation
    inputs = model.prepare_inputs_for_generation(input_ids=input_ids)
    attention_mask = inputs["attention_mask"]  # Define attention_mask here

    # Set token_type_ids explicitly
    inputs["token_type_ids"] = torch.zeros_like(input_ids)  # For GPT2, token_type_ids should be zeros

    output = model.generate(**inputs, max_length=512)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer


# Main function
def main():
    # Step 1: Preprocess the JSON data
    json_file = 'widget.json'
    preprocessed_text = preprocess_json(json_file)

    # Step 2: Tokenize the preprocessed text
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenized_text = tokenize_text(preprocessed_text, tokenizer)

    # Step 3: Fine-tune the GPT model
    fine_tune_gpt(tokenizer, tokenized_text)

    # Initialize GPT2 model
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Ask questions and generate answers
    while True:
        question = input("Ask a question (or 'quit' to exit): ")
        if question == 'quit':
            break
        answer = answer_question(question, model, tokenizer)
        print(f"Answer: {answer}")
if __name__ == "__main__":
    main()
