from transformers import AutoTokenizer, AutoModelForCausalLM
from tools.analyst_tools import *
import torch 
from huggingface_hub import login

login()
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map = "auto")


data = load_data("data/data.csv")

df = data["dataframe"]
schema = data["schema"]


def generate_response(question: str) -> str:
    prompt = f"""<bos><start_of_turn>user
        You are a data analyst. Answer the question using the data context below.

        Available tools:
        - load_data(file_path): loads a CSV file and returns dataframe and schema
        - run_eda(df): runs exploratory data analysis and returns summary statistics
        - generate_chart(df, type, x, y): generates a chart, types are bar, line, scatter, hist
        - run_python_snippet(code, df): executes Python/pandas code, store result in _result
        - get_collection(): returns stored memory
        - update_collection(key, value): stores a value in memory

        Question: {question}
        <end_of_turn>
        <start_of_turn>model
        """
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens =200, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    return response

if __name__ == "__main__":
    while True:

        question = input("Ask a question about the data (or type 'exit' to quit): ")
        if question.lower() in ['exit', 'quit']:
            break

        response = generate_response(question)
        print(response)