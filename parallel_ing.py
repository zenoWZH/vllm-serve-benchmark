import pandas as pd
import subprocess
from multiprocessing import Pool
from backend_request_func import remove_prefix

def run_benchmark(url):
    url[0] = "https://" + remove_prefix(url[0], prefix="http://")
    cmd = f"export OPENAI_API_KEY=token-abc123 && python3 benchmark_serving.py \
            --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct \
            --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
            --num-prompts 40960 --base-url {url[0]} --request_rate=4"
    result = subprocess.call(cmd, shell=True)
    return result

if __name__ == '__main__':
    input_csv = pd.read_csv("API_urls.csv")
    url_list = input_csv[['ing','port']].values.tolist()
    results = []
    with Pool(processes=len(url_list)) as pool:
        r = pool.map_async(run_benchmark, url_list, callback=results.append)
        r.wait()

    with open("parallel.txt", "a+") as f:
        for res in results:
            for r in res:
                f.write(str(r))
            f.write("================================")

