import pandas as pd
import subprocess
from multiprocessing import Pool
from backend_request_func import remove_prefix

def run_benchmark(url):
    url[0] = "https://" + remove_prefix(url[0], prefix="http://")
    cmd = f"export OPENAI_API_KEY=callinferenceaiforgpu && python3 benchmark_serving.py \
            --backend sglang --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
            --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
            --num-prompts 20480 --base-url {url[0]} --request-rate=5.0 --random-input-len=2048"
    result = subprocess.call(cmd, shell=True)
    return result

if __name__ == '__main__':
    input_csv = pd.read_csv("API_urls_local24.csv")
    url_list = input_csv[['ing','port']].values.tolist()
    results = []
    with Pool(processes=len(url_list)) as pool:
        r = pool.map_async(run_benchmark, url_list, callback=results.append)
        r.wait()

#    with open("parallel.txt", "a+") as f:
#        for res in results:
#            for r in res:
#                f.write(str(r))
#            f.write("================================")
