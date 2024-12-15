python method_direct_checking.py \
    --llm-host-address http://10.254.138.191:9148 \
    --job-folder /workspace/home/hoangpv4/fact_checking_with_cognitive_graph_rag/main_methods/benchmark_results/benchmark_results_direct \
    --preprocessed-dataset-path /workspace/home/hoangpv4/fact_checking_with_cognitive_graph_rag/main_methods/processed_datasets/process_dataset_main \
    --dataset hover \
    --partition 2hop \
    --n-workers 50 \
    --temperature 0.8 \
    --n-results 10 \

# python method_direct_checking.py \
#     --llm-host-address http://10.254.138.191:9148 \
#     --job-folder /workspace/home/hoangpv4/fact_checking_with_cognitive_graph_rag/main_methods/benchmark_results/benchmark_results_direct \
#     --preprocessed-dataset-path /workspace/home/hoangpv4/fact_checking_with_cognitive_graph_rag/main_methods/processed_datasets/process_dataset_main \
#     --dataset hover \
#     --partition 3hop \
#     --n-workers 50 \
#     --temperature 0.5 \
#     --n-results 5 \

# python method_direct_checking.py \
#     --llm-host-address http://10.254.138.191:9148 \
#     --job-folder /workspace/home/hoangpv4/fact_checking_with_cognitive_graph_rag/main_methods/benchmark_results/benchmark_results_direct \
#     --preprocessed-dataset-path /workspace/home/hoangpv4/fact_checking_with_cognitive_graph_rag/main_methods/processed_datasets/process_dataset_main \
#     --dataset hover \
#     --partition 4hop \
#     --n-workers 50 \
#     --temperature 0.5 \
#     --n-results 5 \

# python method_direct_checking.py \
#     --llm-host-address http://10.254.138.191:9148 \
#     --job-folder /workspace/home/hoangpv4/fact_checking_with_cognitive_graph_rag/main_methods/benchmark_results/benchmark_results_direct \
#     --preprocessed-dataset-path /workspace/home/hoangpv4/fact_checking_with_cognitive_graph_rag/main_methods/processed_datasets/process_dataset_main \
#     --dataset feverous \
#     --partition feverous_sentence \
#     --n-workers 50 \
#     --temperature 0.5 \
#     --n-results 5 \