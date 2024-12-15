# sleep 6000
# echo "Start"

python method_ablation_graph.py \
    --job-folder /workspace/home/hoangpv4/fact_checking_with_cognitive_graph_rag/main_methods/benchmark_results/benchmark_results_ablation_graph \
    --preprocessed-dataset-path /workspace/home/hoangpv4/fact_checking_with_cognitive_graph_rag/main_methods/processed_datasets/process_dataset_main \
    --dataset hover \
    --partition 2hop \
    --n-workers 20 \
    --temperature 0.5 \
    --llm-host-address http://10.254.138.191:9002 \
    --n-results 5

python method_ablation_graph.py \
    --job-folder /workspace/home/hoangpv4/fact_checking_with_cognitive_graph_rag/main_methods/benchmark_results/benchmark_results_ablation_graph \
    --preprocessed-dataset-path /workspace/home/hoangpv4/fact_checking_with_cognitive_graph_rag/main_methods/processed_datasets/process_dataset_main \
    --dataset hover \
    --partition 3hop \
    --n-workers 20 \
    --temperature 0.5 \
    --llm-host-address http://10.254.138.191:9002 \
    --n-results 5

python method_ablation_graph.py \
    --job-folder /workspace/home/hoangpv4/fact_checking_with_cognitive_graph_rag/main_methods/benchmark_results/benchmark_results_ablation_graph \
    --preprocessed-dataset-path /workspace/home/hoangpv4/fact_checking_with_cognitive_graph_rag/main_methods/processed_datasets/process_dataset_main \
    --dataset hover \
    --partition 4hop \
    --n-workers 20 \
    --temperature 0.5 \
    --llm-host-address http://10.254.138.191:9002 \
    --n-results 5

python method_ablation_graph.py \
    --job-folder /workspace/home/hoangpv4/fact_checking_with_cognitive_graph_rag/main_methods/benchmark_results/benchmark_results_ablation_graph \
    --preprocessed-dataset-path /workspace/home/hoangpv4/fact_checking_with_cognitive_graph_rag/main_methods/processed_datasets/process_dataset_main \
    --dataset feverous \
    --partition feverous_sentence \
    --n-workers 20 \
    --temperature 0.5 \
    --llm-host-address http://10.254.138.191:9002 \
    --n-results 5