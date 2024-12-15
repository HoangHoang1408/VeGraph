from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

def multi_process_task_dict(task_dictionary, num_workers, show_progress=True):
    final_results = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for id_, task in task_dictionary.items():
            to_submit_task = lambda id_=id_, task=task: {"id": id_, "task_result": task()}
            futures.append(executor.submit(to_submit_task))
        if show_progress:
            with tqdm(total=len(task_dictionary)) as progress_bar:
                for future in as_completed(futures):
                    result = future.result()
                    final_results[result['id']] = result['task_result']
                    progress_bar.update(1)
        else:
            for future in as_completed(futures):
                result = future.result()
                final_results[result['id']] = result['task_result']
    return final_results
                    
