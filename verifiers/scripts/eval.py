import argparse
import importlib
import importlib.util
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import Dataset
from openai import OpenAI

import verifiers as vf
from verifiers.utils.tool_utils import sanitize_tool_calls
from verifiers.types import GenerateOutputs


def _load_completions_from_path(path: str) -> list:
    """
    Load completions from a .jsonl or .json file.
    - For .jsonl: expects each line to be a JSON object with a "completion" or "text" field,
      or a raw JSON string representing the completion.
    - For .json: expects either a list of completions or an object with a "completion" key (list).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Completions file not found: {path}")

    completions: list = []
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, str):
                        completions.append(obj)
                    elif isinstance(obj, dict):
                        if "completion" in obj:
                            completions.append(obj["completion"])
                        elif "text" in obj:
                            completions.append(obj["text"])
                        else:
                            raise ValueError(
                                "JSONL object must have 'completion' or 'text' field, or be a string"
                            )
                    else:
                        raise ValueError(
                            "JSONL line must be a string or object with 'completion'/'text'"
                        )
                except json.JSONDecodeError:
                    raise ValueError(
                        "Invalid JSONL: lines must be valid JSON values (string or object)"
                    )
    elif p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, list):
                completions = obj
            elif isinstance(obj, dict):
                if "completion" in obj and isinstance(obj["completion"], list):
                    completions = obj["completion"]
                elif "completions" in obj and isinstance(obj["completions"], list):
                    completions = obj["completions"]
                else:
                    raise ValueError(
                        "JSON must be a list of completions or contain 'completion'/'completions' list"
                    )
            else:
                raise ValueError(
                    "JSON must be a list of completions or an object with 'completion'/'completions' list"
                )
    else:
        raise ValueError("Unsupported completions file type. Use .jsonl or .json")

    return completions


def eval_environment(
    env: str,
    env_args: dict,
    env_dir_path: str,
    endpoints_path: str,
    model: str,
    api_key_var: str,
    api_base_url: str,
    num_examples: int,
    rollouts_per_example: int,
    max_concurrent_requests: int,
    max_tokens: int,
    temperature: float | None,
    verbose: bool,
    save_dataset: bool,
    save_to_hf_hub: bool,
    hf_hub_dataset_name: str,
    # offline mode args
    offline: bool = False,
    use_dataset_completions: bool = False,
    completions_path: str = "",
    use_answer_as_completion: bool = False,
):
    try:
        endpoints_path_obj = Path(endpoints_path)
        if endpoints_path_obj.is_dir():
            endpoints_file = endpoints_path_obj / "endpoints.py"
        else:
            endpoints_file = endpoints_path_obj

        if endpoints_file.exists():
            spec = importlib.util.spec_from_file_location("endpoints", endpoints_file)
            assert spec and spec.loader
            endpoints_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(endpoints_module)
            ENDPOINTS = endpoints_module.ENDPOINTS
        else:
            raise ImportError(f"endpoints.py not found at {endpoints_file}")
    except (ImportError, AttributeError):
        print(
            f"No local endpoint registry found at {endpoints_path}. \nPlease specify the model name (-m), API host base URL (-b), and API key variable name (-k)."
        )
        ENDPOINTS = {}

    if model in ENDPOINTS:
        api_key_var = ENDPOINTS[model]["key"]
        api_base_url = ENDPOINTS[model]["url"]
        model = ENDPOINTS[model]["model"]

    vf_env = vf.load_environment(env_id=env, **env_args)

    # Offline mode: bypass inference, use provided/dataset completions
    if offline:
        # Load dataset (eval or train) and apply rollouts_per_example via repeat
        if vf_env.eval_dataset is None:
            inputs = vf_env.get_dataset(n=num_examples)
        else:
            inputs = vf_env.get_eval_dataset(n=num_examples)
        assert inputs is not None, "No dataset found"
        if rollouts_per_example > 1:
            inputs = inputs.repeat(rollouts_per_example)

        # Extract fields similar to Environment.a_generate pre-processing
        results_dict: dict = {}
        for col in inputs.column_names:
            if col == "info":
                results_dict[col] = [dict(item) for item in inputs[col]]
            else:
                results_dict[col] = list(inputs[col])
        if "prompt" not in results_dict:
            raise ValueError("prompt column not found in inputs for offline evaluation")
        if "answer" not in results_dict and "info" not in results_dict:
            # allow empty answers if not provided
            results_dict["answer"] = [""] * len(results_dict["prompt"])
        if "answer" not in results_dict:
            results_dict["answer"] = [""] * len(results_dict["prompt"])
        if "task" not in results_dict:
            results_dict["task"] = ["default"] * len(results_dict["prompt"])
        if "info" not in results_dict:
            results_dict["info"] = [{}] * len(results_dict["prompt"])

        # Determine completions source
        completions: list = []
        if use_answer_as_completion:
            completions = list(results_dict["answer"])  # type: ignore
        elif use_dataset_completions and "completion" in results_dict:
            completions = list(results_dict["completion"])  # type: ignore
        elif completions_path:
            completions = _load_completions_from_path(completions_path)
        elif "completion" in results_dict:
            # if dataset already has completions, default to using them
            completions = list(results_dict["completion"])  # type: ignore
        else:
            raise ValueError(
                "Offline mode requires completions. Provide --use-dataset-completions if dataset contains a 'completion' column, --completions-path to a JSON/JSONL with completions, or --use-answer-as-completion."
            )

        if len(completions) != len(results_dict["prompt"]):
            raise ValueError(
                f"Number of completions ({len(completions)}) does not match number of prompts ({len(results_dict['prompt'])})."
            )

        # Sanitize completions for tool calls if necessary during save
        # Build states as empty dicts
        states = [{} for _ in completions]

        # Score offline rollouts
        rollout_scores = vf_env.rubric.score_rollouts(
            prompts=results_dict["prompt"],
            completions=completions,
            answers=results_dict["answer"],
            states=states,
            tasks=results_dict["task"],
            infos=results_dict["info"],
        )
        # score_rollouts may be async or return a coroutine depending on rubric; handle both
        if hasattr(rollout_scores, "__await__"):
            # run in a minimal event loop via Environment.generate helper
            # Reuse Environment.a_generate run loop pattern by creating a temporary GenerateOutputs via coroutine
            from asyncio import get_running_loop, new_event_loop, set_event_loop
            from concurrent.futures import ThreadPoolExecutor

            try:
                loop = get_running_loop()
                import nest_asyncio  # type: ignore

                nest_asyncio.apply()
                rollout_scores = loop.run_until_complete(rollout_scores)  # type: ignore
            except RuntimeError:
                executor = ThreadPoolExecutor(max_workers=vf_env.max_workers)
                loop = new_event_loop()
                try:
                    loop.set_default_executor(executor)
                    set_event_loop(loop)
                    rollout_scores = loop.run_until_complete(rollout_scores)  # type: ignore
                finally:
                    loop.close()
                    set_event_loop(None)
                    executor.shutdown(wait=False)

        results = GenerateOutputs(
            prompt=list(results_dict["prompt"]),
            completion=list(completions),
            answer=list(results_dict["answer"]),
            state=states,
            info=list(results_dict["info"]),
            task=list(results_dict["task"]),
            reward=list(rollout_scores.reward),  # type: ignore
            metrics=dict(rollout_scores.metrics),  # type: ignore
        )
    else:
        client = OpenAI(api_key=os.getenv(api_key_var, "EMPTY"), base_url=api_base_url)
        sampling_args: dict[str, int | float | None] = {
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            sampling_args["temperature"] = temperature
        results = vf_env.evaluate(
            client=client,
            model=model,
            sampling_args=sampling_args,
            num_examples=num_examples,
            rollouts_per_example=rollouts_per_example,
            max_concurrent_requests=max_concurrent_requests,
        )

    print("--- Evaluation ---")
    print(f"Environment: {env}")
    print(f"Model: {model}")
    print(f"Provider: {api_base_url}")
    print(f"Examples: {num_examples}")
    print(f"Rollouts per example: {rollouts_per_example}")

    print("--- Example ---")
    vf.print_prompt_completions_sample(
        results.prompt, results.completion, results.reward, step=0
    )
    print("--- All ---")
    print("Rewards:")
    print(
        f"reward: avg - {sum(results.reward) / len(results.reward):.3f}, std - {np.std(results.reward):.3f}"
    )
    if verbose:
        for i in range(len(results.prompt)):
            print(f"Prompt: {results.prompt[i]}")
            print(f"Completion: {results.completion[i]}")
            print(f"Reward: {results.reward[i]}")
            print(f"Answer: {results.answer[i]}")
            print(f"Info: {results.info[i]}")
            print(f"Task: {results.task[i]}")
    n = num_examples
    r = rollouts_per_example
    if n < 0:
        n = len(results.reward) // r
    if r <= 0:
        r = 1
    for i in range(r):
        # rounded to 3 decimal places
        trials = [round(results.reward[(i * n) + j], 3) for j in range(n)]
        out = f"r{i + 1}: {trials}"
        print(out)
    for k in results.metrics:
        v = results.metrics[k]
        print(f"{k}: avg - {sum(v) / len(v):.3f}, std - {np.std(v):.3f}")
        for i in range(r):
            # rounded to 3 decimal places
            trials = [round(v[(i * n) + j], 3) for j in range(n)]
            out = f"r{i + 1}: {trials}"
            print(out)

    if save_dataset or save_to_hf_hub:
        # Determine effective number of examples
        effective_n = len(results.reward) // rollouts_per_example if rollouts_per_example > 0 else len(results.reward)
        ids = [i // rollouts_per_example for i in range(effective_n * rollouts_per_example)] if rollouts_per_example > 0 else list(range(len(results.reward)))
        prompts = results.prompt
        completions = []
        for c in results.completion:
            sanitized_c = sanitize_tool_calls(c)
            completions.append(sanitized_c)
        rewards = results.reward
        tasks = results.task
        data_dict = {
            "id": ids,
            "prompt": prompts,
            "completion": completions,
            "task": tasks,
        }
        if results.info[0] != {}:
            data_dict["info"] = results.info
        if results.answer[0] != "":
            data_dict["answer"] = results.answer
        data_dict["reward"] = rewards
        for k in results.metrics:
            v = results.metrics[k]
            data_dict[k] = v

        dataset = Dataset.from_dict(data_dict)
        metadata = {
            "env": env,
            "model": model,
            "num_examples": effective_n,
            "rollouts_per_example": rollouts_per_example,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "avg_reward": sum(results.reward) / len(results.reward),
        }
        for k in results.metrics:
            metadata[f"avg_{k}"] = sum(results.metrics[k]) / len(results.metrics[k])

        uuid_str = str(uuid.uuid4())[:8]
        env_model_str = f"{env}--{model.replace('/', '--')}"
        if save_dataset:
            module_name = env.replace("-", "_")
            local_env_dir = Path(env_dir_path) / module_name
            if local_env_dir.exists():
                results_path = (
                    local_env_dir / "outputs" / "evals" / env_model_str / uuid_str
                )
            else:
                results_path = Path("./outputs") / "evals" / env_model_str / uuid_str
            results_path.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_json(results_path / "results.jsonl")
            with open(results_path / "metadata.json", "w") as f:
                json.dump(metadata, f)

            print(f"Saved dataset to {results_path}")
        if save_to_hf_hub:
            if hf_hub_dataset_name == "":
                dataset_name = (
                    f"{env}_{model}_n={effective_n}_r={rollouts_per_example}"
                )
            else:
                dataset_name = hf_hub_dataset_name
            dataset.push_to_hub(dataset_name)
            print(f"Saved dataset to Hugging Face Hub: {dataset_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env", type=str, default="gsm8k", help="Environment module name"
    )
    parser.add_argument(
        "--env-args",
        "-a",
        type=json.loads,
        default={},
        help="Environment module arguments as JSON object (e.g., '{\"key\": \"value\", \"num\": 42}')",
    )
    parser.add_argument(
        "--env-dir-path",
        "-p",
        type=str,
        default="./environments",
        help="Path to environments directory",
    )
    parser.add_argument(
        "--endpoints-path",
        "-e",
        type=str,
        default="./configs/endpoints.py",
        help="Path to API endpoints registry",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt-4.1-mini",
        help="Name of model to evaluate",
    )
    parser.add_argument(
        "--api-key-var",
        "-k",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name for API key",
    )
    parser.add_argument(
        "--api-base-url",
        "-b",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for API",
    )
    parser.add_argument(
        "--num-examples",
        "-n",
        type=int,
        default=5,
        help="Number of examples to evaluate",
    )
    parser.add_argument(
        "--rollouts-per-example",
        "-r",
        type=int,
        default=3,
        help="Number of rollouts per example",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        "-c",
        type=int,
        default=32,
        help="Maximum number of concurrent requests",
    )
    parser.add_argument(
        "--max-tokens",
        "-t",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature", "-T", type=float, default=None, help="Temperature for sampling"
    )
    parser.add_argument(
        "--verbose", "-v", default=False, action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--save-dataset",
        "-s",
        default=False,
        action="store_true",
        help="Save dataset to disk",
    )
    parser.add_argument(
        "--save-to-hf-hub",
        "-H",
        default=False,
        action="store_true",
        help="Save dataset to Hugging Face Hub",
    )
    parser.add_argument(
        "--hf-hub-dataset-name",
        "-D",
        type=str,
        default="",
        help="Name of dataset to save to Hugging Face Hub",
    )
    # Offline options
    parser.add_argument(
        "--offline",
        "-O",
        default=False,
        action="store_true",
        help=(
            "Offline evaluation: skip inference and score provided completions or answers. "
            "Use --use-dataset-completions if your dataset has a 'completion' column, "
            "--completions-path to provide an external list, or --use-answer-as-completion."
        ),
    )
    parser.add_argument(
        "--use-dataset-completions",
        "-C",
        default=False,
        action="store_true",
        help="Use the 'completion' column from the environment dataset (if present)",
    )
    parser.add_argument(
        "--completions-path",
        "-f",
        type=str,
        default="",
        help="Path to a .jsonl/.json file containing completions",
    )
    parser.add_argument(
        "--use-answer-as-completion",
        "-A",
        default=False,
        action="store_true",
        help="Use dataset 'answer' as completion (idealized ground truth)",
    )
    args = parser.parse_args()

    eval_environment(
        env=args.env,
        env_args=args.env_args,
        env_dir_path=args.env_dir_path,
        endpoints_path=args.endpoints_path,
        model=args.model,
        api_key_var=args.api_key_var,
        api_base_url=args.api_base_url,
        num_examples=args.num_examples,
        rollouts_per_example=args.rollouts_per_example,
        max_concurrent_requests=args.max_concurrent_requests,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        verbose=args.verbose,
        save_dataset=args.save_dataset,
        save_to_hf_hub=args.save_to_hf_hub,
        hf_hub_dataset_name=args.hf_hub_dataset_name,
        offline=args.offline,
        use_dataset_completions=args.use_dataset_completions,
        completions_path=args.completions_path,
        use_answer_as_completion=args.use_answer_as_completion,
    )


if __name__ == "__main__":
    main()
