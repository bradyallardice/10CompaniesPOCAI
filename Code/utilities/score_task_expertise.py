import json
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).parent.parent.parent / "config.env", override=False)

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "Data"
INPUT_FILE = DATA_DIR / "task_statements_20.xlsx"
OUTPUT_FILE = DATA_DIR / "task_expertise_scores.csv"
CHECKPOINT_FILE = DATA_DIR / "task_expertise_scores_checkpoint.csv"

BATCH_SIZE = 30
CHECKPOINT_EVERY = 100
MODEL = "gpt-5-mini"

SYSTEM_PROMPT = """Rate each occupational task by the expertise required to perform it. Use a 1–10 integer scale:
1 = Any adult with no training (e.g., "Answer phone calls", "File documents alphabetically")
4 = Some on-the-job or vocational training (e.g., "Operate a forklift", "Record customer orders in a database")
7 = Substantial specialized education or years of experience (e.g., "Diagnose engine faults using diagnostic software", "Prepare corporate tax returns")
10 = Deep expert knowledge, typically advanced degree or years of specialist training (e.g., "Interpret electroencephalogram results", "Derive optimal control algorithms for nonlinear systems")

Return ONLY a JSON array where each element has "task_id" (integer) and "score" (integer 1–10). No other text."""


def load_tasks():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    df = pd.read_excel(INPUT_FILE)
    missing = {"Task ID", "Task"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\nAvailable: {list(df.columns)}"
        )
    logger.info(f"Loaded {len(df)} tasks from {INPUT_FILE}")
    return df


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        df = pd.read_csv(CHECKPOINT_FILE)
        completed = set(df["task_id"].tolist())
        logger.info(f"Checkpoint found: {len(completed)} tasks already scored")
        return df, completed
    return pd.DataFrame(columns=["task_id", "expertise_score"]), set()


def score_batch(client, batch):
    user_content = json.dumps(
        [{"task_id": int(row["Task ID"]), "task": row["Task"]} for row in batch]
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],

    )
    raw = response.choices[0].message.content.strip()

    try:
        results = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON from model: {e}\nRaw response (first 500 chars): {raw[:500]}"
        )

    if not isinstance(results, list):
        raise ValueError(f"Expected JSON array, got {type(results)}: {raw[:200]}")

    batch_ids = {int(row["Task ID"]) for row in batch}
    for item in results:
        if "task_id" not in item or "score" not in item:
            raise ValueError(f"Missing keys in result item: {item}")
        if not isinstance(item["score"], int) or not (1 <= item["score"] <= 10):
            raise ValueError(
                f"Score out of range for task {item['task_id']}: {item['score']}"
            )
        if item["task_id"] not in batch_ids:
            raise ValueError(f"Unexpected task_id in response: {item['task_id']}")

    return results


def write_final_output(df_tasks, df_scores):
    df_scores = df_scores.rename(columns={"task_id": "Task ID"})
    df_out = df_tasks.merge(df_scores, on="Task ID", how="left")

    n_missing = df_out["expertise_score"].isna().sum()
    if n_missing > 0:
        logger.warning(f"{n_missing} tasks have no expertise score")

    df_out.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Output written to {OUTPUT_FILE} ({len(df_out)} rows)")

    scored = df_out["expertise_score"].dropna()
    logger.info(f"Scored: {len(scored)}/{len(df_out)} tasks")
    logger.info(f"Mean: {scored.mean():.2f}  Std: {scored.std():.2f}")
    logger.info(f"Distribution:\n{scored.value_counts().sort_index().to_string()}")


def main():
    client = OpenAI()

    df_tasks = load_tasks()
    df_results, completed_ids = load_checkpoint()

    tasks_remaining = df_tasks[~df_tasks["Task ID"].isin(completed_ids)].to_dict("records")
    logger.info(
        f"Total: {len(df_tasks)}  Completed: {len(completed_ids)}  Remaining: {len(tasks_remaining)}"
    )

    if not tasks_remaining:
        logger.info("All tasks already scored. Writing final output.")
        write_final_output(df_tasks, df_results)
        return

    all_results = df_results.to_dict("records")
    batch_count = 0

    for i in range(0, len(tasks_remaining), BATCH_SIZE):
        batch = tasks_remaining[i : i + BATCH_SIZE]

        try:
            results = score_batch(client, batch)
            all_results.extend(
                [{"task_id": r["task_id"], "expertise_score": r["score"]} for r in results]
            )
            batch_count += 1

            if batch_count % CHECKPOINT_EVERY == 0:
                pd.DataFrame(all_results).to_csv(CHECKPOINT_FILE, index=False)
                logger.info(
                    f"Checkpoint saved at batch {batch_count} "
                    f"({i + len(batch)}/{len(tasks_remaining)} tasks processed)"
                )

            if batch_count % 10 == 0:
                logger.info(
                    f"Progress: {i + len(batch)}/{len(tasks_remaining)} tasks processed"
                )

        except Exception as e:
            pd.DataFrame(all_results).to_csv(CHECKPOINT_FILE, index=False)
            logger.info(f"Emergency checkpoint saved: {len(all_results)} tasks scored")
            raise

    write_final_output(df_tasks, pd.DataFrame(all_results))

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint file removed after successful run")


if __name__ == "__main__":
    main()
