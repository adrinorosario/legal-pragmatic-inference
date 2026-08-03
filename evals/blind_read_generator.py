import json
import random

CATEGORIES = ["Cap On Liability", "Third Party Beneficiary", "Non-Compete"]
PER_CATEGORY_LIMIT = 10
TOTAL_TARGET = PER_CATEGORY_LIMIT * len(CATEGORIES)  # 30


def load_json(file_path):
    """Load either a standard JSON file (single array) or a JSONL file
    (one JSON object per line, e.g. dpo_pairs_2.jsonl)."""
    if file_path.endswith(".jsonl"):
        data = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed line {line_num}: {e}")
        return data
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)


def generate_blind_reads(data):
    """Sample up to PER_CATEGORY_LIMIT items per category, capped at TOTAL_TARGET total."""
    categories_counts = {c: 0 for c in CATEGORIES}
    blind_reads = []

    # Shuffle indices once instead of repeatedly sampling randint (avoids
    # infinite loops / long stalls once a category or the pool is exhausted)
    indices = list(range(len(data)))
    random.shuffle(indices)

    for idx in indices:
        if len(blind_reads) >= TOTAL_TARGET:
            break

        item = data[idx]
        category = item["cuad_category"]

        if category not in categories_counts:
            continue
        if categories_counts[category] >= PER_CATEGORY_LIMIT:
            continue

        blind_reads.append({
            "category": category,
            "question": item["prompt_anchor"],
            "answer": item["chosen"],
            "rejected": item["rejected"],
        })
        categories_counts[category] += 1

    if len(blind_reads) < TOTAL_TARGET:
        print(f"Warning: only found {len(blind_reads)}/{TOTAL_TARGET} items "
              f"(not enough data to fill every category).")

    random.shuffle(blind_reads)  # randomize question order too
    return blind_reads


def test_blind_read(blind_reads):
    correct, wrong = 0, 0
    results = []  # per-question log for the summary/report

    total = len(blind_reads)
    for i, item in enumerate(blind_reads, start=1):
        question = item["question"]
        answer = item["answer"]      # the "chosen"/correct one
        rejected = item["rejected"]  # the wrong one

        # Randomize which slot (A/B) holds the correct answer each time
        correct_is_a = random.choice([True, False])
        option_a = answer if correct_is_a else rejected
        option_b = rejected if correct_is_a else answer
        correct_letter = "a" if correct_is_a else "b"

        print(f"\n--- Question {i}/{total} [{item['category']}] ---")
        print(f'"{question}"\n')
        print(f"Option A: {option_a}\n")
        print(f"Option B: {option_b}\n")

        user_answer = input("Enter your answer (A or B): ").strip().lower()
        while user_answer not in ("a", "b"):
            user_answer = input("Please enter 'A' or 'B': ").strip().lower()

        is_correct = (user_answer == correct_letter)
        if is_correct:
            correct += 1
            print("✅ Correct!\n")
        else:
            wrong += 1
            print(f"❌ Wrong — correct answer was Option {correct_letter.upper()}.\n")

        results.append({
            "category": item["category"],
            "question": question,
            "your_answer": user_answer.upper(),
            "correct_answer": correct_letter.upper(),
            "is_correct": is_correct,
        })

    return correct, wrong, results


def print_summary(correct, wrong, results):
    total = correct + wrong
    accuracy = (correct / total * 100) if total else 0

    print("\n" + "=" * 40)
    print("           FINAL SCORE")
    print("=" * 40)
    print(f"Correct: {correct}/{total}")
    print(f"Wrong:   {wrong}/{total}")
    print(f"Accuracy: {accuracy:.1f}%")

    # Breakdown by category
    by_cat = {}
    for r in results:
        c = r["category"]
        by_cat.setdefault(c, {"correct": 0, "total": 0})
        by_cat[c]["total"] += 1
        if r["is_correct"]:
            by_cat[c]["correct"] += 1

    print("\nBy category:")
    for cat, stats in by_cat.items():
        acc = (stats["correct"] / stats["total"] * 100) if stats["total"] else 0
        print(f"  {cat}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    print("=" * 40)


def save_all_results(sessions, out_path="blind_read_results.json"):
    """Save every session's results plus an overall summary to one JSON file."""
    total_correct = sum(s["correct"] for s in sessions)
    total_wrong = sum(s["wrong"] for s in sessions)
    total = total_correct + total_wrong
    overall_accuracy = (total_correct / total * 100) if total else 0

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_sessions": len(sessions),
            "overall": {
                "correct": total_correct,
                "wrong": total_wrong,
                "accuracy": round(overall_accuracy, 1),
            },
            "sessions": sessions,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nAll session results saved to {out_path}")


def main(num_runs=3):
    # file_path = input("Path to JSON data file: ").strip()
    data = load_json("../dpo_training_env/datasets/dpo_pairs_2.jsonl")

    sessions = []
    for run_num in range(1, num_runs + 1):
        print("\n" + "#" * 40)
        print(f"   SESSION {run_num}/{num_runs}")
        print("#" * 40)

        blind_reads = generate_blind_reads(data)
        correct, wrong, results = test_blind_read(blind_reads)
        print_summary(correct, wrong, results)

        sessions.append({
            "session": run_num,
            "correct": correct,
            "wrong": wrong,
            "results": results,
        })

    # Overall summary across all sessions
    total_correct = sum(s["correct"] for s in sessions)
    total_wrong = sum(s["wrong"] for s in sessions)
    total = total_correct + total_wrong
    overall_accuracy = (total_correct / total * 100) if total else 0

    print("\n" + "=" * 40)
    print("        OVERALL (ALL SESSIONS)")
    print("=" * 40)
    print(f"Correct: {total_correct}/{total}")
    print(f"Wrong:   {total_wrong}/{total}")
    print(f"Accuracy: {overall_accuracy:.1f}%")
    print("=" * 40)

    save_all_results(sessions)


if __name__ == "__main__":
    main()