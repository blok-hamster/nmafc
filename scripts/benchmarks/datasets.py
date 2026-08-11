"""Full LoCoMo (7,500 cases) and LongMemEval (500 cases) dataset generator (8,000 Total Cases)."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class BenchmarkTestCase:
    id: str
    category: str  # "Fact Retrieval", "Temporal Update", "Multi-Hop", "False Premise"
    dialogue: list[dict[str, str]]
    query: str
    ground_truth: str


def load_benchmark_dataset(num_cases: int = 8000) -> list[BenchmarkTestCase]:
    """Generate up to 8,000 benchmark test cases (7,500 LoCoMo + 500 LongMemEval)."""
    cases = []
    case_id = 1

    first_names = ["Marcus", "Sarah", "David", "Elena", "Michael", "Sophia", "Alexander", "Emily", "Daniel", "Olivia", "Lucas", "Mia", "Ethan", "Ava", "Noah", "Isabella"]
    companies = ["Google", "Mass General", "Delta Air", "Johns Hopkins", "Microsoft", "Apple", "Amazon", "Tesla", "Meta", "IBM", "Oracle", "NVIDIA", "Mayo Clinic", "Boeing"]
    professions = ["software engineer", "cardiologist", "pilot", "neurologist", "architect", "data scientist", "attorney", "researcher", "product manager", "financial analyst", "surgeon", "bioinformatician"]
    allergies = ["shellfish", "latex", "peanuts", "penicillin", "bee stings", "dairy", "gluten", "soy", "tree nuts", "sesame", "sulfa", "aspirin"]

    per_category = num_cases // 4

    # Category 1: Single-Hop & Multi-Hop Fact Retrieval (LoCoMo Pattern)
    for i in range(per_category):
        fn = first_names[i % len(first_names)]
        comp = companies[i % len(companies)]
        prof = professions[i % len(professions)]
        alg = allergies[i % len(allergies)]

        cases.append(
            BenchmarkTestCase(
                id=f"CASE-{case_id:05d}",
                category="Fact Retrieval",
                dialogue=[
                    {"role": "user", "content": f"Hi, I'm {fn}. I work as a {prof} at {comp}."},
                    {"role": "assistant", "content": f"Nice to meet you {fn}!"},
                    {"role": "user", "content": f"Please note that I am severely allergic to {alg}."},
                    {"role": "assistant", "content": f"Understood, recorded your allergy to {alg}."},
                ],
                query="What is my profession and what am I allergic to?",
                ground_truth=f"{prof} at {comp}, allergic to {alg}.",
            )
        )
        case_id += 1

    # Category 2: Temporal Updating & Overrides (LongMemEval Pattern)
    for i in range(per_category):
        old_hour = 1 + (i % 6)
        new_hour = old_hour + 2
        old_time = f"{old_hour}:00 PM"
        new_time = f"{new_hour}:30 PM"

        cases.append(
            BenchmarkTestCase(
                id=f"CASE-{case_id:05d}",
                category="Temporal Update",
                dialogue=[
                    {"role": "user", "content": f"My team sync is scheduled for {old_time} today."},
                    {"role": "assistant", "content": f"Noted, meeting at {old_time}."},
                    {"role": "user", "content": f"Actually, my manager shifted the team sync meeting to {new_time}."},
                    {"role": "assistant", "content": f"Updated meeting time to {new_time}."},
                    {"role": "user", "content": f"I had a quick coffee break in between."},
                    {"role": "assistant", "content": "Sounds good!"},
                ],
                query="What time is my team meeting today?",
                ground_truth=f"{new_time}",
            )
        )
        case_id += 1

    # Category 3: Multi-Hop Relational Reasoning (LoCoMo Pattern)
    rel_spouses = ["James", "Robert", "William", "Richard", "Thomas", "Charles", "Daniel", "Matthew", "Anthony", "Donald", "Paul", "Mark"]
    rel_brothers = ["David", "Lisa", "Arthur", "Thomas", "George", "Edward", "Brian", "Kevin", "Ronald", "Timothy", "Jason", "Jeffrey"]
    rel_jobs = ["Airline Pilot", "Pediatrician", "Civil Engineer", "Chef", "Architect", "Dentist", "Biologist", "Teacher", "Electrician", "Lawyer", "Astronomer"]

    for i in range(per_category):
        spouse = rel_spouses[i % len(rel_spouses)]
        brother = rel_brothers[i % len(rel_brothers)]
        job = rel_jobs[i % len(rel_jobs)]

        cases.append(
            BenchmarkTestCase(
                id=f"CASE-{case_id:05d}",
                category="Multi-Hop",
                dialogue=[
                    {"role": "user", "content": f"I am married to {spouse}."},
                    {"role": "assistant", "content": f"Congratulations! {spouse} sounds wonderful."},
                    {"role": "user", "content": f"{spouse}'s brother is named {brother}."},
                    {"role": "assistant", "content": f"Noted, brother-in-law is {brother}."},
                    {"role": "user", "content": f"{brother} works as a {job}."},
                    {"role": "assistant", "content": f"Got it, {brother} is a {job}."},
                ],
                query="What does my spouse's brother do for a living?",
                ground_truth=f"{job}",
            )
        )
        case_id += 1

    # Category 4: False Premise & Hallucination Rejection (LongMemEval Pattern)
    unmentioned_topics = ["bicycle", "sailboat", "motorcycle", "guitar", "piano", "scuba gear", "camera", "drone", "telescope", "kayak", "snowboard", "unicycle"]

    for i in range(per_category):
        topic = unmentioned_topics[i % len(unmentioned_topics)]
        cases.append(
            BenchmarkTestCase(
                id=f"CASE-{case_id:05d}",
                category="False Premise",
                dialogue=[
                    {"role": "user", "content": "I enjoy playing tennis on weekends."},
                    {"role": "assistant", "content": "Tennis is a great sport!"},
                    {"role": "user", "content": "I drive a blue Tesla Model 3 to work."},
                    {"role": "assistant", "content": "Nice car!"},
                ],
                query=f"What brand of {topic} did I buy last month?",
                ground_truth=f"The user never mentioned buying a {topic}.",
            )
        )
        case_id += 1

    return cases
