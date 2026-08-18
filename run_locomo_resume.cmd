@echo off
REM Detached resume launcher for the LoCoMo full benchmark.
REM Run via Task Scheduler so the process does not belong to any terminal
REM session and survives an editor / SSH / network disconnect.
REM Credentials are read from nmafc\.env by load_dotenv() - nothing here.

cd /d "C:\Users\edogu\OneDrive\Documents\new nmac\nmafc"

"C:\Users\edogu\AppData\Local\Programs\Python\Python312\python.exe" -u -m scripts.benchmarks.run_locomo ^
  --arms raw,rag,neuromorphic,neuromorphic_tuned ^
  --max-hops 0 ^
  --checkpoint resume ^
  --log-file logs/locomo_full.log ^
  --output scripts/benchmarks/results/locomo_full/ >> logs\locomo_full.stdout 2>&1
