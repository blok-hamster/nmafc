#!/usr/bin/env bash
# Build the turn index into each per-conversation LoCoMo store.
#
# Sequential on purpose. Each store is opened, embedded and closed; running ten
# at once would multiply the open LanceDB handles for no gain, because the cost
# here is one embedding call per few hundred turns and the provider is nowhere
# near saturated by one store at a time.
#
# `--store` names the conversation, not a directory: the builder assembles
# `<run>/stores/<arm>__<store>` itself. Passing it a path silently produces a
# path that does not exist and sqlite reports it as "unable to open database".
set -u
run="${1:-C:/nmafc_ab/locomo_real}"
for conv in 26 30 41 42 43 44 47 48 49 50; do
  echo "=== conv-$conv"
  PYTHONIOENCODING=utf-8 python -u scripts/benchmarks/_build_turn_index.py \
    --run "$run" --store "conv-$conv" 2>&1 | grep -v "ERROR lance\|auto_cleanup"
done
echo "=== index build finished"
