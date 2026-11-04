#!/usr/bin/env bash
# Save this bot's paper account(s) + a fresh REPORT.md to the repo, safely, even while
# other bots are pushing too. Usage: scripts/save_results.sh "<commit message>" <our state files...>
set +e
MSG="$1"; shift
BR="${GITHUB_REF_NAME:-$(git rev-parse --abbrev-ref HEAD)}"
git config user.name "paper-trading-bot"
git config user.email "paper-trading-bot@users.noreply.github.com"
for attempt in 1 2 3 4 5; do
  git fetch -q origin "$BR" || { sleep 3; continue; }
  git reset -q "origin/$BR"                       # latest remote; our working files stay as-is
  for f in paper/*_state.json; do                  # take OTHER bots' latest accounts from the remote
    case " $* " in *" $f "*) ;; *) git checkout -q "origin/$BR" -- "$f" 2>/dev/null ;; esac
  done
  python -m swing_bot report --state paper/options_state.json --stocks-state paper/stocks_state.json \
    --premarket-state paper/premarket_state.json --out paper/REPORT.md
  git add "$@" paper/REPORT.md 2>/dev/null
  git diff --cached --quiet && exit 0             # nothing new
  git commit -qm "$MSG"
  git push -q origin "HEAD:$BR" && { echo "saved results"; exit 0; }
  sleep $((RANDOM % 5 + 2))
done
echo "could not save results"
