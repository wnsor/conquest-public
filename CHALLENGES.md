# Challenges & lessons

A short retrospective on the hardest problems in building Conquest and how they were resolved.

- **Overfitting vs. a real edge.** The central risk in any backtested strategy. Addressed with strict-Pareto promotion gating (a change ships only if it beats the incumbent on *every* metric — the large majority of candidates were rejected), walk-forward validation on unseen windows, overfit-deflated Sharpe ratios, and a written record of what failed so it isn't retried.

- **Look-ahead bias.** The subtlest failure mode, and it recurred. `ctactical` once showed an alarming drawdown that turned out to be an artifact of a regime signal built on later-revised data. Fixed with true point-in-time computation (release-date stamping, vintage macro data) and by having `ctactical` read a dedicated, look-ahead-free regime feed so its backtest reproduces run-to-run.

- **Optimistic early numbers.** Initial results were flattered by missing costs. A bias-correction pass — point-in-time data, realistic slippage, intra-day drawdown, a proper risk-free Sharpe — brought them down to believable levels.

- **Options never paid off at small scale.** Many long-option overlay variants were tested and rejected; theta decay overwhelms the edge at retail size under a no-selling constraint. The lesson was discipline — defer the layer rather than force it.

- **Silent data staleness.** Hardcoded end-dates once froze live signals for months. Fixed with dynamic dates and a rule that every signal an algorithm reads must have automated refresh and fail loudly if missing — never default silently.

- **Fragile live deployment.** Broker-gateway resets caused init-timeout crashes, and a billing outage once silenced monitoring entirely. Addressed with a watchdog that auto-restarts but defers during the reset window, an external dead-man's-switch that reports independently, and a guard that refuses to deploy the wrong project.

- **Many parallel work-streams.** Concurrent development against shared cloud projects and data feeds risked collisions. Addressed with a lane protocol — each stream owns one project, no shared-feed writes without a recorded heads-up — and read-before-write discipline.

- **Accumulated clutter.** Research left several working names per model and dozens of throwaway cloud projects. Consolidated to one canonical name per model (`surge`, `ctactical`, `cstability`, `cgrowth`, `chybrid`) and archived or removed the rest, taking backups first.

**Throughline:** most of these were beaten by discipline rather than cleverness — point-in-time correctness, rejecting more than we added, documenting failures so they stayed dead, and wrapping safety nets around an inherently fragile live-trading setup.
