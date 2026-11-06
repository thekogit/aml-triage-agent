# POL-04 Layering: gather-scatter, scatter-gather, stacks and bipartite flows

## Description
Funds are moved through layers of accounts to separate them from their source. Common shapes:
gather-scatter (many in, then many out), scatter-gather (one source splits, then recombines),
stacked layers of intermediaries, and bipartite flows between two groups of accounts.

## Indicators
- Pass-through: an account forwards 80% to 100% of what it received in the previous 24 hours.
- The same intermediaries appear in many chains.
- Rapid movement across currencies or payment formats.

## Escalation rule
Escalate when pass-through is combined with fan-in or fan-out, or when the same intermediaries
link otherwise unrelated customers.
