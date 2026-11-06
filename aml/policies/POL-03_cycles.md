# POL-03 Cycles and round-tripping

## Description
Money leaves an account and returns to it through one or more intermediaries. Used to create the
appearance of legitimate turnover or to obscure the origin of funds.

## Indicators
- A payment chain that starts and ends at the same account within 14 days.
- Each hop keeps most of the amount (small fees deducted on the way).
- Intermediaries with little other activity.

## Escalation rule
Escalate every confirmed cycle of 3 or more accounts where each hop keeps at least 80% of the
amount. Two-account back-and-forth payments are escalated only with another indicator.
