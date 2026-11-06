# POL-01 Fan-out and structuring

## Description
One account sends many payments to many different receivers in a short time, often just below
internal review thresholds. Purpose: split a large amount so no single payment draws attention
("smurfing"), or spread funds before layering.

## Indicators
- 5 or more outgoing payments within 24 hours to 5 or more distinct receivers.
- Amounts that are similar to each other or just below a round threshold.
- Receivers with no previous relationship to the sender.
- Outgoing total far above the customer's declared monthly turnover.

## Escalation rule
Escalate when at least two indicators are present and the customer profile does not explain the
activity (for example, a payroll company paying staff on a fixed date explains a fan-out).
