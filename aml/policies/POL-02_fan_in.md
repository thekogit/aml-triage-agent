# POL-02 Fan-in and funnel accounts

## Description
Many senders pay into one account in a short time. The account works as a collection point
("funnel account"); funds are often moved on soon after arrival.

## Indicators
- 5 or more incoming payments within 24 hours from 5 or more distinct senders.
- Senders spread across several banks or currencies.
- Most of the incoming money leaves again within 24 hours (pass-through).
- Individual customer receiving business-like volumes.

## Escalation rule
Escalate when the account receives from many unrelated senders AND forwards most of the money
within 24 hours. A fan-in alone, explained by the profile (for example, a shop), may be closed.
