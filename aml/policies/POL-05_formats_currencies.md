# POL-05 Payment formats, currencies and customer risk

## Description
Some payment formats and currency conversions make tracing harder. They are not suspicious alone,
but they raise the weight of other indicators.

## Indicators
- Cash or cryptocurrency (Bitcoin) legs inside a chain of bank transfers.
- Cross-currency payments without a business reason in the customer profile.
- Customer risk rating HIGH, or a recently onboarded account with high volumes.
- Activity far above declared monthly turnover.

## Escalation rule
Treat these as aggravating factors: one of them plus one typology indicator from POL-01 to POL-04
is enough to escalate.
