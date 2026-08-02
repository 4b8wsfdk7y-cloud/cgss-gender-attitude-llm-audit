# Reproducibility and data boundary

This repository supports two different reproduction claims.

## Fully public

The synthetic demo, Python package, tests, benchmark schema, and metric
calculations can be reproduced without private data or an LLM endpoint.

## Reproducible with authorized inputs

The empirical CGSS comparison requires licensed CGSS 2012, 2018, and 2021
microdata. Authorized users can rebuild the matched benchmark using the
documented R scripts. The repository does not redistribute respondent records,
derived profiles, or profile-linked model outputs.

## What the public demo does not establish

- It does not reproduce the paper's numerical findings.
- Its deterministic mock adapter is not a language model.
- Its synthetic records are not representative of the Chinese population.
- A passing CI workflow verifies software behavior, not empirical validity.

Aggregate manuscript tables and figures are public artifacts. Any numerical
claim made from them must remain tied to the frozen empirical configuration
and the interpretation boundaries in the main README and manuscript.
