# FITREPextractor v0.4.0

## Highlights
- Marine last name extraction anchored by the Marine EDIPI line on page 1 for higher accuracy across varied layouts.

## Details
- Added `extract_marine_last_name_by_edipi` heuristic to scan lines above the Marine EDIPI to infer LASTNAME.
- Integrated into the main extraction flow with a safe fallback to existing methods.
- Minor line-ending normalization in the touched block.

## Upgrade Notes
- No breaking changes. This augments name extraction only; other fields are unaffected.

## Commit
- eaa5a66: fitrep_extractor: extract Marine last name by EDIPI anchor
