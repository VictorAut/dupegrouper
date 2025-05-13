- ignore_na param: are nulls currently deduped? 
    - need to add to tests

- mask: i.e. apply only to subset
    - e.g. deduplication technique only for strings of len > 1: This is possible via `Custom`

- Jaccard similarity for sets of numeric columns:
    - include validation
    - retrospective validation for "string" dedupers
    - need to think about how this is an unparametrised method that by defintion needs several columns to work.

- record selection:
    - last/first

- mapPartitions -> mapPartitionsWithIndex ?

- need to test copying GROUP_ID from ID and then deduping on a a string for id's such as "a001"

- Include validation that id must be populated if it's a spark frame + spark session passed (or gotten?)

- Throw error if .dedupe() is used without positional argument and the strategies used are stored under "default".