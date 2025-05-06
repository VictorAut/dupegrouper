- ignore_na param: are nulls currently deduped? 
    - need to add to tests

- mask: i.e. apply only to subset
    - e.g. deduplication technique only for strings of len > 1: This is possible via `Custom`

- Jaccard similarity for sets of numeric columns:
    - include validation
    - retrospective validation for "string" dedupers
    - need to think about how this is an unparametrised method that by defintion needs several columns to work.