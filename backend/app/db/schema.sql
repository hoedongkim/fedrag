-- FedRAG: FOMC meetings, their documents, each statement's policy stance and the stored
-- statement comparisons, in their own Postgres schema "fedrag" (Neon).
--
-- Idempotent: app.db.load.load_processed and app.db.stance.load_stance apply this file before
-- every write, and app.db.compare.save_comparison applies it only while
-- fedrag.statement_comparisons is missing, so it only ever creates what is missing. It never
-- alters existing tables; a column change needs a migration.
--
-- meeting_date is the LAST day of a meeting, normally the day its statement is released (the
-- unscheduled meetings of 2019-10-04 and 2020-03-02 released theirs later). release_date is the
-- day a document became public. Point-in-time queries filter on release_date: minutes come out
-- about three weeks after the meeting. meetings.unscheduled is true for an unscheduled meeting and
-- also for a notation vote that issued a statement (2020-03-23).

CREATE SCHEMA IF NOT EXISTS fedrag;

CREATE TABLE IF NOT EXISTS fedrag.meetings (
    meeting_date         DATE    PRIMARY KEY,
    start_date           DATE    NOT NULL,
    unscheduled          BOOLEAN NOT NULL,
    has_sep              BOOLEAN NOT NULL,
    has_press_conference BOOLEAN NOT NULL,
    CONSTRAINT meetings_start_not_after_end CHECK (start_date <= meeting_date)
);

CREATE TABLE IF NOT EXISTS fedrag.documents (
    doc_id              TEXT        PRIMARY KEY,
    doc_type            TEXT        NOT NULL,
    meeting_date        DATE        NOT NULL REFERENCES fedrag.meetings (meeting_date),
    release_date        DATE        NOT NULL,
    release_date_source TEXT        NOT NULL,
    -- The page's "Last Update" stamp. Later than release_date when the Fed revised the page
    -- after publishing it: text holds the revised version, so point-in-time use can flag it.
    page_updated        DATE,
    title               TEXT        NOT NULL,
    url                 TEXT        NOT NULL,
    text                TEXT        NOT NULL,
    paragraphs          JSONB       NOT NULL,
    word_count          INT         NOT NULL,
    sha256              TEXT        NOT NULL,
    parser_version      INT         NOT NULL,
    loaded_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT documents_one_per_meeting UNIQUE (doc_type, meeting_date),
    CONSTRAINT documents_doc_type_known
        CHECK (doc_type IN ('statement', 'minutes', 'sep', 'press_conference')),
    CONSTRAINT documents_released_after_meeting CHECK (release_date >= meeting_date)
);

CREATE INDEX IF NOT EXISTS documents_release_date_idx ON fedrag.documents (release_date);

-- Policy stance (Phase 5), loaded by app.db.stance.load_stance: one row per classified document
-- and classifier. score runs from -1 (most dovish) to +1 (most hawkish); score and confidence are
-- model-reported or derived, never calibrated probabilities. key_factors and evidence are JSON
-- arrays of strings; evidence sentences are quoted verbatim from the document.
CREATE TABLE IF NOT EXISTS fedrag.stance_results (
    doc_id        TEXT             NOT NULL REFERENCES fedrag.documents (doc_id),
    meeting_date  DATE             NOT NULL REFERENCES fedrag.meetings (meeting_date),
    classifier    TEXT             NOT NULL,
    stance        TEXT             NOT NULL,
    score         DOUBLE PRECISION NOT NULL,
    confidence    DOUBLE PRECISION,
    key_factors   JSONB            NOT NULL,
    evidence      JSONB            NOT NULL,
    model_version TEXT             NOT NULL,
    computed_at   TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_id, classifier),
    CONSTRAINT stance_results_classifier_known
        CHECK (classifier IN ('keyword', 'finbert', 'llm_zero_shot', 'llm_few_shot')),
    CONSTRAINT stance_results_stance_known CHECK (stance IN ('hawkish', 'neutral', 'dovish')),
    CONSTRAINT stance_results_score_range CHECK (score BETWEEN -1 AND 1),
    CONSTRAINT stance_results_confidence_range CHECK (confidence BETWEEN 0 AND 1)
);

CREATE INDEX IF NOT EXISTS stance_results_classifier_date_idx
    ON fedrag.stance_results (classifier, meeting_date);

-- The federal funds target range (FRED DFEDTARL and DFEDTARU, in percent), change points only:
-- each row holds from its date until the next row's date.
CREATE TABLE IF NOT EXISTS fedrag.fed_funds_target (
    date  DATE    PRIMARY KEY,
    lower NUMERIC NOT NULL,
    upper NUMERIC NOT NULL,
    CONSTRAINT fed_funds_target_range CHECK (lower >= 0 AND lower <= upper)
);

-- Statement Comparison (Phase 6), written by app.db.compare.save_comparison: one LLM analysis
-- per cache_key, the SHA-256 of both doc_ids, both statement texts' SHA-256, the diff version,
-- the prompt version and the model (app.compare.types.cache_key), so a revised statement text,
-- or a bumped diff or prompt version, gives a new key (DIFF_VERSION is bumped by hand when
-- sentence splitting or alignment changes; release dates are not in the key). analysis is a
-- ComparisonAnalysis (app.schemas.compare) whose key changes cite the diff's change ids; usage
-- holds the calls, tokens and cost of writing it. Its confidence is model-reported, never a
-- calibrated probability.
CREATE TABLE IF NOT EXISTS fedrag.statement_comparisons (
    cache_key      TEXT        PRIMARY KEY,
    base_doc_id    TEXT        NOT NULL REFERENCES fedrag.documents (doc_id),
    target_doc_id  TEXT        NOT NULL REFERENCES fedrag.documents (doc_id),
    model          TEXT        NOT NULL,
    prompt_version TEXT        NOT NULL,
    analysis       JSONB       NOT NULL,
    usage          JSONB       NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT statement_comparisons_two_documents CHECK (base_doc_id <> target_doc_id),
    CONSTRAINT statement_comparisons_analysis_object CHECK (jsonb_typeof(analysis) = 'object'),
    CONSTRAINT statement_comparisons_usage_object CHECK (jsonb_typeof(usage) = 'object')
);

CREATE INDEX IF NOT EXISTS statement_comparisons_pair_idx
    ON fedrag.statement_comparisons (base_doc_id, target_doc_id);
