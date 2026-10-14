-- FedRAG public demo: a Postgres role that can only read schema fedrag, for the Hugging Face
-- Space's DATABASE_URL. The demo (PUBLIC_DEMO=true) only ever runs SELECTs: meetings, stance
-- results, the federal funds target range, statements and stored comparison analyses.
--
-- Run it once in the Neon SQL editor (Console > your project > SQL Editor), on the database that
-- holds schema fedrag, as the role that loads FedRAG (the one in your local backend/.env
-- DATABASE_URL, usually neondb_owner): the default privileges below cover the tables that role
-- creates later. Do not commit the password.
--
-- 1. Replace <choose-a-strong-password> with a long random password, e.g. the output of
--    `openssl rand -hex 24` (hex needs no percent-encoding in a URL; Neon rejects weak ones).
-- 2. Run the whole script, then read the checks at the end.
-- 3. Build the Space's secret from Neon's pooled connection string, with this role and password:
--    postgresql://fedrag_demo_reader:<password>@<endpoint>-pooler.<region>.aws.neon.tech/neondb?sslmode=require

CREATE ROLE fedrag_demo_reader WITH LOGIN PASSWORD '<choose-a-strong-password>'
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS;
-- The API's pool opens at most 4 connections (backend/app/db/connection.py); this leaves headroom.
ALTER ROLE fedrag_demo_reader CONNECTION LIMIT 10;

-- Every session starts read-only, and no query runs for long. These are session defaults, which
-- the role itself could override (SET ...), not caps: the privileges below are the write barrier,
-- and the password is the real boundary, so keep the URL secret.
ALTER ROLE fedrag_demo_reader SET default_transaction_read_only = on;
ALTER ROLE fedrag_demo_reader SET statement_timeout = '15s';
ALTER ROLE fedrag_demo_reader SET idle_in_transaction_session_timeout = '60s';

DO $$
BEGIN
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO fedrag_demo_reader', current_database());
END
$$;
GRANT USAGE ON SCHEMA fedrag TO fedrag_demo_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA fedrag TO fedrag_demo_reader;
-- Tables this role (the one running the script) creates in fedrag later, such as
-- fedrag.statement_comparisons on a fresh database, are readable too.
ALTER DEFAULT PRIVILEGES IN SCHEMA fedrag GRANT SELECT ON TABLES TO fedrag_demo_reader;

-- Checks -----------------------------------------------------------------------------------------

-- Every fedrag table: can_select true, can_write false.
SELECT c.relname AS table_name,
       has_table_privilege('fedrag_demo_reader', c.oid, 'SELECT') AS can_select,
       has_table_privilege('fedrag_demo_reader', c.oid, 'INSERT, UPDATE, DELETE, TRUNCATE') AS can_write
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'fedrag' AND c.relkind IN ('r', 'p')
ORDER BY c.relname;

-- No CREATE in fedrag or public, read-only sessions, and membership in no other role
-- (member_of should be empty: the role gets no more than this script grants).
SELECT r.rolname,
       has_schema_privilege(r.oid, 'fedrag', 'CREATE') AS can_create_in_fedrag,
       has_schema_privilege(r.oid, 'public', 'CREATE') AS can_create_in_public,
       r.rolconfig AS settings,
       ARRAY(
           SELECT g.rolname
           FROM pg_auth_members m
           JOIN pg_roles g ON g.oid = m.roleid
           WHERE m.member = r.oid
       ) AS member_of
FROM pg_roles r
WHERE r.rolname = 'fedrag_demo_reader';

-- No table, view or foreign table outside schema fedrag (e.g. another project's schema, or a
-- table in public) is readable: this returns no rows. A grant to PUBLIC would show up here.
SELECT n.nspname AS schema_name, c.relname AS table_name
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname NOT IN ('pg_catalog', 'information_schema', 'fedrag')
  AND n.nspname NOT LIKE 'pg_toast%'
  AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
  AND has_table_privilege('fedrag_demo_reader', c.oid, 'SELECT');

-- To rotate the password later:  ALTER ROLE fedrag_demo_reader WITH PASSWORD '<new-password>';
-- To remove the role:            DROP OWNED BY fedrag_demo_reader; DROP ROLE fedrag_demo_reader;
