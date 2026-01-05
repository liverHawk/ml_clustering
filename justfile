all-dataset overwrite="false":
    uv run dataset_clustering.py -d all -n 0
    uv run analyze.py -t all {{ if overwrite == "true" { "-o" } else { "" } }}
    uv run db_score_sub.py -t all {{ if overwrite == "true" { "-o" } else { "" } }}

analyze overwrite="false":
    uv run analyze.py -t all {{ if overwrite == "true" { "-o" } else { "" } }}
    uv run db_score_sub.py -t all {{ if overwrite == "true" { "-o" } else { "" } }}