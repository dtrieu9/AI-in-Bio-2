import re
import time
from typing import Dict, List, Optional, Sequence
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import random
import py3Dmol

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_BASE = "https://data.rcsb.org/rest/v1/core"
CACHE_DIR = Path(__file__).resolve().parent / "data"
DATASET_CACHE = CACHE_DIR / "protein_dataset.csv"
STRUCTURE_CACHE_DIR = CACHE_DIR / "structures"

class AminoAcidComposition(BaseEstimator, TransformerMixin):
    # Convert sequences to [length + amino-acid frequencies].

    def fit(self, X, y=None):
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:
        feats = np.zeros((len(X), len(AA_ORDER) + 1), dtype=float)
        for i, seq in enumerate(X):
            clean = re.sub(r"[^A-Z]", "", str(seq).upper())
            n = max(len(clean), 1)
            feats[i, 0] = len(clean)
            for j, aa in enumerate(AA_ORDER, start=1):
                feats[i, j] = clean.count(aa) / n
        return feats


def _request_json(
    method: str,
    url: str,
    session: requests.Session,
    *,
    payload: Optional[dict] = None,
    timeout: int = 30,
    max_attempts: int = 3,
) -> dict:
    """Request JSON from RCSB, retrying temporary network/server failures."""

    for attempt in range(1, max_attempts + 1):
        try:
            response = session.request(
                method,
                url,
                json=payload,
                timeout=timeout,
            )

            # Do not waste time retrying a permanent client-side error such as 404.
            if 400 <= response.status_code < 500 and response.status_code != 429:
                response.raise_for_status()

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as error:
            status = error.response.status_code

            # 429 = rate limited; 5xx = server problem.
            # These may resolve shortly, so retry them.
            if status != 429 and status < 500:
                raise

            if attempt == max_attempts:
                raise

            wait_seconds = 2 ** (attempt - 1)
            print(
                f"RCSB returned HTTP {status}. "
                f"Retrying in {wait_seconds} second(s)..."
            )
            time.sleep(wait_seconds)

        except requests.exceptions.RequestException as error:
            # Timeout, connection loss, DNS issue, etc.
            if attempt == max_attempts:
                raise

            wait_seconds = 2 ** (attempt - 1)
            print(
                f"Network error ({error}). "
                f"Retrying in {wait_seconds} second(s)..."
            )
            time.sleep(wait_seconds)


def http_get_json(
    url: str,
    session: requests.Session,
    timeout: int = 30,
) -> dict:
    return _request_json(
        "GET",
        url,
        session,
        timeout=timeout,
    )


def http_post_json(
    url: str,
    payload: dict,
    session: requests.Session,
    timeout: int = 30,
) -> dict:
    return _request_json(
        "POST",
        url,
        session,
        payload=payload,
        timeout=timeout,
    )


def rcsb_search_entries(term: str, rows: int = 200, session: Optional[requests.Session] = None) -> List[str]:
    session = session or requests.Session()
    payload = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": term},
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": rows}},
    }
    data = http_post_json(SEARCH_URL, payload, session=session)
    return [item["identifier"] for item in data.get("result_set", [])]


def fetch_entry(entry_id: str, session: requests.Session) -> dict:
    return http_get_json(f"{DATA_BASE}/entry/{entry_id}", session=session)


def fetch_polymer_entity(entry_id: str, entity_id: str, session: requests.Session) -> dict:
    return http_get_json(f"{DATA_BASE}/polymer_entity/{entry_id}/{entity_id}", session=session)


def recursive_find_first_string(obj, candidate_key_fragments: Sequence[str]) -> Optional[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(fragment.lower() in str(k).lower() for fragment in candidate_key_fragments):
                if isinstance(v, str) and v.strip():
                    return v
                if isinstance(v, list) and v:
                    for item in v:
                        if isinstance(item, str) and item.strip():
                            return item
            if (result := recursive_find_first_string(v, candidate_key_fragments)) is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            if (result := recursive_find_first_string(item, candidate_key_fragments)) is not None:
                return result
    return None


def extract_sequence(entity_json: dict) -> Optional[str]:
    candidates = [
        ("entity_poly", "pdbx_seq_one_letter_code_can"),
        ("entity_poly", "pdbx_seq_one_letter_code"),
        ("entity_poly", "seq_one_letter_code_can"),
        ("entity_poly", "seq_one_letter_code"),
        ("rcsb_polymer_entity", "pdbx_seq_one_letter_code_can"),
        ("rcsb_polymer_entity", "pdbx_seq_one_letter_code"),
    ]
    for outer, inner in candidates:
        if (block := entity_json.get(outer)) and isinstance(block, dict):
            if (value := block.get(inner)) and isinstance(value, str) and value.strip():
                return re.sub(r"\s+", "", value).upper()

    if (seq := recursive_find_first_string(
        entity_json,
        ["pdbx_seq_one_letter_code", "seq_one_letter_code", "canonical_seq", "sequence"],
    )):
        if (cleaned := re.sub(r"[^A-Z]", "", seq.upper())):
            return cleaned
    return None


def is_protein_entity(entity_json: dict) -> bool:
    type_text = recursive_find_first_string(
        entity_json,
        ["rcsb_entity_polymer_type", "entity_polymer_type", "polymer_type", "type"],
    )
    return bool(type_text and "protein" in type_text.lower())


def build_dataset(
    class_terms: Dict[str, str],
    samples_per_class: int = 120,
    min_seq_len: int = 40,
    sleep_s: float = 0.05,
) -> pd.DataFrame:
    session = requests.Session()
    rows = []
    seen_entity_ids = set()

    for label, term in class_terms.items():
        print(f"Searching class '{label}' with term '{term}' ...")
        entry_ids = rcsb_search_entries(term, rows=max(300, samples_per_class * 5), session=session)

        class_count = 0
        for entry_id in entry_ids:
            if class_count >= samples_per_class:
                break

            try:
                entry_json = fetch_entry(entry_id, session=session)
            except requests.HTTPError:
                continue

            polymer_entity_ids = entry_json.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])
            for entity_id in polymer_entity_ids:
                if class_count >= samples_per_class:
                    break

                unique_id = f"{entry_id}_{entity_id}"
                if unique_id in seen_entity_ids:
                    continue

                try:
                    entity_json = fetch_polymer_entity(entry_id, entity_id, session=session)
                except requests.HTTPError:
                    continue

                if not is_protein_entity(entity_json):
                    continue

                seq = extract_sequence(entity_json)
                if not seq or len(seq) < min_seq_len:
                    continue

                seen_entity_ids.add(unique_id)
                rows.append({
                    "entity_id": unique_id,
                    "entry_id": entry_id,
                    "label": label,
                    "sequence": seq,
                    "length": len(seq),
                })
                class_count += 1
                if class_count % 20 == 0:
                    print(f"  collected {class_count}/{samples_per_class} for {label}")
                time.sleep(sleep_s)

        print(f"Collected {class_count} sequences for '{label}'.")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No training data collected. Try different class terms or increase rows.")
    return df

def build_or_load_dataset(
    class_terms: Dict[str, str],
    samples_per_class: int = 120,
    min_seq_len: int = 40,
    sleep_s: float = 0.05,
) -> pd.DataFrame:
    """
    Try RCSB first. On success, refresh the local cache.
    If RCSB is unavailable, use data/protein_dataset.csv instead.
    """

    try:
        print("Retrieving data from RCSB PDB...")
        df = build_dataset(
            class_terms,
            samples_per_class=samples_per_class,
            min_seq_len=min_seq_len,
            sleep_s=sleep_s,
        )

        # Do not replace a known-good cache with an incomplete download.
        counts = df["label"].value_counts()
        missing = [
            label for label in class_terms
            if counts.get(label, 0) < samples_per_class
        ]
        if missing:
            raise RuntimeError(
                "RCSB returned an incomplete dataset for: "
                + ", ".join(missing)
            )

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATASET_CACHE, index=False)
        print(f"Saved refreshed dataset to {DATASET_CACHE}")
        return df

    except (requests.exceptions.RequestException, RuntimeError) as error:
        if DATASET_CACHE.exists():
            print(f"RCSB unavailable or incomplete ({error})")
            print(f"Using cached dataset: {DATASET_CACHE}")
            return pd.read_csv(DATASET_CACHE)

        raise RuntimeError(
            "Could not retrieve data from RCSB, and no local backup dataset "
            f"was found at {DATASET_CACHE}."
        ) from error

def protein_visualization(df, max_retries=10, tried_indices=set(), protein_found=False):
    for _ in range(max_retries):
        if len(tried_indices) == len(df): # Avoid infinite loop if all proteins have been tried
            print("All proteins in the dataset have been tried, but none could be visualized.")
            break

        random_index = random.randint(0, len(df) - 1)
        while random_index in tried_indices: # Ensure a new random protein is selected
            random_index = random.randint(0, len(df) - 1)

        tried_indices.add(random_index)

        selected_protein_entry_id = df.iloc[random_index]['entry_id']
        selected_protein_label = df.iloc[random_index]['label']

        print(f"Attempting to visualize protein: {selected_protein_label} (Entry ID: {selected_protein_entry_id})")

        # Fetch the PDB file content
        pdb_url = f"https://files.rcsb.org/download/{selected_protein_entry_id}.pdb"
        try:
            response = requests.get(pdb_url)
            response.raise_for_status() # Raise an exception for HTTP errors
            pdb_content = response.text

            # Create a 3Dmol viewer
            view = py3Dmol.view(width=800, height=400)
            view.addModel(pdb_content, 'pdb')
            view.setStyle({'cartoon': {'color': 'spectrum'}})
            view.zoomTo()
            view.show()
            protein_found = True
            break # Exit loop if a protein is successfully visualized
        except requests.exceptions.RequestException as e:
            print(f"Error fetching PDB file for {selected_protein_entry_id}: {e}")
            print("Retrying with a different protein...")

    if not protein_found and len(tried_indices) < len(df):
        print(f"Could not visualize a protein after {max_retries} attempts.")


def train_and_evaluate(df: pd.DataFrame):
    X = df["sequence"].tolist()
    y = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ("features", AminoAcidComposition()),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, preds), 4))
    print("\nClassification report:\n")
    print(classification_report(y_test, preds))
    return model
