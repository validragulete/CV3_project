from __future__ import annotations

import gzip
import io
import json
import math
import re
import zipfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

REPO = "https://api.github.com/repos/cisco-ie/telemetry/contents"
REPO_WEB = "https://github.com/cisco-ie/telemetry"
NORMAL_SCENARIOS = {"0", "1"}
EVENT_SCENARIOS = {"2", "3", "4", "5", "6", "7", "8", "9", "10"}
TIME_NAME = re.compile(r"time|date|epoch|collect|recorded|stamp", re.I)
NON_METRIC = re.compile(r"(^id$|_id$|index|interface|router|device|neighbor|address|prefix|asn|port|timestamp|epoch|time|date|sample|row)", re.I)


def list_scenario_files(scenario: str, session=None) -> list[dict]:
    session = session or requests.Session()
    url = f"{REPO}/{quote(str(scenario))}"
    r = session.get(url, timeout=40, headers={"Accept": "application/vnd.github+json"})
    r.raise_for_status()
    items = r.json()
    if not isinstance(items, list):
        raise ValueError(f"GitHub API nu a returnat lista pentru scenariul {scenario}: {str(items)[:220]}")
    return items


def identify_csv(items: list[dict]) -> dict:
    cand = [it for it in items if it.get("type") == "file" and
            re.search(r"\.csv(?:\.gz|\.zip)?$", it.get("name", ""), re.I)]
    cand = [it for it in cand if not re.search(r"header|definition|case|event|label", it["name"], re.I)]
    if not cand:
        raise ValueError("Nu am identificat fișier .csv/.csv.gz/.csv.zip; verifică listing-ul tipărit.")
    return sorted(cand, key=lambda it: (-int(it.get("size", 0)), it["name"]))[0]


def download_scenarios(scenarios=("0", "2", "4", "9"), directory="data/cisco_raw",
                       max_file_mb=650, session=None) -> dict[str, Path]:
    session = session or requests.Session()
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = {}
    manifest = []
    for scenario in map(str, scenarios):
        files = list_scenario_files(scenario, session)
        info = identify_csv(files)
        size = int(info.get("size", 0))
        if size > max_file_mb * 1024 * 1024:
            raise ValueError(f"Scenariul {scenario}: {info['name']} are {size/1048576:.0f} MB, "
                             f"peste MAX_FILE_MB={max_file_mb}. Mărește limita sau selectează un subset.")
        file_out = destination / f"{scenario}_{info['name']}"
        print(f"Cisco {scenario}: {info['name']} ({size / 1048576:.1f} MB)")
        if not file_out.exists() or file_out.stat().st_size != size:
            url = info.get("download_url")
            if not url:
                raise ValueError(f"Fișierul {info['name']} nu are download_url")
            with session.get(url, stream=True, timeout=(25, 150)) as r:
                r.raise_for_status()
                with open(file_out.with_suffix(file_out.suffix + '.tmp'), "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
            temp = file_out.with_suffix(file_out.suffix + '.tmp')
            if size and temp.stat().st_size != size:
                raise IOError(f"Fișier incomplet: {temp.stat().st_size} / {size} bytes")
            temp.replace(file_out)
        downloaded[scenario] = file_out
        manifest.append({"scenario": scenario, "file": file_out.name,
                         "size_bytes": size, "source": info.get("html_url")})
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return downloaded


def open_csv(path: str | Path):
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".zip"):
        z = zipfile.ZipFile(path, "r")
        members = [m for m in z.namelist() if m.lower().endswith(".csv") and not m.endswith('/')]
        if not members:
            z.close()
            raise ValueError(f"No CSV inside {path.name}")
        chosen = max(members, key=lambda m: z.getinfo(m).file_size)
        return z, io.TextIOWrapper(z.open(chosen), encoding='utf-8-sig', errors='replace')
    if name.endswith(".gz"):
        return None, gzip.open(path, 'rt', encoding='utf-8-sig', errors='replace')
    return None, open(path, 'r', encoding='utf-8-sig', errors='replace')


def sampled_csv(path: str | Path, rows_per_chunk=300, chunk_size=15000,
                max_sampled_rows=24000) -> pd.DataFrame:
    archive, fp = open_csv(path)
    retained = []
    chunks = 0
    try:
        reader = pd.read_csv(fp, chunksize=chunk_size, low_memory=False,
                             on_bad_lines="skip", dtype=str)
        for chunk in reader:
            if chunk.empty:
                continue
            chunks += 1
            positions = np.linspace(0, len(chunk) - 1, num=min(rows_per_chunk, len(chunk)), dtype=int)
            retained.append(chunk.iloc[positions].copy())
            if sum(len(x) for x in retained) > 2 * max_sampled_rows:
                retained = [pd.concat(retained, ignore_index=True).iloc[::2].reset_index(drop=True)]
    finally:
        fp.close()
        if archive is not None:
            archive.close()
    if not retained:
        raise ValueError(f"No readable rows in {path}")
    df = pd.concat(retained, ignore_index=True)
    if len(df) > max_sampled_rows:
        df = df.iloc[np.linspace(0, len(df) - 1, max_sampled_rows, dtype=int)].reset_index(drop=True)
    print(f" {Path(path).name}: scanned {chunks} chunks; sampled {len(df)} rows; columns={len(df.columns)}")
    return df


def parse_time(values: pd.Series) -> pd.Series:
    s = values.astype('string').str.strip()
    num = pd.to_numeric(s, errors='coerce')
    mostly_numeric = float(num.notna().mean()) > 0.85
    if mostly_numeric:
        median = num.abs().median()
        if 1e9 <= median < 1e11: unit = 's'
        elif 1e11 <= median < 1e14: unit = 'ms'
        elif 1e14 <= median < 1e17: unit = 'us'
        elif 1e17 <= median < 1e20: unit = 'ns'
        else: return pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns, UTC]')
        result = pd.to_datetime(num, unit=unit, utc=True, errors='coerce')
    else:
        result = pd.to_datetime(s, utc=True, errors='coerce', format='mixed')
    return result


def find_timestamp_column(df: pd.DataFrame, override: str | None = None):
    if override:
        if override not in df:
            raise ValueError(f"TIME_COLUMN={override!r} nu există. Coloane: {list(df.columns)[:30]}")
        ts = parse_time(df[override])
        if ts.notna().sum() < 20:
            raise ValueError(f"TIME_COLUMN={override!r} does not parse as date/Unix epoch")
        return override, ts
    candidates = [c for c in df if TIME_NAME.search(str(c))]
    candidates += [c for c in list(df)[:8] if c not in candidates]
    best = None
    for c in candidates[:25]:
        ts = parse_time(df[c]); valid = ts.dropna()
        if len(valid) < 20:
            continue
        plausible = valid[(valid.dt.year >= 2000) & (valid.dt.year <= 2040)]
        uniqueness = plausible.nunique()
        ratio = float(len(plausible)) / len(df)
        if uniqueness < 15 or ratio < 0.75:
            continue
        score = (1 if TIME_NAME.search(str(c)) else 0, ratio, uniqueness)
        if best is None or score > best[0]:
            best = (score, c, ts)
    if best is None:
        raise ValueError("Coloana temporală nu poate fi identificată sigur. "
                         "Setează TIME_COLUMN_OVERRIDES în notebook. "
                         f"Primele coloane sunt: {list(df.columns)[:30]}")
    return best[1], best[2]


FEATURE_PRIORITY = (
    'input-data-rate', 'output-data-rate',
    'input-packet-rate', 'output-packet-rate',
    'input-load', 'output-load',
    'input-drops', 'output-drops', 'input-errors', 'crc-errors',
    'carrier-transitions', 'packets-received', 'packets-sent',
    'bytes-received', 'bytes-sent',
    'reliability', 'total-cpu-one-minute', 'total-cpu-five-minute',
    'free-physical-memory', 'bandwidth',
)


def sanitize_numeric(df: pd.DataFrame, timestamp_col: str, min_row_coverage=0.005) -> pd.DataFrame:
    numeric = {}
    for c in df.columns:
        if c == timestamp_col or NON_METRIC.search(str(c)):
            continue
        v = pd.to_numeric(df[c].astype('string').str.strip(), errors='coerce')
        if v.notna().mean() >= min_row_coverage and v.nunique(dropna=True) >= 2:
            numeric[c] = v.astype(float)
    if not numeric:
        raise ValueError(
            'Nicio metrica numerica variabila in CSV. Verifica coloanele brute; '
            'NU folosi automat bandwidth constant ca feature de model.'
        )
    return pd.DataFrame(numeric, index=df.index)


def load_cisco_timeseries(files: dict[str, Path], resample_seconds=5,
                          override_timestamp: dict | None=None, max_features=16):
    override_timestamp = override_timestamp or {}
    provisional = {}
    for scenario, path in files.items():
        raw = sampled_csv(path)
        timestamp_col, ts = find_timestamp_column(raw, override_timestamp.get(str(scenario)))
        num = sanitize_numeric(raw, timestamp_col)
        num.insert(0, '__timestamp', ts)
        num = num.dropna(subset=['__timestamp'])
        num = num[(num['__timestamp'].dt.year >= 2000) & (num['__timestamp'].dt.year <= 2040)]
        if num.empty:
            raise ValueError(f'Scenariul {scenario}: nu au ramas timestamps valide')
        num['__timestamp'] = num['__timestamp'].dt.floor(f'{int(resample_seconds)}s')
        agg = num.groupby('__timestamp', as_index=False).median(numeric_only=True)
        provisional[str(scenario)] = agg
        print(f' Scenario {scenario}: {len(raw)} randuri esantionate, '
              f'{len(num.columns)-1} metrici numerice candidate, {len(agg)} intervale temporale')

    shared = set.intersection(*(set(x.columns) - {'__timestamp'} for x in provisional.values()))
    if not shared:
        raise ValueError('Nicio metrica comuna: este necesar mapping explicit intre scenarii.')

    diagnostic = []
    selected = []
    for col in shared:
        coverages = [float(frame[col].notna().mean()) for frame in provisional.values()]
        unique_per_scenario = [int(frame[col].nunique(dropna=True)) for frame in provisional.values()]
        variating = sum(v > 1 for v in unique_per_scenario)
        good = min(coverages) >= 0.02 and variating >= 2
        diagnostic.append({'feature':col, 'min_time_coverage':round(min(coverages),3),
                           'varying_scenarios':variating, 'selected_candidate':good})
        if good: selected.append(col)

    def rank(col):
        if col in FEATURE_PRIORITY:
            return (0, FEATURE_PRIORITY.index(col), col)
        return (1, -min(provisional[s][col].notna().mean() for s in provisional), col)

    selected = sorted(selected,key=rank)[:max_features]
    print('Feature quality after resampling (shared schema):')
    for item in sorted(diagnostic,key=lambda e:rank(e['feature'])):
        if item['selected_candidate']:
            print(f"  OK {item['feature']:<35} min time coverage={item['min_time_coverage']:.1%}; "
                  f"varying scenarios={item['varying_scenarios']}")
    if len(selected) < 4:
        raise ValueError(
            f'Au ramas doar {len(selected)} metrici variabile comune: {selected}. '
            'Nu continua antrenarea; inspecteaza setul original/sampling/resampling.'
        )
    print(f'Selected {len(selected)} real features:', selected)
    out = {scene:frame[['__timestamp']+selected].copy()
           for scene,frame in provisional.items()}
    for scene,frame in out.items():
        vary = {c:int(frame[c].nunique(dropna=True)) for c in selected}
        print(f'Scenario {scene} feature variability: ', vary)
    return out,selected


def event_labels(times: pd.Series, scenario: str, events: pd.DataFrame):
    matches = events.loc[events.scenario.astype(str) == str(scenario)]
    labels = np.zeros(len(times), dtype=np.int8)
    for _,ev in matches.iterrows():
        start, end = pd.to_datetime(ev['start'], utc=True), pd.to_datetime(ev['end'], utc=True)
        if end < start: raise ValueError('Invalid event interval end before start')
        labels |= ((times >= start) & (times <= end)).to_numpy().astype(np.int8)
    return labels


def canonicalize_and_split(series: dict[str, pd.DataFrame],
                           output='data/cisco_labeled.csv', mode='scenario_proxy',
                           events_csv=None, window=16, embargo=8,
                           split_ratios=(0.70, 0.15, 0.15)):
    if mode not in ('scenario_proxy','event_intervals'):
        raise ValueError("mode must be scenario_proxy or event_intervals")
    if not np.isclose(sum(split_ratios), 1):
        raise ValueError('split_ratios must sum to 1')
    events = None
    if mode == 'event_intervals':
        if events_csv is None or not Path(events_csv).exists():
            raise FileNotFoundError('events_csv with scenario,start,end intervals required')
        events = pd.read_csv(events_csv)
        if not {'scenario','start','end'}.issubset(events):
            raise ValueError('events CSV must have scenario,start,end columns')
        annotated = set(events.scenario.astype(str))
        missing = {s for s in series if s not in NORMAL_SCENARIOS} - annotated
        if missing:
            raise ValueError(f'No reviewed event intervals for anomaly scenarios: {sorted(missing)}. '
                             'Do not treat absent annotations as normal ground truth.')
    parts = []
    audit = {}
    groups = {'train': [], 'val': [], 'test': []}
    for scenario,frame in series.items():
        frame = frame.sort_values('__timestamp').drop_duplicates('__timestamp').reset_index(drop=True)
        n = len(frame)
        cut1, cut2 = math.floor(n * split_ratios[0]), math.floor(n * sum(split_ratios[:2]))
        ranges = {
            'train': (0, cut1 - embargo),
            'val': (cut1 + embargo, cut2 - embargo),
            'test': (cut2 + embargo, n)
        }
        if any(b-a < window for a,b in ranges.values()):
            raise ValueError(f'Scenario {scenario}: {n} time bins => too few for '
                             f'window={window}, embargo={embargo}, ratios={split_ratios}. '
                             'Lower resample_seconds/window or add more data.')
        if mode == 'scenario_proxy':
            if scenario not in NORMAL_SCENARIOS | EVENT_SCENARIOS:
                raise ValueError(f'Unknown scenario: {scenario}')
            scene_labels = np.full(n, 0 if scenario in NORMAL_SCENARIOS else 1, dtype=np.int8)
        else:
            scene_labels = event_labels(frame['__timestamp'], scenario, events)
        audit[scenario] = {}
        for name,(a,b) in ranges.items():
            x = frame.iloc[a:b].copy()
            scene_key = f'{scenario}__{name}'
            x.insert(0, 'scenario', scene_key)
            x.insert(1, 'timestamp', x.pop('__timestamp').astype(str))
            x.insert(2, 'label', scene_labels[a:b])
            parts.append(x)
            groups[name].append(scene_key)
            audit[scenario][name] = {'rows':len(x),'normal':int((scene_labels[a:b] == 0).sum()),
                                     'anomalous':int((scene_labels[a:b] == 1).sum()),
                                     'start':str(frame.iloc[a]['__timestamp']),
                                     'end':str(frame.iloc[b-1]['__timestamp'])}
    df = pd.concat(parts,ignore_index=True)
    out = Path(output); out.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out,index=False)
    (out.parent/'split_audit.json').write_text(json.dumps({'label_mode':mode,'audit':audit,'groups':groups},indent=2),encoding='utf-8')
    print(f'Canonical CSV {out} rows={len(df)} features={len(df.columns)-3}, label_mode={mode}')
    return out, groups, audit


def download_case_images(scenarios=("0", "2", "4", "9"), directory="data/cisco_case_files",
                         session=None, max_size_mb=20):
    session = session or requests.Session()
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    results = {}
    for scenario in map(str, scenarios):
        items = list_scenario_files(scenario, session=session)
        candidates = [item for item in items if re.search(r'\.(?:png|jpe?g)$', item.get('name',''),re.I)
                      and re.search(r'case|event|key|anomal', item.get('name',''),re.I)]
        for item in candidates:
            if int(item.get('size',0)) > max_size_mb*1024*1024:
                continue
            url = item.get('download_url')
            if not url:
                continue
            path = destination / f'{scenario}_{item["name"]}'
            if not path.exists() or path.stat().st_size != int(item.get('size',0)):
                r = session.get(url,timeout=75)
                r.raise_for_status()
                path.write_bytes(r.content)
            results.setdefault(scenario,[]).append(path)
    return results
