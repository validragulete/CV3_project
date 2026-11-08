from pathlib import Path
import sys
import gzip
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cisco_integration import (parse_time, sampled_csv, load_cisco_timeseries,
                               canonicalize_and_split, event_labels)
from data_pipeline import prepare_data


def build_local_fixture(tmp_path, n=650):
    paths = {}
    for scene in ['0','2','4','9']:
        rng = np.random.default_rng(42+int(scene))
        frame = pd.DataFrame({
            'timestamp': pd.date_range('2018-07-24',periods=n,freq='5s',tz='UTC').astype(str),
            'cpu_load': rng.normal(int(scene),1,n),
            'interface_errors': rng.poisson(5+int(scene),n),
            'device_id': ['router-1']*n,
            'drop_rate': rng.random(n),
            'error_counter': rng.poisson(5,n),
            'tx_bytes': rng.integers(1,10000,n),
        })
        path = tmp_path / f'{scene}.csv.{"zip" if scene in ["2","9"] else "gz"}'
        if path.suffix=='.zip':
            with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
                z.writestr('telemetry.csv',frame.to_csv(index=False))
        else:
            with gzip.open(path,'wt') as f: frame.to_csv(f,index=False)
        paths[scene]=path
    return paths


def test_parse_unix_and_calendar():
    assert parse_time(pd.Series(['1532390400','1532390405'])).notna().all()
    assert parse_time(pd.Series(['2018-07-24T00:00:00Z','2018-07-24T00:00:05Z'])).notna().all()


def test_real_archive_reader_and_preprocessing(tmp_path):
    files=build_local_fixture(tmp_path)
    series,features=load_cisco_timeseries(files,resample_seconds=5,max_features=12)
    assert set(series)=={'0','2','4','9'}
    assert 'cpu_load' in features
    output,splits,audit=canonicalize_and_split(series,output=tmp_path/'canonical.csv',window=16,embargo=8)
    assert output.exists()
    bundle=prepare_data(output,splits=splits,window=16,step=8,batch_size=24)
    for part in ['train','val','test']:
        x,y=next(iter(bundle.loaders[part])); assert x.ndim==4 and x.shape[2]==16
        assert y.ndim==1
        assert bundle.summary[part]['normal']>0
        assert bundle.summary[part]['anomalous']>0
    for s in ['0','2','4','9']:
        assert audit[s]['train']['end']<audit[s]['val']['start']
        assert audit[s]['val']['end']<audit[s]['test']['start']


def test_event_intervals_not_silently_guessed(tmp_path):
    files=build_local_fixture(tmp_path,n=500)
    series,_=load_cisco_timeseries(files)
    with pytest.raises(FileNotFoundError):
        canonicalize_and_split(series,output=tmp_path/'out.csv',mode='event_intervals')
    events=tmp_path/'events.csv'
    pd.DataFrame({'scenario':['2','4'], 'start':['2018-07-24T00:10:00Z']*2,
                  'end':['2018-07-24T00:20:00Z']*2}).to_csv(events,index=False)
    with pytest.raises(ValueError,match='No reviewed event intervals'):
        canonicalize_and_split(series,output=tmp_path/'out.csv',mode='event_intervals',events_csv=events)


def test_sparse_asynchronous_metric_selection(tmp_path):
    scenes = {}
    for scene in ['0','2','4','9']:
        rng = np.random.default_rng(100+int(scene))
        n = 1800
        timestamps = pd.date_range('2018-07-24', periods=600, freq='5s',tz='UTC')
        frame = pd.DataFrame({'time':np.repeat(timestamps.astype(str),3),
                              'bandwidth':100_000_000.0})
        interface = np.arange(n)%3 == 0
        for feature in ['input-data-rate','output-data-rate','input-errors',
                        'output-drops','carrier-transitions','input-packet-rate']:
            values = np.full(n,np.nan)
            values[interface] = rng.uniform(1,100,interface.sum()) + np.arange(interface.sum())*.05
            frame[feature] = values
        file = tmp_path/f'{scene}.csv.gz'
        with gzip.open(file,'wt') as fp:
            frame.to_csv(fp,index=False)
        scenes[scene] = file
    result, cols = load_cisco_timeseries(scenes,resample_seconds=5,max_features=16)
    assert 'bandwidth' not in cols
    assert len(cols)>=4
    assert 'input-data-rate' in cols
    assert result['9']['output-data-rate'].std()>0
