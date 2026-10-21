"""Run the selected PINN experiment and save its outputs in local results/.

Examples:
    python run_experiments.py E1
    python run_experiments.py E2 --inner-steps 5 --seed 5
    python run_experiments.py E1 --smoke --seed 5
    python run_experiments.py --validate

Reuses the local pinn_helpers.py. No training starts without an explicit
experiment selection; --validate executes only reduced smoke checks.
"""

# CONFIG
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import importlib.metadata

ROOT = Path(__file__).resolve().parent
PROBLEM = 'burgers'
SEEDS = (5, 6, 7, 8, 9)
ACTIVATIONS = ('tanh', 'relu', 'pau', 'smpaf')


@dataclass(frozen=True)
class Config:
    problem: str = PROBLEM
    activation: str = 'smpaf'
    seed: int = 0
    data_seed: int = 0
    network_size: tuple = (2, 256, 256, 1)
    n_coeffs: int = 9
    outer_steps: int = 8000
    inner_steps: int = 40
    lr_params: float = 1e-3
    lr_coeffs: float = 1e-4
    kappa: float = 0.3
    train_sizes: tuple = (512, 128, 128)
    val_sizes: tuple = (256, 64, 64)
    print_every: int = 50
    nx_ref: int = 256
    nt_ref: int = 1001
    pred_batch_size: int = 8192
    smoke: bool = False
    # Problem constants/losses are read directly from the original helpers and
    # included in the identity through their source digest; never overridden.

    def __post_init__(self):
        if self.problem != PROBLEM or self.activation not in ACTIVATIONS:
            raise ValueError('Unknown problem or activation')
        if min(self.outer_steps, self.inner_steps, self.print_every, *self.train_sizes, *self.val_sizes) < 1:
            raise ValueError('Steps and sample sizes must be positive')
        if self.kappa <= 0 or min(self.lr_params, self.lr_coeffs) <= 0:
            raise ValueError('Kappa and learning rates must be positive')

    @property
    def helper_path(self):
        return ROOT / 'pinn_helpers.py'

    def identity(self):
        import jax
        files = [Path(__file__).resolve(), self.helper_path]
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        return {**asdict(self), 'implementation_sha256': digest.hexdigest(),
                'jax_version': importlib.metadata.version('jax'),
                'jaxlib_version': importlib.metadata.version('jaxlib'),
                'numpy_version': importlib.metadata.version('numpy'),
                'precision': 'float32', 'optimizer': 'sgd',
                'jax_x64_enabled': bool(jax.config.x64_enabled),
                'matmul_precision': str(jax.config.jax_default_matmul_precision),
                'hardware': {'backend': jax.default_backend(),
                             'device_kinds': [d.device_kind for d in jax.devices()]},
                'adaptive_update': 'first_order_bilevel',
                'coefficient_key': 'fold_in(PRNGKey(seed), 1)',
                'initialization_scale': 0.5}

    @property
    def run_id(self):
        return hashlib.sha256(json.dumps(self.identity(), sort_keys=True).encode()).hexdigest()[:24]


def configurations(experiment, problem=PROBLEM, seeds=SEEDS, smoke=False):
    base = Config(problem=problem)
    if smoke:
        base = replace(base, smoke=True, outer_steps=2, inner_steps=40,
                       network_size=(2, 4, 4, 1), train_sizes=(4, 3, 3),
                       val_sizes=(3, 2, 2), print_every=1, nx_ref=16, nt_ref=101)
    if experiment == 'E1':
        variants = [replace(base, activation=a) for a in ACTIVATIONS]
    elif experiment == 'E2':
        variants = [replace(base, inner_steps=v) for v in (1, 5, 20, 40)]
    elif experiment == 'E3':
        variants = [replace(base, kappa=v) for v in (1e-4, 1e-2, 0.3, 1.)]
    elif experiment == 'E4':
        variants = [replace(base, lr_coeffs=v) for v in (1e-5, 1e-4, 5e-4, 1e-3)]
    else:
        raise ValueError(experiment)
    return [replace(c, seed=seed) for c in variants for seed in seeds]

# BACKEND
from functools import lru_cache
import importlib.util
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
import jax.numpy as jnp
import numpy as np


@lru_cache(maxsize=32)
def backend(problem, activation, kappa):
    path = Config(problem=problem).helper_path
    spec = importlib.util.spec_from_file_location(f'_pinn_{problem}_{activation}_{kappa}', path)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    # Module-local binding: existing forward_adaptive calls the exact original
    # scalar function with the requested kappa. No global cross-run mutation.
    scalar = h.smpaf_scalar
    h.smpaf_scalar = lambda y, c: scalar(y, c, kappa=kappa)
    if activation == 'relu':
        def forward_relu(X, params):
            value = h.normalize_inputs(X) if problem == 'kdv' else X
            for i, (w, b) in enumerate(params):
                value = value @ w + b
                if i < len(params)-1:
                    value = jax.nn.relu(value)
            return value
        # Reuse the entire existing fixed-activation PDE/loss/update pipeline.
        h.forward_tanh = forward_relu
    adaptive = activation in ('pau', 'smpaf')
    if adaptive:
        loss = jax.jit(lambda p, c, b: h.adaptive_loss_components(p, c, b, activation))
        update = h.make_adaptive_update(activation)
        forward = jax.jit(lambda X, p, c: h.forward_adaptive(X, p, c, activation))
    else:
        loss = jax.jit(lambda p, c, b: h.tanh_loss_components(p, b))
        update = h.update_tanh
        forward = jax.jit(lambda X, p, c: h.forward_tanh(X, p))
    return h, loss, update, forward


def initialize(config, h):
    # Weight initializer uses its original PRNGKey(seed) and Xavier procedure.
    params = h.init_network_params(config.network_size, seed=config.seed)
    coeffs = (h.init_activation_coeffs(config.network_size, config.n_coeffs,
              key=jax.random.fold_in(jax.random.PRNGKey(config.seed), 1))
              if config.activation in ('pau', 'smpaf') else [])
    return params, coeffs


def datasets(config, h):
    return (h.make_pinn_points(*config.train_sizes, seed=config.data_seed),
            h.make_pinn_points(*config.val_sizes, seed=config.data_seed+1))


def count(tree):
    return sum(int(v.size) for v in jax.tree_util.tree_leaves(tree))


@jax.jit
def finite_tree(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in leaves]))


def sync(tree):
    return jax.block_until_ready(tree)


def flatten_values(result):
    total, components = result
    return [float(v) for v in (total, *components)]


def restore(h, path, config):
    with np.load(path) as z:
        params = [(jnp.asarray(z[f'param_W_{i}']), jnp.asarray(z[f'param_b_{i}']))
                  for i in range(len(config.network_size)-1)]
        coeffs = [jnp.asarray(z[f'coeffs_{i}']) for i in range(len(config.network_size)-2)] if config.activation in ('pau', 'smpaf') else []
        return params, coeffs, int(z['completed_steps']), float(z['training_seconds']), {
            k[5:]: z[k].tolist() for k in z.files if k.startswith('hist_')}

# RUNNER
import hashlib
import fcntl
import json
import os
from pathlib import Path
from time import perf_counter
import traceback
import jax
import numpy as np

EXPECTED = ('config.json', 'metrics.json', 'loss_history.npz', 'predictions.npz',
            'parameters.npz', 'datasets.npz', 'initial_parameters.npz')


def json_save(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    os.replace(temp, path)


def npz_save(path, **values):
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **values)
    os.replace(temp, path)


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete(path, identity):
    try:
        marker = json.loads((path/'complete.json').read_text())
        return (json.loads((path/'config.json').read_text())['configuration'] == json.loads(json.dumps(identity))
                and json.loads((path/'metrics.json').read_text())['status'] == 'success'
                and all((path/f).is_file() and file_digest(path/f) == marker['sha256'][f] for f in EXPECTED))
    except (OSError, ValueError, KeyError):
        return False


def run(config, root, experiment, checkpoint_every=50, retry_failed=False, stop_after=None):
    """POSIX advisory lock prevents concurrent writers for an identical run."""
    locks = Path(root)/('smoke' if config.smoke else 'full')/'locks'
    locks.mkdir(parents=True, exist_ok=True)
    with (locks/f'{config.run_id}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f'WAIT for concurrent run {config.run_id}', flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        return _run(config, root, experiment, checkpoint_every, retry_failed, stop_after)


def _run(config, root, experiment, checkpoint_every=50, retry_failed=False, stop_after=None):
    """stop_after is an interruption hook used to validate checkpoint recovery."""
    root = Path(root)/('smoke' if config.smoke else 'full')
    path = root/'runs'/config.run_id
    path.mkdir(parents=True, exist_ok=True)
    membership = root/experiment
    membership.mkdir(parents=True, exist_ok=True)
    json_save(membership/f'{config.run_id}.json', {
        'experiment': experiment, 'run_id': config.run_id,
        'configuration': config.identity(), 'run_path': f'../runs/{config.run_id}'})
    if complete(path, config.identity()):
        print(f'REUSE {experiment} {config.activation} seed={config.seed} {config.run_id}', flush=True)
        return path
    if (path/'failure.json').exists() and not retry_failed:
        print(f'SKIP failed {config.run_id}; use --retry-failed', flush=True)
        return path
    h = None
    params, coeffs = [], []
    history = {'outer_step': [], 'inner_updates': [], 'coefficient_updates': []}
    for prefix in ('train', 'val'):
        history.update({f'{prefix}_{key}': [] for key in ('total', 'pde', 'ic', 'bc')})
    elapsed, completed = 0., 0
    start = None
    best_val = float('inf')
    try:
        if jax.config.x64_enabled:
            raise ValueError('Original experiments use JAX float32. Unset JAX_ENABLE_X64.')
        h, loss, update, forward = backend(config.problem, config.activation, config.kappa)
        params, coeffs = initialize(config, h)
        initial = {**h.params_to_npz(params), **h.coeffs_to_npz(coeffs)}
        train_np, val_np = datasets(config, h)
        data = {f'{prefix}_{k}': v for prefix, d in [('train', train_np), ('val', val_np)] for k,v in d.items()}
        if not (path/'checkpoint.npz').exists():
            npz_save(path/'initial_parameters.npz', **initial)
            npz_save(path/'datasets.npz', **data)
        problem_constants = {key: getattr(h,key) for key in (
            'X_MIN','X_MAX','T_MIN','T_MAX','LAMBDA_F','LAMBDA_IC','LAMBDA_BC')}
        for key in ('ALPHA','NU','SPEED','BETA','X0'):
            if hasattr(h,key):
                problem_constants[key] = float(getattr(h,key))
        json_save(path/'config.json', {'configuration': config.identity(), 'run_id': config.run_id,
                  'problem_constants': problem_constants, 'backend': jax.default_backend(),
                  'devices': [str(d) for d in jax.devices()], 'experiment_first_requested': experiment,
                  'reference': 'Rusanov + central diffusion' if config.problem=='burgers' else 'exact single soliton',
                  'weights_biases': count(params), 'activation_coefficients': count(coeffs),
                  'total_parameters': count(params)+count(coeffs)})
        if (path/'checkpoint.npz').exists():
            params, coeffs, completed, elapsed, history = restore(h, path/'checkpoint.npz', config)
        train, val = h.to_jax_batch(train_np), h.to_jax_batch(val_np)
        adaptive = config.activation in ('pau','smpaf')

        def checkpoint():
            npz_save(path/'checkpoint.npz', **h.params_to_npz(params), **h.coeffs_to_npz(coeffs),
                     completed_steps=completed, training_seconds=elapsed,
                     **{f'hist_{k}': np.asarray(v) for k,v in history.items()})

        def measure(step):
            tr = flatten_values(sync(loss(params, coeffs, train)))
            va = flatten_values(sync(loss(params, coeffs, val)))
            if not np.all(np.isfinite(tr+va)):
                raise FloatingPointError(f'Non-finite train/validation loss at outer step {step}: {tr}, {va}')
            history['outer_step'].append(step)
            history['inner_updates'].append(step*config.inner_steps)
            history['coefficient_updates'].append(step if adaptive else 0)
            for prefix, values in [('train',tr),('val',va)]:
                for key, value in zip(('total','pde','ic','bc'),values):
                    history[f'{prefix}_{key}'].append(value)
            print(f'{experiment} {config.problem} {config.activation} seed={config.seed} '
                  f'outer={step} updates={step*config.inner_steps} train={tr[0]:.5e} val={va[0]:.5e}', flush=True)

        # Compile without counting compilation as optimization time. No dry update
        # is applied to model state. Subsequent seeds reuse the cached executables.
        compile_start = perf_counter()
        loss.lower(params, coeffs, train).compile()
        if adaptive:
            update.lower(params, coeffs, train, val, lrp=config.lr_params,
                         lrc=config.lr_coeffs, inner_steps=config.inner_steps).compile()
        else:
            update.lower(params, train, lr=config.lr_params, inner_steps=config.inner_steps).compile()
        sync(finite_tree((params,coeffs)))
        compilation_seconds = perf_counter()-compile_start
        sync((params,coeffs,train,val))
        if not history['outer_step']:
            measure(completed)
        best_val = min(history['val_total'])
        start = perf_counter()
        for step in range(completed+1, config.outer_steps+1):
            if adaptive:
                params, coeffs = update(params, coeffs, train, val, lrp=config.lr_params,
                                        lrc=config.lr_coeffs, inner_steps=config.inner_steps)
            else:
                params = update(params, train, lr=config.lr_params, inner_steps=config.inner_steps)
            sync((params,coeffs))
            completed = step
            if not bool(finite_tree((params,coeffs))):
                raise FloatingPointError(f'Non-finite parameters/gradient update at outer step {step}')
            if step % config.print_every == 0 or step == config.outer_steps:
                measure(step)
                best_val = min(best_val,history['val_total'][-1])
            if step % checkpoint_every == 0 or step == config.outer_steps or step == stop_after:
                sync((params,coeffs))
                elapsed += perf_counter()-start
                start = None
                checkpoint()
                if step == stop_after:
                    return path
                start = perf_counter()
        sync((params,coeffs))
        elapsed += perf_counter()-start
        start = None
        if config.problem=='burgers':
            x,t,reference = h.solve_burgers_rusanov(nx=config.nx_ref,nt=config.nt_ref)
        else:
            x,t,reference = h.reference_solution(nx=config.nx_ref,nt=config.nt_ref)
        T,X = np.meshgrid(t,x,indexing='ij')
        inputs = np.stack([X.ravel(),T.ravel()],axis=1).astype(np.float32)
        prediction = h.predict_batched(lambda xb: forward(xb,params,coeffs),inputs,config.pred_batch_size).reshape(reference.shape)
        error = float(np.linalg.norm(prediction-reference)/np.linalg.norm(reference))
        if not np.isfinite(error) or not np.all(np.isfinite(prediction)):
            raise FloatingPointError('Non-finite reference-grid predictions or relative L2')
        metrics = {'status':'success','numerical_stability':'finite', 'relative_l2':error,
                   'training_seconds':elapsed, 'compilation_seconds_this_process':compilation_seconds,
                   'best_validation_loss':best_val, 'completed_outer_steps':completed,
                   'total_inner_updates':completed*config.inner_steps,
                   'total_coefficient_updates':completed if adaptive else 0,
                   'lr_ratio':config.lr_coeffs/config.lr_params if adaptive else None,
                   'weights_biases':count(params),'activation_coefficients':count(coeffs),
                   'total_parameters':count(params)+count(coeffs)}
        metrics.update({k:v[-1] for k,v in history.items() if k.startswith(('train_','val_'))})
        npz_save(path/'loss_history.npz', **{k:np.asarray(v) for k,v in history.items()})
        npz_save(path/'predictions.npz',x_grid=x,t_grid=t,U_reference=reference,U_prediction=prediction)
        npz_save(path/'parameters.npz',**h.params_to_npz(params),**h.coeffs_to_npz(coeffs))
        json_save(path/'metrics.json', metrics)
        json_save(path/'complete.json',{'sha256':{f:file_digest(path/f) for f in EXPECTED}})
        if (path/'failure.json').exists():
            (path/'failure.json').unlink()
        return path
    except Exception as exc:
        if start is not None:
            sync((params,coeffs))
            elapsed += perf_counter()-start
        failure = {'status':'failed','numerical_stability':'nonfinite' if isinstance(exc,FloatingPointError) else 'error',
                   'error':str(exc),'traceback':traceback.format_exc(),'completed_outer_steps':completed,
                   'training_seconds':elapsed, 'configuration':config.identity()}
        json_save(path/'failure.json',failure)
        failures = root/'failures'/config.run_id
        failures.mkdir(parents=True, exist_ok=True)
        json_save(failures/'failure.json',failure)
        npz_save(failures/'loss_history.npz',**{k:np.asarray(v) for k,v in history.items()})
        if h is not None:
            npz_save(failures/'parameters.npz',**h.params_to_npz(params),**h.coeffs_to_npz(coeffs))
        print(f'FAILED {config.run_id}: {exc}',flush=True)
        return path

# METRICS
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def read_runs(root, experiment):
    rows = []
    for member in sorted((Path(root)/experiment).glob('*.json')):
        entry = json.loads(member.read_text())
        path = Path(root)/'runs'/entry['run_id']
        cfg = entry['configuration']
        row = {'experiment':experiment,'run_id':entry['run_id'], 'path':str(path), **cfg}
        if complete(path,cfg):
            row.update(json.loads((path/'metrics.json').read_text()))
        elif (path/'failure.json').exists():
            failure = json.loads((path/'failure.json').read_text())
            row.update(status='failed', numerical_stability=failure['numerical_stability'],error=failure['error'])
        else:
            row.update(status='incomplete',numerical_stability='incomplete')
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()} for row in rows)


def group_key(row):
    # All identity fields except initialization seed. Distinct smoke modes,
    # implementations, environments or budgets must never be pooled.
    return json.dumps({k:v for k,v in row.items() if k in (
        'problem','activation','data_seed','network_size','n_coeffs','outer_steps','inner_steps',
        'lr_params','lr_coeffs','kappa','train_sizes','val_sizes','print_every','nx_ref',
        'nt_ref','pred_batch_size','smoke','implementation_sha256','jax_version',
        'jaxlib_version','numpy_version','precision','optimizer','adaptive_update',
        'coefficient_key','initialization_scale','hardware','jax_x64_enabled','matmul_precision')},sort_keys=True)


def aggregate(root, experiment):
    rows = read_runs(root,experiment)
    output = Path(root)/'summary'
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/f'{experiment}_runs.csv',rows)
    groups = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    summaries=[]
    for key, members in groups.items():
        valid = [r for r in members if r['status']=='success']
        out = {**json.loads(key),'experiment':experiment,
               'seeds_requested':';'.join(str(r['seed']) for r in members),
               'n_requested':len(members),'n_success':len(valid),
               'n_failed':sum(r['status']=='failed' for r in members),
               'n_incomplete':sum(r['status']=='incomplete' for r in members)}
        for metric in ('relative_l2','training_seconds'):
            values = [r[metric] for r in valid]
            out[f'{metric}_mean'] = float(np.mean(values)) if values else None
            out[f'{metric}_std'] = float(np.std(values,ddof=1)) if len(values)>1 else None
            if metric=='relative_l2':
                for label, fn in [('median',np.median),('min',np.min),('max',np.max)]:
                    out[f'{metric}_{label}'] = float(fn(values)) if values else None
        summaries.append(out)
    write_csv(output/f'{experiment}_summary.csv',summaries)
    return rows, summaries

# PLOTTING
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pade_af_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':0.25,
                     'savefig.bbox':'tight','pdf.fonttype':42,'ps.fonttype':42})


def generate(root, experiment, rows):
    output=Path(root)/'summary'/f'{experiment}_figures'
    output.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    if not groups:
        return
    axis_key={'E1':'activation','E2':'inner_steps','E3':'kappa','E4':'lr_coeffs'}[experiment]
    groups=sorted(groups.values(),key=lambda g: g[0][axis_key])
    labels=[str(g[0][axis_key]) for g in groups]
    valid_groups=[[r for r in g if r['status']=='success'] for g in groups]

    def save(fig,name):
        for ext in ('pdf','png'):
            fig.savefig(output/f'{name}.{ext}',dpi=300)
        plt.close(fig)

    def comparison(metric,title,ylabel,name):
        fig,ax=plt.subplots(figsize=(5.4,3.5))
        for i,valid in enumerate(valid_groups):
            values=[r[metric] for r in valid]
            if not values:
                continue
            x=i if experiment=='E1' or metric=='training_seconds' else float(labels[i])
            ax.errorbar(x,np.mean(values),yerr=np.std(values,ddof=1) if len(values)>1 else 0,
                        fmt='o',capsize=4,color='C0')
            ax.scatter(np.full(len(values),x),values,s=12,color='C0',alpha=.45)
        categorical=experiment=='E1' or metric=='training_seconds'
        if categorical:
            ax.set_xticks(range(len(labels)),labels)
        elif experiment in ('E3','E4'):
            ax.set_xscale('log')
        ax.set(xlabel={'activation':'Activation','inner_steps':'Inner steps','kappa':'Kappa',
                       'lr_coeffs':'Coefficient learning rate'}[axis_key],ylabel=ylabel,
               title=title+' (successful runs; mean ± sample SD)')
        save(fig,name)

    comparison('relative_l2','Relative L2 error','Relative L2 error','relative_l2')
    if experiment in ('E1','E4'):
        comparison('training_seconds','Training time','Time (s)','training_time')
    # Mean/SD at identical actual step indices; never interpolate or smooth.
    for xkey,xlabel in [('outer_step','Outer steps'),('inner_updates','Cumulative weight updates')]:
        fig,ax=plt.subplots(figsize=(5.4,3.5))
        for label,valid in zip(labels,valid_groups):
            histories=[]
            for row in valid:
                with np.load(Path(row['path'])/'loss_history.npz') as data:
                    histories.append((data[xkey],data['train_total']))
            if not histories:
                continue
            common=sorted(set.intersection(*(set(x.tolist()) for x,y in histories)))
            if not common:
                continue
            values=np.array([[dict(zip(x,y))[step] for step in common] for x,y in histories])
            mean=np.mean(values,axis=0)
            std=np.std(values,axis=0,ddof=1) if len(values)>1 else np.zeros_like(mean)
            line,=ax.plot(common,mean,label=f'{label} (n={len(valid)})')
            ax.fill_between(common,np.maximum(mean-std,np.finfo(float).tiny),mean+std,
                            alpha=.16,color=line.get_color())
        ax.set(xlabel=xlabel,ylabel='Training total loss',yscale='log')
        if ax.lines:
            ax.legend()
        save(fig,f'convergence_{xkey}')
    if experiment=='E1':
        fig,ax=plt.subplots(figsize=(5.4,3.5))
        for j,key in enumerate(('train_pde','train_ic','train_bc')):
            means=[np.mean([r[key] for r in g]) if g else np.nan for g in valid_groups]
            std=[np.std([r[key] for r in g],ddof=1) if len(g)>1 else 0 for g in valid_groups]
            ax.bar(np.arange(len(labels))+(j-1)*.24,means,width=.24,yerr=std,capsize=3,label=key[6:].upper())
        ax.set_xticks(range(len(labels)),labels)
        ax.set(ylabel='Final unweighted loss component',yscale='log',xlabel='Activation')
        ax.legend()
        save(fig,'loss_components')
    if experiment=='E2':
        fig,ax=plt.subplots(figsize=(5.4,3.5))
        for label,valid in zip(labels,valid_groups):
            ax.scatter([r['training_seconds'] for r in valid],[r['relative_l2'] for r in valid],label=f'Inner steps={label}')
        ax.set(xlabel='Training time (s)',ylabel='Relative L2 error')
        ax.legend()
        save(fig,'error_vs_time')
    if experiment=='E3':
        fig,ax=plt.subplots(figsize=(5.4,3.5))
        bottom=np.zeros(len(groups))
        for status in ('success','failed','incomplete'):
            values=np.array([sum(r['status']==status for r in g) for g in groups])
            ax.bar(range(len(groups)),values,bottom=bottom,label=status)
            bottom+=values
        ax.set_xticks(range(len(labels)),labels)
        ax.set(xlabel='Kappa',ylabel='Run count')
        ax.legend()
        save(fig,'stability')
    # Save raw failed trajectories separately, including all finite samples before
    # failure. Never silently turn them into successful convergence curves.
    failed=[r for r in rows if r['status']=='failed']
    if failed:
        fig,ax=plt.subplots(figsize=(5.4,3.5))
        for row in failed:
            path=Path(root)/'failures'/row['run_id']/'loss_history.npz'
            if path.exists():
                with np.load(path) as data:
                    ax.plot(data['outer_step'],data['train_total'],label=f"{row[axis_key]}, seed={row['seed']}")
        ax.set(xlabel='Outer steps',ylabel='Training total loss before failure',yscale='log')
        if ax.lines:
            ax.legend(fontsize=7)
        save(fig,'failed_trajectories')

# VALIDATE
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np
import jax
import jax.numpy as jnp


def assert_arrays_equal(a,b):
    aa,bb=jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)
    assert len(aa)==len(bb)
    for x,y in zip(aa,bb):
        np.testing.assert_array_equal(x,y)


def validate():
    with tempfile.TemporaryDirectory(prefix='pinn-validation-') as temp:
        output=Path(temp)
        for problem in (PROBLEM,):
            base=Config(problem=problem)
            suites={e:configurations(e,problem) for e in ('E1','E2','E3','E4')}
            assert all(len(v)==20 for v in suites.values())
            assert len({c.run_id for values in suites.values() for c in values})==65
            for seed in (5,6,7,8,9):
                baseline=replace(base,seed=seed)
                for e in ('E1','E2','E3','E4'):
                    assert baseline in suites[e]
            # Each ablation modifies only its intended field relative to baseline.
            for e,field in [('E2','inner_steps'),('E3','kappa'),('E4','lr_coeffs')]:
                for c in configurations(e,problem,seeds=(5,)):
                    assert {k for k,v in asdict(c).items() if v!=asdict(base)[k]} <= {field}
            shared_weights=None
            shared_data=None
            for activation in ('tanh','relu','pau','smpaf'):
                cfg=replace(base,activation=activation)
                h,loss,update,forward=backend(problem,activation,cfg.kappa)
                p,c=initialize(cfg,h)
                assert count(p)==21477
                assert count(c)==(2628 if activation in ('pau','smpaf') else 0)
                if shared_weights is None:
                    shared_weights=p
                    shared_data=datasets(cfg,h)
                assert_arrays_equal(p,shared_weights)
                assert_arrays_equal(datasets(cfg,h),shared_data)
                assert_arrays_equal(datasets(replace(cfg,seed=9),h),shared_data)
                if activation=='relu':
                    z=jnp.array([.123,.456])
                    # Unmodified ReLU's classical second derivative is zero a.e.
                    u=lambda zz: forward(zz[None,:],p,c)[0,0]
                    assert float(jax.grad(lambda zz:jax.grad(u)(zz)[0])(z)[0])==0.
                # Numerically compare preserved losses/updates to an independent
                # original helper instance with only the requested kappa binding.
                if activation in ('tanh','smpaf'):
                    spec=importlib.util.spec_from_file_location('_validation_original',cfg.helper_path)
                    original=importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(original)
                    if activation=='smpaf':
                        scalar=original.smpaf_scalar
                        original.smpaf_scalar=lambda y,c:scalar(y,c,kappa=.3)
                    small=replace(configurations('E1',problem,seeds=(5,),smoke=True)[0],activation=activation)
                    pp,cc=initialize(small,h)
                    b=h.to_jax_batch(datasets(small,h)[0])
                    current=loss(pp,cc,b)
                    expected=(original.tanh_loss_components(pp,b) if activation=='tanh' else
                              original.adaptive_loss_components(pp,cc,b,'smpaf'))
                    for a,bv in zip(jax.tree_util.tree_leaves(current),jax.tree_util.tree_leaves(expected)):
                        np.testing.assert_allclose(a,bv,rtol=3e-5,atol=1e-6)
            print(f'PASS {problem}: matrix, counts, paired initialization, fixed data, ReLU derivatives, original loss equivalence',flush=True)
            smoke=configurations('E1',problem,seeds=(5,),smoke=True)
            paths=[]
            for cfg in smoke:
                path=run(cfg,output,'E1',checkpoint_every=1)
                assert complete(path,cfg.identity()),(path,(path/'failure.json').read_text() if (path/'failure.json').exists() else '')
                assert all((path/name).exists() for name in EXPECTED)
                metrics=json.loads((path/'metrics.json').read_text())
                assert metrics['numerical_stability']=='finite'
                with np.load(path/'loss_history.npz') as hist:
                    np.testing.assert_array_equal(hist['outer_step'],[0,1,2])
                    np.testing.assert_array_equal(hist['inner_updates'],[0,40,80])
                paths.append(path)
            # Shared parameters and datasets are serialized identically too.
            with np.load(paths[0]/'initial_parameters.npz') as first:
                for path in paths[1:]:
                    with np.load(path/'initial_parameters.npz') as other:
                        for key in first.files:
                            np.testing.assert_array_equal(first[key],other[key])
            baseline=smoke[-1]
            path=paths[-1]
            mtime=(path/'complete.json').stat().st_mtime_ns
            for e in ('E1','E2','E3','E4'):
                assert run(baseline,output,e)==path
                assert (path/'complete.json').stat().st_mtime_ns==mtime
            root=output/'smoke'
            for e in ('E1','E2','E3','E4'):
                rows,summary=aggregate(root,e)
                assert sum(s['n_success'] for s in summary)==(4 if e=='E1' else 1)
                generate(root,e,rows)
                assert list((root/'summary'/f'{e}_figures').glob('*.pdf'))
                assert list((root/'summary'/f'{e}_figures').glob('*.png'))
            # Checkpoint recovery reproduces uninterrupted weights and coeffs.
            resume_root=output/'resume'
            interrupted=run(baseline,resume_root,'E1',checkpoint_every=1,stop_after=1)
            assert not complete(interrupted,baseline.identity())
            resumed=run(baseline,resume_root,'E1',checkpoint_every=1)
            assert complete(resumed,baseline.identity())
            with np.load(path/'parameters.npz') as expected, np.load(resumed/'parameters.npz') as actual:
                for key in expected.files:
                    np.testing.assert_array_equal(expected[key],actual[key])
            # Completion marker validates every file, not just config/status.
            prediction=path/'predictions.npz'
            content=prediction.read_bytes()
            prediction.write_bytes(b'broken')
            assert not complete(path,baseline.identity())
            prediction.write_bytes(content)
            assert complete(path,baseline.identity())
            # Non-finite gradients/updates fail locally; the suite remains usable.
            unstable=replace(smoke[0],lr_params=1e30)
            failure=run(unstable,output,'failure_test',checkpoint_every=1)
            assert (failure/'failure.json').exists()
            rows,summary=aggregate(root,'failure_test')
            assert summary[0]['n_failed']==1 and summary[0]['relative_l2_mean'] is None
            assert run(smoke[0],output,'E1')==paths[0]
            print(f'PASS {problem}: four finite smoke runs, persistence, resume, baseline reuse, failure isolation, CSV/PDF/PNG',flush=True)
        print('All validation checks passed. No full 8000-step runs were launched.',flush=True)



# CLI
import argparse
from dataclasses import replace
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description='Reproducible PINN activation experiments')
    parser.add_argument('experiment', nargs='?', choices=('E1','E2','E3','E4','all'),
                        help='Experiment to execute (required unless --validate)')
    parser.add_argument('--validate', action='store_true', help='Run local reduced validation checks')
    parser.add_argument('--seed',type=int,choices=SEEDS,help='Run only this initialization seed')
    parser.add_argument('--activation',choices=ACTIVATIONS)
    parser.add_argument('--inner-steps',type=int)
    parser.add_argument('--kappa',type=float)
    parser.add_argument('--lr-coeffs',type=float)
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    parser.add_argument('--smoke',action='store_true',help='Tiny network/data, two outer steps; separate results namespace')
    parser.add_argument('--dry-run',action='store_true',help='Print configurations without training')
    parser.add_argument('--summarize-only',action='store_true')
    parser.add_argument('--no-plots',action='store_true')
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--checkpoint-every',type=int,default=50)
    args=parser.parse_args()
    if args.validate:
        validate()
        return
    if args.experiment is None:
        parser.error('Select E1, E2, E3, E4 or all; no training was started')
    if args.checkpoint_every<1:
        parser.error('--checkpoint-every must be positive')
    experiments=['E1','E2','E3','E4'] if args.experiment=='all' else [args.experiment]
    for name in experiments:
        configs=configurations(name,PROBLEM,(args.seed,) if args.seed else SEEDS,args.smoke)
        filters={'activation':args.activation,'inner_steps':args.inner_steps,
                 'kappa':args.kappa,'lr_coeffs':args.lr_coeffs}
        configs=[c for c in configs if all(v is None or getattr(c,k)==v for k,v in filters.items())]
        if not configs:
            parser.error(f'No {name} configuration matches the requested filters')
        if args.dry_run:
            for c in configs:
                print(name,c.run_id,c)
            continue
        if not args.summarize_only:
            for config in configs:
                run(config,args.results,name,checkpoint_every=args.checkpoint_every,retry_failed=args.retry_failed)
        root=args.results/('smoke' if args.smoke else 'full')
        rows,_=aggregate(root,name)
        if not args.no_plots:
            generate(root,name,rows)




if __name__ == "__main__":
    main()
